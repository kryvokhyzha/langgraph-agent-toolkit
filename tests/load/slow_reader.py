"""Hold local HTTP response bodies to test service backpressure."""

import asyncio
import json
import math
import socket
import time
from contextlib import suppress
from urllib.parse import urlsplit
from uuid import uuid4


def _endpoint(base_url: str) -> tuple[str, int]:
    try:
        parsed = urlsplit(base_url)
        valid = (
            parsed.scheme == "http"
            and parsed.hostname == "127.0.0.1"
            and parsed.port is not None
            and 0 < parsed.port < 65536
            and parsed.path in {"", "/"}
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Use http://127.0.0.1:<port> for the slow-reader test")
    return "127.0.0.1", parsed.port


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        async with asyncio.timeout(0.5):
            await writer.wait_closed()
    except (TimeoutError, OSError):
        writer.transport.abort()
    except asyncio.CancelledError:
        writer.transport.abort()
        raise


async def measure_slow_readers(base_url: str, secret: str, concurrency: int = 2, hold_seconds: float = 3) -> dict:
    """Read response headers and then hold each body without reading it.

    The caller configures the local model's response size before this call.
    All owned sockets close when the call completes or is cancelled.
    """
    host, port = _endpoint(base_url)
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or not 1 <= concurrency <= 128:
        raise ValueError("concurrency must be an integer between 1 and 128")
    if not isinstance(hold_seconds, (int, float)) or not math.isfinite(hold_seconds) or not 0 <= hold_seconds <= 60:
        raise ValueError("hold_seconds must be between 0 and 60")
    if not isinstance(secret, str) or not secret or any(not 32 <= ord(character) <= 126 for character in secret):
        raise ValueError("The test credential must contain printable ASCII characters")

    start = time.monotonic()
    prefix = f"slow-reader-{uuid4().hex}"
    owned_count = 0
    closed_count = 0

    async def measure(index: int) -> dict:
        nonlocal owned_count, closed_count
        request_id = f"{prefix}-{index}"
        result = {"request_id": request_id, "status": None, "headers": {}, "error": None}
        raw = None
        writer = None
        stage = "connect"
        try:
            raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            owned_count += 1
            raw.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            raw.setblocking(False)
            result["receive_buffer_bytes"] = raw.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            async with asyncio.timeout(10):
                await asyncio.get_running_loop().sock_connect(raw, (host, port))
                reader, writer = await asyncio.open_connection(sock=raw, limit=4096)
                body = json.dumps(
                    {
                        "input": {"message": request_id},
                        "thread_id": request_id,
                        "user_id": "load-user",
                        "stream_tokens": True,
                    },
                    separators=(",", ":"),
                ).encode()
                request = (
                    "POST /load-agent/stream/jsonl HTTP/1.1\r\n"
                    f"Host: {host}:{port}\r\n"
                    f"Authorization: Bearer {secret}\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("ascii") + body
                writer.write(request)
                await writer.drain()
                stage = "headers"
                headers = await reader.readuntil(b"\r\n\r\n")
            lines = headers.decode("latin-1").split("\r\n")
            protocol, status, *_ = lines[0].split(" ")
            if protocol != "HTTP/1.1" or not status.isdigit():
                raise ValueError("Invalid HTTP response status")
            result["status"] = int(status)
            for line in lines[1:]:
                if line:
                    name, value = line.split(":", 1)
                    result["headers"][name.strip().lower()] = value.strip()
            result["headers_seconds"] = time.monotonic() - start
            if result["status"] != 200:
                result["error"] = {"type": "UnexpectedHTTPStatus", "stage": "headers"}
                return result
            # Stop transport reads as soon as the complete headers arrive.
            writer.transport.pause_reading()
            stage = "hold"
            await asyncio.sleep(hold_seconds)
            result["held_seconds"] = hold_seconds
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Do not include request headers or credentials in error output.
            result["error"] = {"type": type(exc).__name__, "stage": stage}
        finally:
            try:
                if writer is not None:
                    await _close_writer(writer)
            finally:
                if raw is not None:
                    raw.close()
                    closed_count += 1
        return result

    tasks = [asyncio.create_task(measure(index), name=f"slow-reader-{index}") for index in range(concurrency)]
    try:
        results = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        # Finish socket cleanup before cancellation reaches the caller.
        cleanup = asyncio.gather(*tasks, return_exceptions=True)
        while not cleanup.done():
            with suppress(asyncio.CancelledError):
                await asyncio.shield(cleanup)

    return {
        "requested": concurrency,
        "owned_count": owned_count,
        "closed_count": closed_count,
        "elapsed_seconds": time.monotonic() - start,
        "statuses": [result["status"] for result in results],
        "headers": [result["headers"] for result in results],
        "errors": [result["error"] for result in results if result["error"] is not None],
        "results": results,
    }
