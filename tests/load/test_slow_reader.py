"""Check slow-reader request integrity and socket cleanup over local TCP."""

import asyncio
import json
import time
from contextlib import asynccontextmanager, suppress

import pytest
from slow_reader import measure_slow_readers


@asynccontextmanager
async def local_server(*, send_headers=True, close_before_headers=False):
    requests = []
    tasks = set()
    closed = 0

    async def handle(reader, writer):
        nonlocal closed
        task = asyncio.current_task()
        tasks.add(task)
        try:
            header = (await reader.readuntil(b"\r\n\r\n")).decode("ascii")
            headers = dict(line.split(": ", 1) for line in header.split("\r\n")[1:] if line)
            body = json.loads(await reader.readexactly(int(headers["Content-Length"])))
            requests.append({"headers": headers, "body": body, "request_line": header.split("\r\n")[0]})
            if close_before_headers:
                return
            if send_headers:
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nContent-Length: 1000000\r\n\r\n"
                    b"unread-body"
                )
                await writer.drain()
            await reader.read()
        except (OSError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()
            closed += 1
            tasks.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", requests, lambda: closed
    finally:
        server.close()
        await server.wait_closed()
        for task in list(tasks):
            task.cancel()
        async with asyncio.timeout(1):
            await asyncio.gather(*tasks, return_exceptions=True)


async def wait_for_count(counter, expected):
    async with asyncio.timeout(1):
        while counter() != expected:
            await asyncio.sleep(0.005)


async def test_slow_readers_hold_headers_and_close_every_socket():
    async with local_server() as (url, requests, closed):
        result = await measure_slow_readers(url, "local-test-only", concurrency=3, hold_seconds=0.03)
        await wait_for_count(closed, 3)
    assert result["statuses"] == [200, 200, 200]
    assert result["errors"] == []
    assert result["owned_count"] == result["closed_count"] == 3
    assert result["elapsed_seconds"] >= 0.03
    assert len({request["body"]["thread_id"] for request in requests}) == 3
    for request in requests:
        assert request["request_line"] == "POST /load-agent/stream/jsonl HTTP/1.1"
        assert request["headers"]["Authorization"] == "Bearer local-test-only"
        assert request["body"]["input"]["message"] == request["body"]["thread_id"]
        assert request["body"]["user_id"] == "load-user"
        assert request["body"]["stream_tokens"] is True
    assert "local-test-only" not in json.dumps(result)


@pytest.mark.parametrize("send_headers", [True, False])
async def test_cancellation_closes_all_sockets_during_headers_or_hold(send_headers):
    async with local_server(send_headers=send_headers) as (url, requests, closed):
        task = asyncio.create_task(measure_slow_readers(url, "local-test-only", concurrency=2, hold_seconds=30))
        await wait_for_count(lambda: len(requests), 2)
        await asyncio.sleep(0.01)
        started = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await wait_for_count(closed, 2)
        assert time.monotonic() - started < 1


async def test_header_failure_is_reported_after_socket_cleanup():
    async with local_server(close_before_headers=True) as (url, requests, closed):
        result = await measure_slow_readers(url, "local-test-only", concurrency=1, hold_seconds=0)
        await wait_for_count(closed, 1)
    assert len(requests) == 1
    assert result["owned_count"] == result["closed_count"] == 1
    assert result["statuses"] == [None]
    assert result["errors"] == [{"type": "IncompleteReadError", "stage": "headers"}]


@pytest.mark.parametrize(
    "url",
    ["https://127.0.0.1:9000", "http://localhost:9000", "http://192.0.2.1:9000", "http://127.0.0.1:9000/other"],
)
async def test_slow_reader_rejects_nonlocal_or_ambiguous_endpoints(url):
    with pytest.raises(ValueError, match="127.0.0.1"):
        await measure_slow_readers(url, "local-test-only")


async def test_slow_reader_rejects_header_injection():
    with pytest.raises(ValueError, match="printable ASCII"):
        await measure_slow_readers("http://127.0.0.1:9000", "local\r\nInjected: value")
