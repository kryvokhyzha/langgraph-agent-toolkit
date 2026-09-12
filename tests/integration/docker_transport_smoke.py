"""Check both OpenAI transports inside the API image without external access.

Run this script in a container with --network none. Only loopback is required.
The script uses the image's dependencies. It does not require pytest.
"""

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


TEST_KEY = "container-smoke-only"
REPLY = "Container transport ready."
USAGE = {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self):
        super().__init__(("127.0.0.1", 0), ModelHandler)
        self.requests = {"invoke": 0, "stream": 0}
        self.lock = threading.Lock()


class ModelHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if (
            self.path != "/v1/chat/completions"
            or self.headers.get("Authorization") != f"Bearer {TEST_KEY}"
            or not 0 < length <= 8192
        ):
            self.send_error(400)
            return
        request = json.loads(self.rfile.read(length))
        streaming = request.get("stream", False)
        with self.server.lock:
            self.server.requests["stream" if streaming else "invoke"] += 1
        base = {"id": "chatcmpl-container-smoke", "created": 1, "model": "container-smoke-model"}
        usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
        if streaming:
            if not request.get("stream_options", {}).get("include_usage"):
                self.send_error(400)
                return
            events = [
                {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": REPLY}, "finish_reason": None}],
                },
                {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
                {**base, "object": "chat.completion.chunk", "choices": [], "usage": usage},
            ]
            body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
            body += b"data: [DONE]\n\n"
            content_type = "text/event-stream"
        else:
            body = json.dumps(
                {
                    **base,
                    "object": "chat.completion",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": REPLY}, "finish_reason": "stop"}
                    ],
                    "usage": usage,
                }
            ).encode()
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()


def prepare_environment():
    """Disable environment files, external tracing, and inherited proxies."""
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2", "LANGFUSE_TRACING_ENABLED"):
        os.environ[name] = "false"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "OPENAI_PROXY"):
        os.environ.pop(name, None)
        os.environ.pop(name.lower(), None)
    os.environ["MODEL_CONFIGS"] = "{}"
    os.environ["USE_FAKE_MODEL"] = "false"


async def check_transports(server):
    from openai import DefaultAioHttpClient, DefaultAsyncHttpxClient

    from langgraph_agent_toolkit.core._base_settings import Settings
    from langgraph_agent_toolkit.core.models import CompletionModelFactory, LLMTransportManager

    selected = Settings(_env_file=None)
    default_transport = selected.LLM_HTTP_ASYNC_TRANSPORT
    assert default_transport == "aiohttp"
    results = []
    for transport in (default_transport, "httpx"):
        if transport == "httpx":
            os.environ["LLM_HTTP_ASYNC_TRANSPORT"] = transport
            selected = Settings(_env_file=None)
        assert selected.LLM_HTTP_ASYNC_TRANSPORT == transport
        async with LLMTransportManager.from_settings(selected) as manager:
            with manager.bind():

                def model(streaming):
                    return CompletionModelFactory.create(
                        "openai",
                        "container-smoke-model",
                        configurable_fields=(),
                        config_prefix="",
                        model_parameter_values=(),
                        api_key=TEST_KEY,
                        base_url=f"http://127.0.0.1:{server.server_port}/v1",
                        streaming=streaming,
                        disable_streaming=not streaming,
                        stream_usage=True,
                        max_retries=0,
                        timeout=5,
                        use_responses_api=False,
                    )

                nonstream = model(False)
                reply = await nonstream.ainvoke("Check the container transport.")
                assert reply.content == REPLY
                assert {key: reply.usage_metadata[key] for key in USAGE} == USAGE
                stream = model(True)
                combined = None
                async for chunk in stream.astream("Check the container stream."):
                    combined = chunk if combined is None else combined + chunk
                assert combined is not None and combined.content == REPLY
                assert {key: combined.usage_metadata[key] for key in USAGE} == USAGE
                assert len(manager._pools) == 1
                sync_client, async_client = next(iter(manager._pools.values()))
                expected = DefaultAioHttpClient if transport == "aiohttp" else DefaultAsyncHttpxClient
                assert isinstance(async_client, expected)
                assert nonstream.http_async_client is stream.http_async_client is async_client
                assert not sync_client.is_closed and not async_client.is_closed
        assert sync_client.is_closed and async_client.is_closed
        results.append({"transport": transport, "invoke": True, "stream": True, "usage": True, "clients_closed": True})
    assert server.requests == {"invoke": 2, "stream": 2}
    print(
        json.dumps({"default_transport": default_transport, "transports": results, "local_requests": server.requests})
    )


async def bounded_check(server):
    async with asyncio.timeout(45):
        await check_transports(server)


def main():
    prepare_environment()
    server = LocalServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        asyncio.run(bounded_check(server))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


if __name__ == "__main__":
    main()
