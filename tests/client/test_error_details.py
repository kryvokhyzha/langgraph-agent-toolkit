"""Keep safe HTTP error details across client request formats."""

import json

import httpx
import pytest

from langgraph_agent_toolkit.client import AgentClient, AgentClientError


class ErrorBody(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, chunks, fail=False):
        self.chunks = chunks
        self.fail = fail
        self.read_chunks = 0
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            self.read_chunks += 1
            yield chunk
        if self.fail:
            raise httpx.ReadTimeout("Synthetic error-body timeout")

    async def __aiter__(self):
        for chunk in self:
            yield chunk

    def close(self):
        self.closed = True

    async def aclose(self):
        self.close()


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("route", ["invoke", "stream", "stream_jsonl"])
@pytest.mark.parametrize(
    "status, code, retry_after",
    [(503, "request_capacity_exceeded", "3"), (504, "request_timeout", None)],
)
async def test_http_failure_preserves_details_and_closes_response(mode, route, status, code, retry_after):
    """Callers can separate overload from server timeout without parsing text."""
    body = ErrorBody([json.dumps({"error_code": code, "detail": "private-response-detail"}).encode()])
    responses = []

    def respond(request):
        response = httpx.Response(
            status,
            headers={"Retry-After": retry_after} if retry_after else {},
            stream=body,
            request=request,
        )
        responses.append(response)
        return response

    sync_http = httpx.Client(transport=httpx.MockTransport(respond))
    async_http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    async with AgentClient(
        base_url="http://local.invalid",
        agent="agent",
        get_info=False,
        http_client=sync_http,
        async_http_client=async_http,
    ) as client:
        try:
            with pytest.raises(AgentClientError) as caught:
                call = getattr(client, f"a{route}" if mode == "async" else route)
                result = call({"message": "hello"})
                if route == "invoke":
                    if mode == "async":
                        await result
                elif mode == "async":
                    async for _ in result:
                        pytest.fail("An HTTP failure must not emit a stream value.")
                else:
                    list(result)
            error = caught.value
            assert (error.status_code, error.error_code, error.retry_after) == (status, code, retry_after)
            with pytest.raises(httpx.HTTPStatusError) as original:
                responses[0].raise_for_status()
            assert str(error) == f"Error: {original.value}"
            assert "private-response-detail" not in str(error)
            assert isinstance(error.__cause__, httpx.HTTPStatusError)
            assert body.closed
        finally:
            sync_http.close()
            await async_http.aclose()


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("failure", ["oversized", "read_timeout"])
async def test_stream_error_body_cannot_mask_http_status_or_grow_without_limit(mode, failure):
    if failure == "oversized":
        body = ErrorBody([b"x" * 4096] * 1000)
    else:
        body = ErrorBody([b'{"error_code":'], fail=True)
    retry_after = "Wed, 21 Oct 2037 07:28:00 GMT"

    def respond(request):
        return httpx.Response(503, headers={"Retry-After": retry_after}, stream=body, request=request)

    sync_http = httpx.Client(transport=httpx.MockTransport(respond))
    async_http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    async with AgentClient(
        base_url="http://local.invalid",
        agent="agent",
        get_info=False,
        http_client=sync_http,
        async_http_client=async_http,
    ) as client:
        try:
            with pytest.raises(AgentClientError) as caught:
                if mode == "async":
                    async for _ in client.astream({"message": "hello"}):
                        pytest.fail("An HTTP failure must not emit a stream value.")
                else:
                    list(client.stream({"message": "hello"}))
            error = caught.value
            assert (error.status_code, error.error_code, error.retry_after) == (503, None, retry_after)
            assert body.closed
            if failure == "oversized":
                assert body.read_chunks <= 5
        finally:
            sync_http.close()
            await async_http.aclose()


@pytest.mark.parametrize(
    "payload",
    [b"not-json", b"[]", b'{"error_code": 503}', b'{"error_code":"private body text"}', b"[" * 2000 + b"]" * 2000],
)
def test_invalid_error_payload_keeps_status_without_copying_body(payload):
    def respond(request):
        return httpx.Response(429, content=payload, request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        with AgentClient(base_url="http://local.invalid", agent="agent", get_info=False, http_client=http) as client:
            with pytest.raises(AgentClientError) as caught:
                client.invoke({"message": "hello"})
    assert caught.value.status_code == 429
    assert caught.value.error_code is None
    assert caught.value.retry_after is None
    assert payload.decode() not in str(caught.value)


def test_info_error_keeps_its_existing_message_prefix():
    def respond(request):
        return httpx.Response(401, json={"error_code": "authentication_failed"}, request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        with pytest.raises(AgentClientError) as caught:
            AgentClient(base_url="http://local.invalid", http_client=http)
    assert str(caught.value).startswith("Error getting service info: ")
    assert (caught.value.status_code, caught.value.error_code) == (401, "authentication_failed")


def test_transport_and_legacy_errors_have_no_invented_http_status():
    failure = httpx.ConnectTimeout("Synthetic connect timeout")

    def respond(request):
        raise failure

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        with AgentClient(base_url="http://local.invalid", agent="agent", get_info=False, http_client=http) as client:
            with pytest.raises(AgentClientError) as caught:
                client.invoke({"message": "hello"})
    assert str(caught.value) == "Error: Synthetic connect timeout"
    assert caught.value.__cause__ is failure
    assert (caught.value.status_code, caught.value.error_code, caught.value.retry_after) == (None, None, None)
    legacy = AgentClientError("message", 7)
    assert legacy.args == ("message", 7)
    assert str(legacy) == "('message', 7)"
    assert legacy.status_code is None


def test_sync_trickled_error_checks_budget_before_buffering_a_full_chunk(monkeypatch):
    import langgraph_agent_toolkit.client.client as client_module

    elapsed = 0.0

    class TrickleBody(ErrorBody):
        def __iter__(self):
            nonlocal elapsed
            for chunk in super().__iter__():
                elapsed += 0.5
                yield chunk

    monkeypatch.setattr(client_module, "monotonic", lambda: elapsed)
    body = TrickleBody([b" "] * 1000)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(503, stream=body, request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        with AgentClient(base_url="http://local.invalid", agent="agent", get_info=False, http_client=http) as client:
            with pytest.raises(AgentClientError) as caught:
                list(client.stream({"message": "hello"}))
    assert caught.value.status_code == 503
    assert caught.value.error_code is None
    assert body.read_chunks == 2
    assert body.closed
    assert requests[0].extensions["timeout"]["read"] == 120.0


@pytest.fixture
def stalled_error_server():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(503)
            self.send_header("Content-Length", "100")
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "2")
            self.end_headers()
            self.wfile.write(b'{"error_code":')
            self.wfile.flush()
            release.wait(timeout=5)
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
        assert not thread.is_alive()


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_real_http_error_body_stall_keeps_prompt_status_delivery(monkeypatch, stalled_error_server, mode):
    from time import monotonic

    import langgraph_agent_toolkit.client.client as client_module

    monkeypatch.setattr(client_module, "_ERROR_BODY_TIMEOUT", 0.1)
    async with AgentClient(base_url=stalled_error_server, agent="agent", get_info=False) as client:
        started = monotonic()
        with pytest.raises(AgentClientError) as caught:
            if mode == "async":
                async for _ in client.astream({"message": "hello"}):
                    pytest.fail("An HTTP failure must not emit a stream value.")
            else:
                list(client.stream({"message": "hello"}))
        assert monotonic() - started < 1.5
        assert (caught.value.status_code, caught.value.error_code, caught.value.retry_after) == (503, None, "2")
        assert isinstance(caught.value.__cause__, httpx.HTTPStatusError)
        assert caught.value.__cause__.response.is_closed
