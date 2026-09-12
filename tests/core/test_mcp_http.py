"""Test MCP over HTTP without opening a network socket."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from langgraph_agent_toolkit.core.mcp import MCPServerConfig, load_mcp_tools


@pytest.fixture
async def http_mcp_server(monkeypatch):
    fastmcp = pytest.importorskip("fastmcp")
    httpx = pytest.importorskip("httpx2")
    from fastmcp.client.transports import StreamableHttpTransport

    server = fastmcp.FastMCP("HTTP test server")
    state = SimpleNamespace(
        calls=[],
        requests=[],
        clients=[],
        active_requests=0,
        blocked=asyncio.Event(),
        block_method=None,
        fail_after_method=None,
    )

    @server.tool
    async def echo(text: str) -> str:
        """Return the supplied text."""
        state.calls.append(text)
        return text

    app = server.http_app(path="/mcp", json_response=True, stateless_http=True, allowed_hosts=["mcp.test"])

    async def authorized_app(scope, receive, send):
        headers = Headers(scope=scope)
        if headers.get("authorization") != "Bearer test-mcp-credential":
            await JSONResponse({"error": "Unauthorized"}, status_code=401)(scope, receive, send)
            return
        await app(scope, receive, send)

    class RecordingTransport(httpx.ASGITransport):
        async def handle_async_request(self, request):
            state.active_requests += 1
            try:
                body = await request.aread()
                data = json.loads(body) if body else {}
                method = data.get("method")
                state.requests.append((method, dict(request.headers)))
                if method == state.block_method and method is not None:
                    state.blocked.set()
                    await asyncio.Event().wait()
                response = await super().handle_async_request(request)
                if method == state.fail_after_method and method is not None:
                    await response.aclose()
                    raise httpx.ReadError("The response was lost after the remote operation.", request=request)
                return response
            finally:
                state.active_requests -= 1

    def client_factory(**kwargs):
        client = httpx.AsyncClient(transport=RecordingTransport(app=authorized_app), **kwargs)
        state.clients.append(client)
        return client

    original_init = StreamableHttpTransport.__init__

    def initialize_transport(self, *args, **kwargs):
        kwargs["httpx_client_factory"] = client_factory
        original_init(self, *args, **kwargs)

    # Keep the toolkit's real HTTP client, headers, timeout, and protocol setup.
    monkeypatch.setattr(StreamableHttpTransport, "__init__", initialize_transport)
    started = asyncio.Event()
    stop = asyncio.Event()

    async def run_lifespan():
        # AnyIO requires the lifespan to enter and exit in the same task.
        async with app.router.lifespan_context(app):
            started.set()
            await stop.wait()

    lifespan = asyncio.create_task(run_lifespan())
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        yield state
    finally:
        stop.set()
        await asyncio.wait_for(lifespan, timeout=5)
    assert state.active_requests == 0
    assert state.clients
    assert all(client.is_closed for client in state.clients)


def http_config(**kwargs):
    return MCPServerConfig(
        url="http://mcp.test/mcp",
        headers={"Authorization": "Bearer test-mcp-credential"},
        **kwargs,
    )


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_http_discovery_and_calls_send_configured_credentials(http_mcp_server, monkeypatch, mode):
    monkeypatch.setenv("TEST_HTTP_MCP_CLIENT", "configured-client")
    config = http_config(mode=mode, headers_env={"X-Client": "TEST_HTTP_MCP_CLIENT"})

    loaded = await load_mcp_tools({"remote": config})
    result = await loaded["remote"][0].ainvoke(
        {"name": "remote_echo", "args": {"text": "over-http"}, "id": "echo", "type": "tool_call"}
    )

    assert result.content[0]["text"] == "over-http"
    assert http_mcp_server.calls == ["over-http"]
    assert any(method == "tools/list" for method, _ in http_mcp_server.requests)
    assert sum(method == "tools/call" for method, _ in http_mcp_server.requests) == 1
    assert all(
        headers["authorization"] == "Bearer test-mcp-credential" and headers["x-client"] == "configured-client"
        for _, headers in http_mcp_server.requests
    )
    assert http_mcp_server.active_requests == 0
    assert all(client.is_closed for client in http_mcp_server.clients)


async def test_http_unauthorized_discovery_closes_all_clients(http_mcp_server):
    invalid = MCPServerConfig(
        url="http://mcp.test/mcp",
        headers={"Authorization": "Bearer rejected-example-credential"},
    )

    with pytest.raises(RuntimeError, match="MCP discovery failed for server 'denied'") as error:
        await load_mcp_tools({"accepted": http_config(), "denied": invalid})

    assert "rejected-example-credential" not in str(error.value)
    assert "test-mcp-credential" not in str(error.value)
    assert http_mcp_server.calls == []
    assert any(
        headers["authorization"] == "Bearer rejected-example-credential" for _, headers in http_mcp_server.requests
    )
    assert http_mcp_server.active_requests == 0
    assert all(client.is_closed for client in http_mcp_server.clients)


async def test_http_discovery_timeout_cancels_requests_and_closes_clients(http_mcp_server):
    http_mcp_server.block_method = "tools/list"

    with pytest.raises(TimeoutError):
        await load_mcp_tools({"slow": http_config(timeout=5)}, discovery_timeout=0.5)

    assert http_mcp_server.blocked.is_set()
    assert http_mcp_server.active_requests == 0
    assert all(client.is_closed for client in http_mcp_server.clients)


async def test_http_lost_response_does_not_repeat_a_completed_operation(http_mcp_server):
    loaded = await load_mcp_tools({"remote": http_config()})
    http_mcp_server.fail_after_method = "tools/call"

    with pytest.raises(Exception):
        await loaded["remote"][0].ainvoke(
            {"name": "remote_echo", "args": {"text": "completed-operation"}, "id": "echo", "type": "tool_call"}
        )

    assert http_mcp_server.calls == ["completed-operation"]
    assert sum(method == "tools/call" for method, _ in http_mcp_server.requests) == 1
    assert http_mcp_server.active_requests == 0
    assert all(client.is_closed for client in http_mcp_server.clients)
