"""Test MCP headers, caching, cleanup, and real HTTP connection reuse."""

import asyncio
import json
import socket
from types import SimpleNamespace
from typing import Annotated
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from pydantic import Field
from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from langgraph_agent_toolkit.core.mcp import MCPServerConfig, MCPTransportManager, load_mcp_tools


@pytest.fixture
async def http_mcp_server(monkeypatch, request):
    fastmcp = pytest.importorskip("fastmcp")
    httpx = pytest.importorskip("httpx2")
    from fastmcp.client.transports import StreamableHttpTransport

    server = fastmcp.FastMCP("HTTP test server", cache_ttl=1)
    state = SimpleNamespace(
        calls=[],
        requests=[],
        clients=[],
        active_requests=0,
        blocked=asyncio.Event(),
        block_method=None,
        fail_after_method=None,
        release_tools=asyncio.Event(),
        tools_started=asyncio.Event(),
        tools_finished=[],
    )

    @server.tool
    async def echo(text: str) -> str:
        """Return the supplied text."""
        state.calls.append(text)
        return text

    @server.tool
    async def annotated_echo(text: str, tenant: Annotated[str, Field(json_schema_extra={"x-mcp-header": "Tenant"})]):
        """Return the supplied text for one tenant."""
        return f"{tenant}:{text}"

    @server.tool
    async def wait(text: str) -> str:
        """Wait until both test calls have started."""
        state.calls.append(text)
        if len(state.calls) == 2:
            state.tools_started.set()
        try:
            await state.release_tools.wait()
            return text
        finally:
            state.tools_finished.append(text)

    @server.tool
    async def large() -> str:
        """Return a result that exceeds the default SSE event limit."""
        state.calls.append("large")
        return "x" * 600_000

    app = server.http_app(
        path="/mcp", json_response=getattr(request, "param", True), stateless_http=True, allowed_hosts=["mcp.test"]
    )

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


@pytest.mark.parametrize("http_mcp_server", [False], indirect=True)
@pytest.mark.parametrize("event_limit", [None, 2 * 1024 * 1024])
async def test_http_sse_result_limit_is_bounded_and_configurable(http_mcp_server, event_limit):
    """Read a larger remote result only when its server limit permits it."""
    from mcp.shared.exceptions import MCPError

    loaded = await load_mcp_tools({"remote": http_config(mode="legacy", max_sse_event_size=event_limit)})
    tool = next(tool for tool in loaded["remote"] if tool.name == "remote_large")
    if event_limit is None:
        with pytest.raises(MCPError, match="SSE stream failed:.*1048576 byte limit"):
            await asyncio.wait_for(tool.ainvoke({}), timeout=5)
    else:
        result = await asyncio.wait_for(tool.ainvoke({}), timeout=5)
        assert result[0]["text"] == "x" * 600_000

    assert http_mcp_server.calls == ["large"]
    assert sum(method == "tools/call" for method, _ in http_mcp_server.requests) == 1
    assert http_mcp_server.active_requests == 0
    assert all(client.is_closed for client in http_mcp_server.clients)


async def test_http_annotated_headers_survive_discovery_and_ttl_cache(http_mcp_server):
    loaded = await load_mcp_tools({"remote": http_config()})
    tool = next(tool for tool in loaded["remote"] if tool.name == "remote_annotated_echo")

    for text in ("first", "second"):
        result = await tool.ainvoke({"text": text, "tenant": "example-tenant"})
        assert result[0]["text"] == f"example-tenant:{text}"

    assert sum(method == "tools/list" for method, _ in http_mcp_server.requests) == 1
    call_headers = [headers for method, headers in http_mcp_server.requests if method == "tools/call"]
    assert len(call_headers) == 2
    assert all(headers["mcp-param-tenant"] == "example-tenant" for headers in call_headers)

    await asyncio.sleep(1.1)
    await tool.ainvoke({"text": "after-expiry", "tenant": "example-tenant"})
    assert sum(method == "tools/list" for method, _ in http_mcp_server.requests) == 2


async def test_http_parallel_calls_prime_shared_session_once(http_mcp_server):
    loaded = await load_mcp_tools({"remote": http_config(mode="legacy")})
    tool = next(tool for tool in loaded["remote"] if tool.name == "remote_wait")
    calls = [asyncio.create_task(tool.ainvoke({"text": text})) for text in ("one", "two")]
    try:
        await asyncio.wait_for(http_mcp_server.tools_started.wait(), timeout=5)
        # Legacy sessions do not use TTL caching. This checks shared priming.
        assert sum(method == "tools/list" for method, _ in http_mcp_server.requests) == 2
        http_mcp_server.release_tools.set()
        results = await asyncio.wait_for(asyncio.gather(*calls), timeout=5)
        assert [result[0]["text"] for result in results] == ["one", "two"]
    finally:
        http_mcp_server.release_tools.set()
        for call in calls:
            call.cancel()
        await asyncio.gather(*calls, return_exceptions=True)
    assert http_mcp_server.active_requests == 0
    assert all(client.is_closed for client in http_mcp_server.clients)


async def test_http_cancelled_schema_prime_does_not_block_other_calls(http_mcp_server):
    loaded = await load_mcp_tools({"remote": http_config(mode="legacy")})
    tool = next(tool for tool in loaded["remote"] if tool.name == "remote_echo")
    http_mcp_server.block_method = "tools/list"
    first = asyncio.create_task(tool.ainvoke({"text": "cancelled"}))
    second = None
    try:
        await asyncio.wait_for(http_mcp_server.blocked.wait(), timeout=5)
        second = asyncio.create_task(tool.ainvoke({"text": "survivor"}))
        await asyncio.sleep(0)
        http_mcp_server.block_method = None
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, timeout=5)
        result = await asyncio.wait_for(second, timeout=5)
        assert result[0]["text"] == "survivor"
        assert http_mcp_server.calls == ["survivor"]
    finally:
        first.cancel()
        if second is not None:
            second.cancel()
        await asyncio.gather(*(call for call in (first, second) if call is not None), return_exceptions=True)
    assert http_mcp_server.active_requests == 0
    assert all(client.is_closed for client in http_mcp_server.clients)


async def test_http_cancelled_call_keeps_other_call_connected(http_mcp_server):
    loaded = await load_mcp_tools({"remote": http_config()})
    tool = next(tool for tool in loaded["remote"] if tool.name == "remote_wait")
    cancelled = asyncio.create_task(tool.ainvoke({"text": "cancelled"}))
    survivor = asyncio.create_task(tool.ainvoke({"text": "survivor"}))
    try:
        await asyncio.wait_for(http_mcp_server.tools_started.wait(), timeout=5)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cancelled, timeout=5)
        assert not survivor.done()
        http_mcp_server.release_tools.set()
        result = await asyncio.wait_for(survivor, timeout=5)
        assert result[0]["text"] == "survivor"
    finally:
        http_mcp_server.release_tools.set()
        cancelled.cancel()
        survivor.cancel()
        await asyncio.gather(cancelled, survivor, return_exceptions=True)
    assert sorted(http_mcp_server.tools_finished) == ["cancelled", "survivor"]
    assert http_mcp_server.active_requests == 0
    assert all(client.is_closed for client in http_mcp_server.clients)


@pytest.fixture
async def tcp_mcp_server(monkeypatch):
    fastmcp = pytest.importorskip("fastmcp")
    httpx = pytest.importorskip("httpx2")
    uvicorn = pytest.importorskip("uvicorn")
    state = SimpleNamespace(peers=set(), requests=[], clients=[])
    server = fastmcp.FastMCP("TCP test server", cache_ttl=60)

    @server.tool
    async def echo(text: str) -> str:
        """Return the supplied text."""
        return text

    app = server.http_app(path="/mcp", json_response=True, stateless_http=True, allowed_hosts=["127.0.0.1"])

    async def record_app(scope, receive, send):
        if scope["type"] == "http":
            state.peers.add(tuple(scope["client"]))
            state.requests.append(dict(Headers(scope=scope)))
        await app(scope, receive, send)

    original_client = httpx.AsyncClient

    class RecordingClient(original_client):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            state.clients.append(self)

    monkeypatch.setattr(httpx, "AsyncClient", RecordingClient)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    state.url = f"http://127.0.0.1:{listener.getsockname()[1]}/mcp"
    process = uvicorn.Server(uvicorn.Config(record_app, lifespan="on", log_level="error", timeout_graceful_shutdown=2))
    task = asyncio.create_task(process.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(5):
            while not process.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield state
    finally:
        process.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        listener.close()
    assert all(client.is_closed for client in state.clients)


@pytest.mark.parametrize("managed", [False, True])
async def test_tcp_pool_reuse_and_shutdown(tcp_mcp_server, managed):
    config = MCPServerConfig(url=tcp_mcp_server.url, headers={"Authorization": "Bearer tcp-example"})

    async def exercise(manager=None):
        loaded = await load_mcp_tools({"remote": config}, manager=manager)
        for text in ("one", "two"):
            assert (await loaded["remote"][0].ainvoke({"text": text}))[0]["text"] == text
        return loaded["remote"][0]

    if managed:
        async with MCPTransportManager() as manager:
            tool = await exercise(manager)
            assert len(tcp_mcp_server.peers) == 1
            assert sum(not client.is_closed for client in tcp_mcp_server.clients) == 1
        with pytest.raises(RuntimeError, match="manager is not open"):
            await tool.ainvoke({"text": "after-shutdown"})
    else:
        await exercise()
        assert len(tcp_mcp_server.peers) == 3
    assert all(headers["authorization"] == "Bearer tcp-example" for headers in tcp_mcp_server.requests)
    assert all(client.is_closed for client in tcp_mcp_server.clients)


async def test_tcp_pool_closes_after_discovery_failure(tcp_mcp_server):
    with pytest.raises(ValueError, match="does not provide"):
        async with MCPTransportManager() as manager:
            await load_mcp_tools(
                {"remote": MCPServerConfig(url=tcp_mcp_server.url, tool_allowlist=["missing"])}, manager=manager
            )
    assert tcp_mcp_server.clients
    assert all(client.is_closed for client in tcp_mcp_server.clients)


async def test_tcp_servers_keep_separate_pools_credentials_and_caches(tcp_mcp_server):
    async with MCPTransportManager() as manager:
        tools = await load_mcp_tools(
            {
                name: MCPServerConfig(url=tcp_mcp_server.url, headers={"Authorization": f"Bearer {name}-example"})
                for name in ("first", "second")
            },
            manager=manager,
        )
        assert len(tcp_mcp_server.peers) == 2
        for name in ("first", "second"):
            result = await tools[name][0].ainvoke({"text": name})
            assert result[0]["text"] == name
        assert len(tcp_mcp_server.peers) == 2
        assert sum(headers.get("mcp-method") == "tools/list" for headers in tcp_mcp_server.requests) == 2
        assert {headers["authorization"] for headers in tcp_mcp_server.requests} == {
            "Bearer first-example",
            "Bearer second-example",
        }
    assert all(client.is_closed for client in tcp_mcp_server.clients)


async def test_service_failed_graph_startup_closes_mcp_pool(tcp_mcp_server, monkeypatch):
    from langgraph_agent_toolkit.core.settings import settings
    from langgraph_agent_toolkit.service import handler

    agent = SimpleNamespace(graph_factory=Mock(side_effect=ValueError("Test graph build failed")))
    executor = Mock()
    executor.get_all_agent_info.return_value = [SimpleNamespace(key="custom")]
    executor.get_agent.return_value = agent
    monkeypatch.setattr(handler, "AgentExecutor", lambda *paths: executor)
    monkeypatch.setattr(settings, "MCP_SERVERS", {"remote": MCPServerConfig(url=tcp_mcp_server.url)})
    monkeypatch.setattr(settings, "MCP_AGENT_SERVERS", {"custom": ["remote"]})
    monkeypatch.setattr(settings, "MEMORY_BACKEND", None)
    monkeypatch.setattr(settings, "OBSERVABILITY_BACKEND", None)
    monkeypatch.setattr(settings, "ENV_MODE", "development")
    monkeypatch.setattr(settings, "AUTH_SECRET", None)
    monkeypatch.setattr(settings, "AUTH_USERS", {})
    app = FastAPI()
    with pytest.raises(ValueError, match="Test graph build failed"):
        async with handler.lifespan(app):
            pytest.fail("The service became ready after graph startup failed")
    assert app.state.ready is False
    assert app.state.mcp_transport_manager is None
    assert tcp_mcp_server.clients
    assert all(client.is_closed for client in tcp_mcp_server.clients)
