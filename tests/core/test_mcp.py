import asyncio
import builtins
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import SecretStr, ValidationError

from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core import mcp
from langgraph_agent_toolkit.core._base_settings import Settings
from langgraph_agent_toolkit.core.mcp import MCPServerConfig, configure_mcp_agents, load_mcp_tools, merge_tools
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.service.factory import _setting_environment_value
from langgraph_agent_toolkit.service.handler import create_app


@pytest.mark.parametrize(
    "value",
    [
        {"url": "server.py"},
        {"url": "file:///tmp/server.py"},
        {"url": "https://user:password@example.invalid/mcp"},
        {"url": "https://example.invalid/mcp", "command": "python"},
        {"transport": "stdio"},
        {"transport": "stdio", "command": "python", "headers": {"X-Test": "value"}},
        {"url": "https://example.invalid/mcp", "headers_env": {"X-Test": "${BAD}"}},
        {
            "url": "https://example.invalid/mcp",
            "headers": {"Authorization": "example-value"},
            "headers_env": {"authorization": "TEST_MCP_AUTH"},
        },
        {"url": "https://example.invalid/mcp", "headers": {"X-Test": "one", "x-test": "two"}},
        {"url": "https://example.invalid/mcp", "headers_env": {"X-Test": "ONE", "x-test": "TWO"}},
        {"url": "https://example.invalid/mcp", "headers_env": {"invalid name": "ONE"}},
        {"url": "https://example.invalid/mcp", "timeout": 0},
        {"url": "https://example.invalid/mcp", "timeout": float("inf")},
        {"url": "https://example.invalid/mcp", "tool_allowlist": ["echo", "echo"]},
        {"url": "https://example.invalid/mcp", "typo": True},
    ],
)
def test_invalid_server_configuration(value):
    with pytest.raises(ValidationError):
        MCPServerConfig.model_validate(value)


def test_settings_and_worker_environment_preserve_mcp_types(monkeypatch):
    config = {
        "docs": {
            "url": "https://example.invalid/mcp",
            "headers": {"Authorization": "example-value"},
            "tool_allowlist": ["search"],
            "timeout": "12.5",
        }
    }
    monkeypatch.setenv("MCP_SERVERS", json.dumps(config))
    candidate = Settings(_env_file=None)
    assert candidate.MCP_SERVERS["docs"].timeout == 12.5
    assert isinstance(candidate.MCP_SERVERS["docs"].headers["Authorization"], SecretStr)
    assert "example-value" not in repr(candidate.MCP_SERVERS)
    monkeypatch.delenv("MCP_SERVERS")
    monkeypatch.setenv("LANGGRAPH_MCP_SERVERS", _setting_environment_value(candidate.MCP_SERVERS))
    child = Settings(_env_file=None)
    child._apply_langgraph_env_overrides()
    assert child.MCP_SERVERS == candidate.MCP_SERVERS


def test_invalid_server_alias():
    with pytest.raises(ValidationError):
        Settings(MCP_SERVERS={"bad name": {"url": "https://example.invalid/mcp"}}, _env_file=None)


def test_http_client_uses_only_explicit_credentials(monkeypatch):
    pytest.importorskip("fastmcp")
    monkeypatch.setenv("TEST_MCP_AUTH", "example-auth-value")
    monkeypatch.setenv("AUTH_SECRET", "example-service-value")
    config = MCPServerConfig(url="https://example.invalid/mcp", headers_env={"Authorization": "TEST_MCP_AUTH"})
    client = mcp._create_client("docs", config)
    assert client.transport.headers == {"Authorization": "example-auth-value"}
    assert client.mode == "auto"
    monkeypatch.delenv("TEST_MCP_AUTH")
    with pytest.raises(ValueError, match="TEST_MCP_AUTH.*missing"):
        mcp._create_client("docs", config)


def test_stdio_client_closes_process_and_resolves_only_explicit_env(monkeypatch):
    pytest.importorskip("fastmcp")
    monkeypatch.setenv("TEST_MCP_PASSWORD", "example-value")
    client = mcp._create_client(
        "local",
        MCPServerConfig(
            transport="stdio", command="python", args=["server.py"], env_env={"PASSWORD": "TEST_MCP_PASSWORD"}
        ),
    )
    assert client.transport.keep_alive is False
    assert client.transport.env == {"PASSWORD": "example-value"}
    assert client.transport.args == ["server.py"]


async def test_unconfigured_mcp_does_not_import_optional_dependency(monkeypatch):
    original_import = builtins.__import__

    def no_mcp_import(name, *args, **kwargs):
        if name.startswith(("fastmcp", "langchain.mcp")):
            raise ImportError("Optional dependency is absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mcp_import)
    assert await load_mcp_tools({}) == {}
    with pytest.raises(RuntimeError, match=r"Install langgraph-agent-toolkit\[mcp\]"):
        await load_mcp_tools({"docs": MCPServerConfig(url="https://example.invalid/mcp")})


@pytest.fixture
def mcp_server(monkeypatch):
    fastmcp = pytest.importorskip("fastmcp")
    server = fastmcp.FastMCP("test-tools")
    calls = []

    @server.tool
    async def echo(text: str) -> str:
        """Return the supplied text."""
        calls.append(text)
        return text

    @server.tool
    async def other() -> str:
        """Return another value."""
        return "other"

    # Use the real protocol and adapter. Replace only the configured transport.
    monkeypatch.setattr(mcp, "_create_client", lambda name, config: fastmcp.Client(server, name=name, mode=config.mode))
    return server, calls


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_namespaces_allowlists_and_upstream_calls(mcp_server, mode):
    config = MCPServerConfig(url="https://example.invalid/mcp", mode=mode, tool_allowlist=["echo"])
    loaded = await load_mcp_tools({"docs": config, "other": config})
    assert [tool.name for tool in loaded["docs"]] == ["docs_echo"]
    assert [tool.name for tool in loaded["other"]] == ["other_echo"]
    result = await loaded["docs"][0].ainvoke(
        {"name": "docs_echo", "type": "tool_call", "id": "1", "args": {"text": "ok"}}
    )
    assert len(result.content) == 1
    assert result.content[0]["text"] == "ok"
    assert mcp_server[1] == ["ok"]


async def test_missing_allowlisted_tool_fails_startup(mcp_server):
    with pytest.raises(ValueError, match="does not provide.*missing"):
        await load_mcp_tools({"docs": MCPServerConfig(url="https://example.invalid/mcp", tool_allowlist=["missing"])})


async def test_failed_discovery_closes_other_contexts_without_logging_credentials(monkeypatch):
    adapter = pytest.importorskip("langchain.mcp")
    closed = []

    class FakeAdapter:
        def __init__(self, name):
            self.name = name

        async def list_tools(self):
            try:
                if self.name == "bad":
                    raise ValueError("example-private-value")
                await asyncio.sleep(0.01)
                return []
            finally:
                closed.append(self.name)

    monkeypatch.setattr(adapter, "MCPAdapter", FakeAdapter)
    monkeypatch.setattr(mcp, "_create_client", lambda name, config: name)
    config = MCPServerConfig(url="https://example.invalid/mcp")
    with pytest.raises(RuntimeError, match=r"server 'bad' \(ValueError\)") as error:
        await load_mcp_tools({"bad": config, "good": config})
    assert "example-private-value" not in str(error.value)
    assert sorted(closed) == ["bad", "good"]


async def test_discovery_deadline_cancels_and_closes_pending_contexts(monkeypatch):
    adapter = pytest.importorskip("langchain.mcp")
    closed = asyncio.Event()

    class FakeAdapter:
        def __init__(self, client):
            pass

        async def list_tools(self):
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

    monkeypatch.setattr(adapter, "MCPAdapter", FakeAdapter)
    monkeypatch.setattr(mcp, "_create_client", lambda name, config: name)
    with pytest.raises(TimeoutError):
        await load_mcp_tools({"slow": MCPServerConfig(url="https://example.invalid/mcp")}, discovery_timeout=0.02)
    assert closed.is_set()


@tool
def local_echo(text: str) -> str:
    """Return text."""
    return text


def test_duplicate_tools_cannot_silently_replace_local_tools():
    with pytest.raises(ValueError, match="Duplicate agent tool name"):
        merge_tools([local_echo], [local_echo])


@pytest.mark.parametrize(
    "selections,has_factory,error",
    [
        ({"missing": ["docs"]}, True, "unloaded agent"),
        ({"custom": ["missing"]}, True, "unknown server"),
        ({"custom": ["docs", "docs"]}, True, "duplicate servers"),
        ({"custom": ["docs"]}, False, "graph_factory"),
    ],
)
async def test_agent_selection_validated_before_network(monkeypatch, selections, has_factory, error):
    agent = SimpleNamespace(graph_factory=Mock() if has_factory else None)
    executor = Mock()
    executor.get_all_agent_info.return_value = [SimpleNamespace(key="custom")]
    executor.get_agent.return_value = agent
    discover = AsyncMock()
    monkeypatch.setattr(mcp, "load_mcp_tools", discover)
    config = Settings(
        MCP_SERVERS={"docs": {"url": "https://example.invalid/mcp"}}, MCP_AGENT_SERVERS=selections, _env_file=None
    )
    with pytest.raises(ValueError, match=error):
        await configure_mcp_agents(executor, config)
    discover.assert_not_called()


async def test_selected_agent_gets_fresh_graph_and_unselected_agents_stay_offline(monkeypatch, mcp_server):
    module = "langgraph_agent_toolkit.agents.blueprints.create_agent.agent"
    executor = AgentExecutor(f"{module}:react_agent")
    agent = executor.get_agent("create-agent")
    imported_graph = agent.graph
    config = Settings(
        MCP_SERVERS={"docs": {"url": "https://example.invalid/mcp", "tool_allowlist": ["echo"]}},
        MCP_AGENT_SERVERS={"create-agent": ["docs"]},
        _env_file=None,
    )
    await configure_mcp_agents(executor, config)
    assert agent.graph is not imported_graph
    assert "docs_echo" in agent.graph.nodes["tools"].bound.tools_by_name
    assert "docs_echo" not in imported_graph.nodes["tools"].bound.tools_by_name


def test_service_attaches_mcp_tools_before_checkpointer_and_readiness(monkeypatch):
    fastmcp = pytest.importorskip("fastmcp")
    import langgraph_agent_toolkit.agents.blueprints.create_agent.agent as blueprint

    server = fastmcp.FastMCP("service-tools")

    @server.tool
    async def echo(text: str) -> str:
        """Return text."""
        return text

    class ScriptedModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    monkeypatch.setattr(
        blueprint,
        "model",
        ScriptedModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "docs_echo", "args": {"text": "from-mcp"}, "id": "echo"}],
                ),
                AIMessage(content="done"),
            ]
        ),
    )
    monkeypatch.setattr(mcp, "_create_client", lambda name, config: fastmcp.Client(server, name=name))
    monkeypatch.setattr(settings, "AGENT_PATHS", [f"{blueprint.__name__}:react_agent"])
    monkeypatch.setattr(settings, "MCP_SERVERS", {"docs": MCPServerConfig(url="https://example.invalid/mcp")})
    monkeypatch.setattr(settings, "MCP_AGENT_SERVERS", {"create-agent": ["docs"]})
    monkeypatch.setattr(settings, "MEMORY_BACKEND", None)
    monkeypatch.setattr(settings, "OBSERVABILITY_BACKEND", None)
    monkeypatch.setattr(settings, "AUTH_SECRET", None)
    monkeypatch.setattr(settings, "AUTH_USERS", {})
    monkeypatch.setattr(settings, "ENV_MODE", "development")
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200
        response = client.post("/invoke", json={"input": {"message": "echo"}, "user_id": "user", "thread_id": "thread"})
        assert response.status_code == 200, response.text
        agent = app.state.agent_executor.get_agent("create-agent")
        assert agent.graph.checkpointer is not None
    assert app.state.ready is False
