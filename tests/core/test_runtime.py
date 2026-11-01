"""Exercise the shared runtime with real graphs and persistent storage."""

import sys
from types import ModuleType

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import SecretStr

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import get_graph_history
from langgraph_agent_toolkit.agents.components.checkpoint.empty import NoOpSaver
from langgraph_agent_toolkit.core.identity import storage_thread_id
from langgraph_agent_toolkit.core.memory.concurrency import SQLiteConversationCoordinator
from langgraph_agent_toolkit.core.models.transport import current_llm_transport_manager
from langgraph_agent_toolkit.core.runtime import agent_runtime
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper import constants
from langgraph_agent_toolkit.schema.execution import ExecutionRequest
from langgraph_agent_toolkit.service.handler import create_app


def build_graph(checkpointer=None):
    builder = StateGraph(MessagesState)
    builder.add_node("reply", lambda state: {"messages": [AIMessage(content="Saved reply")]})
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile(checkpointer=checkpointer)


@pytest.fixture
def configured_runtime(monkeypatch, tmp_path):
    module = ModuleType("runtime_test_agents")
    module.agent = Agent("runtime-agent", "A deterministic runtime agent.", build_graph())
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(settings, "AGENT_PATHS", ["runtime_test_agents:agent"])
    monkeypatch.setattr(settings, "DEFAULT_AGENT", "runtime-agent")
    monkeypatch.setattr(constants, "_runtime_default_agent", None)
    monkeypatch.setattr(settings, "MEMORY_BACKEND", "sqlite")
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "runtime.sqlite"))
    monkeypatch.setattr(settings, "OBSERVABILITY_BACKEND", None)
    monkeypatch.setattr(settings, "MCP_SERVERS", {})
    monkeypatch.setattr(settings, "MCP_AGENT_SERVERS", {})
    monkeypatch.setattr(settings, "AUTH_SECRET", None)
    monkeypatch.setattr(settings, "AUTH_USERS", {})
    monkeypatch.setattr(settings, "AUTH_MODE", "trusted")
    monkeypatch.setattr(settings, "ENV_MODE", "production")
    monkeypatch.setattr(settings, "BACKGROUND_ENABLED", False)
    return module


async def test_runtime_reopens_sqlite_and_http_reads_the_same_history(configured_runtime, monkeypatch):
    owner = "automation-owner"
    public_thread = "work-item-1"
    storage_key = storage_thread_id(owner, "runtime-agent", public_thread)
    config = {"configurable": {"thread_id": storage_key}}
    # A worker runtime has no HTTP bearer credentials or authentication policy.
    async with agent_runtime() as runtime:
        assert runtime.initialized_agents == ("runtime-agent",)
        assert isinstance(runtime.executor.concurrency, SQLiteConversationCoordinator)
        assert current_llm_transport_manager() is runtime.llm_transport_manager
        first_graph = runtime.executor.get_agent("runtime-agent").graph
        response = await runtime.executor.invoke(
            "runtime-agent",
            {"message": "First input"},
            thread_id=storage_key,
            user_id=owner,
            trace_session_id=public_thread,
        )
        assert response.content == "Saved reply"
        connection = runtime.sqlite_connection
        manager = runtime.llm_transport_manager

    assert current_llm_transport_manager() is None
    with pytest.raises(ValueError, match="no active connection"):
        await connection.execute("SELECT 1")
    with pytest.raises(RuntimeError, match="not open"):
        with manager.bind():
            pass
    assert configured_runtime.agent.graph.checkpointer is None

    async with agent_runtime("runtime_test_agents:agent") as runtime:
        graph = runtime.executor.get_agent("runtime-agent").graph
        assert graph is not first_graph
        assert [message.content for message in await get_graph_history(graph, config)] == [
            "First input",
            "Saved reply",
        ]

    monkeypatch.setattr(settings, "AUTH_SECRET", SecretStr("runtime-test-token"))
    app = create_app()
    async with app.router.lifespan_context(app):
        assert app.state.agent_runtime.executor is app.state.agent_executor
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            history = await client.get(
                "/runtime-agent/history",
                params={"thread_id": public_thread, "user_id": owner},
                headers={"Authorization": "Bearer runtime-test-token"},
            )
            assert history.status_code == 200
            assert [message["content"] for message in history.json()["messages"]] == [
                "First input",
                "Saved reply",
            ]
    assert app.state.agent_runtime is None


@pytest.mark.parametrize("saver_factory", [lambda: False, NoOpSaver, MemorySaver])
async def test_runtime_preserves_explicit_savers(configured_runtime, saver_factory):
    saver = saver_factory()
    configured_runtime.agent.graph = build_graph(checkpointer=saver)
    async with agent_runtime() as runtime:
        graph = runtime.executor.get_agent("runtime-agent").graph
        assert graph.checkpointer is saver
        response = await runtime.executor.invoke("runtime-agent", {"message": "hello"}, thread_id="explicit")
        assert response.content == "Saved reply"


async def test_runtime_rebuilds_graphs_and_closes_transport_after_startup_failure(configured_runtime):
    built = []

    def factory(extra_tools):
        assert extra_tools == []
        built.append(current_llm_transport_manager())
        if len(built) == 2:
            raise RuntimeError("Graph initialization failed")
        return build_graph()

    configured_runtime.agent.graph_factory = factory
    async with agent_runtime() as runtime:
        assert runtime.executor.get_agent("runtime-agent").graph.checkpointer is not None
    with pytest.raises(RuntimeError, match="Graph initialization failed"):
        async with agent_runtime():
            pytest.fail("The failed runtime must not accept work")
    assert len(built) == 2
    assert current_llm_transport_manager() is None
    for manager in built:
        with pytest.raises(RuntimeError, match="not open"):
            with manager.bind():
                pass


@pytest.mark.parametrize("approved", [True, False], ids=["approve", "reject"])
async def test_background_example_resumes_in_standalone_runtime_after_restart(configured_runtime, approved):
    path = "langgraph_agent_toolkit.agents.blueprints.background_example.agent:agent"
    assert settings.AUTH_SECRET is None
    assert settings.BACKGROUND_ENABLED is False
    async with agent_runtime(path) as runtime:
        paused = await runtime.executor.execute(
            "background-example",
            ExecutionRequest(thread_id="review", input={"value": 21}),
            owner_id="automation-owner",
        )
        assert paused.status == "interrupted"
        assert paused.checkpoint_id
        assert len(paused.interrupts) == 1
        assert paused.interrupts[0]["value"] == {"question": "Accept this result?", "value": 42}

    async with agent_runtime(path) as runtime:
        saved = await runtime.executor.inspect_execution("background-example", "review", owner_id="automation-owner")
        assert saved.status == "interrupted"
        assert saved.checkpoint_id == paused.checkpoint_id
        done = await runtime.executor.execute(
            "background-example",
            ExecutionRequest(
                operation="resume",
                thread_id="review",
                expected_checkpoint_id=paused.checkpoint_id,
                resume={paused.interrupts[0]["id"]: approved},
            ),
            owner_id="automation-owner",
        )
        assert done.status == "completed"
        assert done.checkpoint_id != paused.checkpoint_id
        assert done.interrupts == []
        assert done.output["approved"] is approved
        assert done.output["result"] == (42 if approved else None)
