"""Test tool injection into the built-in graph factories."""

import importlib

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import Field

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.core.settings import settings


BLUEPRINTS = ["create_agent", "create_agent_structured", "hitl_agent", "react"]


class ScriptedModel(FakeMessagesListChatModel):
    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def final_message(blueprint):
    if blueprint != "create_agent_structured":
        return AIMessage(content="done")
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ResponseSchema",
                "args": {"response": "done", "alternative_response": "complete"},
                "id": "structured",
                "type": "tool_call",
            }
        ],
    )


def load_blueprint(monkeypatch, blueprint, responses):
    monkeypatch.setattr(settings, "USE_FAKE_MODEL", True)
    module = importlib.import_module(f"langgraph_agent_toolkit.agents.blueprints.{blueprint}.agent")
    model = ScriptedModel(responses=responses)
    if blueprint == "react":
        monkeypatch.setattr(module.CompletionModelFactory, "create", lambda **kwargs: model)
    else:
        monkeypatch.setattr(module, "model", model)
    return module


@pytest.mark.parametrize("blueprint", BLUEPRINTS)
async def test_factory_executes_supplied_async_tool(monkeypatch, blueprint):
    calls = []

    @tool
    async def remote_lookup(key: str) -> str:
        """Read a remote value."""
        calls.append(key)
        return f"value for {key}"

    module = load_blueprint(
        monkeypatch,
        blueprint,
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "remote_lookup", "args": {"key": "one"}, "id": "lookup", "type": "tool_call"}],
            ),
            final_message(blueprint),
        ],
    )
    agent = next(value for value in vars(module).values() if isinstance(value, Agent))
    original_graph = agent.graph
    graph = agent.graph_factory([remote_lookup])

    result = await graph.ainvoke({"messages": [HumanMessage("Read one.")]})

    assert calls == ["one"]
    assert any(
        isinstance(message, ToolMessage) and message.content == "value for one" for message in result["messages"]
    )
    assert agent.graph is original_graph
    assert graph is not original_graph
    assert graph.checkpointer is None
    if blueprint == "create_agent_structured":
        assert result["structured_response"].response == "done"


@pytest.mark.parametrize("blueprint", BLUEPRINTS)
async def test_factory_does_not_retry_a_failed_remote_write(monkeypatch, blueprint):
    calls = []

    @tool
    async def remote_write(value: str) -> str:
        """Write a remote value."""
        calls.append(value)
        raise TimeoutError("The response was lost after the write.")

    module = load_blueprint(
        monkeypatch,
        blueprint,
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "remote_write", "args": {"value": "one"}, "id": "write", "type": "tool_call"}],
            ),
            AIMessage(content="done"),
        ],
    )
    graph = module.build_graph([remote_write])

    with pytest.raises(TimeoutError, match="response was lost"):
        await graph.ainvoke({"messages": [HumanMessage("Write one.")]})

    assert calls == ["one"]


@pytest.mark.parametrize("blueprint", ["create_agent", "create_agent_structured"])
def test_native_factory_preserves_both_local_tool_results(monkeypatch, blueprint):
    """Send both tool results to the model and keep them in graph history."""
    module = load_blueprint(
        monkeypatch,
        blueprint,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "add", "args": {"a": 2, "b": 3}, "id": "sum", "type": "tool_call"},
                    {"name": "multiply", "args": {"a": 2, "b": 3}, "id": "product", "type": "tool_call"},
                ],
            ),
            final_message(blueprint),
        ],
    )

    result = module.build_graph().invoke({"messages": [HumanMessage("Add and multiply two and three.")]})

    for messages in (result["messages"], module.model.seen[-1]):
        tool_results = {
            message.tool_call_id: message.content for message in messages if isinstance(message, ToolMessage)
        }
        assert float(tool_results["sum"]) == 5
        assert float(tool_results["product"]) == 6
    if blueprint == "create_agent_structured":
        assert result["structured_response"].model_dump() == {"response": "done", "alternative_response": "complete"}
    else:
        assert result["messages"][-1].content == "done"


@pytest.mark.parametrize("blueprint", ["create_agent", "create_agent_structured"])
async def test_native_factory_limits_local_retries_and_reports_failure(monkeypatch, blueprint):
    """Retry a local read twice and give the error to the model."""
    from langgraph_agent_toolkit.agents.blueprints.create_agent import _shared

    calls = []

    @tool
    def local_search(query: str) -> str:
        """Read a local search result."""
        calls.append(query)
        raise TimeoutError("search is unavailable")

    module = load_blueprint(
        monkeypatch,
        blueprint,
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "local_search", "args": {"query": "one"}, "id": "search", "type": "tool_call"}],
            ),
            final_message(blueprint),
        ],
    )
    monkeypatch.setattr(_shared, "DuckDuckGoSearchResults", lambda: local_search)

    result = await module.build_graph().ainvoke({"messages": [HumanMessage("Find one.")]})

    assert calls == ["one"] * 3
    error = next(message for message in module.model.seen[-1] if isinstance(message, ToolMessage))
    assert error.tool_call_id == "search"
    assert error.status == "error"
    assert "search is unavailable" in error.content
    assert error in result["messages"]


def test_agent_keeps_the_observability_positional_argument():
    graph = object()
    observability = object()

    agent = Agent("name", "description", graph, observability)

    assert agent.graph is graph
    assert agent.observability is observability
    assert agent.graph_factory is None
