from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import StreamWriter

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.agents.blueprints.bg_task_agent.utils import CustomData
from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.helper.utils import langchain_to_chat_message
from langgraph_agent_toolkit.schema.schema import ChatMessage


START_MESSAGE = CustomData(type="start", data={"key1": "value1", "key2": 123})

STATIC_MESSAGES = [
    AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="test_tool",
                args={"arg1": "value1"},
                id="test_call_id",
            ),
        ],
    ),
    ToolMessage(content="42", tool_call_id="test_call_id"),
    AIMessage(content="The answer is 42"),
    CustomData(type="end", data={"time": "end"}).to_langchain(),
]


EXPECTED_OUTPUT_MESSAGES = [langchain_to_chat_message(m) for m in [START_MESSAGE.to_langchain()] + STATIC_MESSAGES]


@pytest.fixture
def httpx_routed_to_app(app):
    """Route httpx.stream through an in-process TestClient so AgentClient.stream hits the real route."""
    # raise_server_exceptions=False so a streaming response isn't re-raised by the test transport
    # (the SSE body is what an HTTP client actually observes).
    client = TestClient(app, raise_server_exceptions=False)  # no `with` -> lifespan does not run

    def _stream(method, url, **kwargs):
        path = url.replace("http://0.0.0.0:8080", "")
        return client.stream(method, path, **kwargs)

    with patch("httpx.stream", _stream):
        yield


def test_messages_conversion() -> None:
    """Verify that our list of messages is converted to the expected output."""
    messages = EXPECTED_OUTPUT_MESSAGES

    # Verify the sequence of messages
    assert len(messages) == 5

    # First message: Custom data start marker
    assert messages[0].type == "custom"
    assert messages[0].custom_data == {"key1": "value1", "key2": 123}

    # Second message: AI with tool call
    assert messages[1].type == "ai"
    assert len(messages[1].tool_calls) == 1
    assert messages[1].tool_calls[0]["name"] == "test_tool"
    assert messages[1].tool_calls[0]["args"] == {"arg1": "value1"}

    # Third message: Tool response
    assert messages[2].type == "tool"
    assert messages[2].content == "42"
    assert messages[2].tool_call_id == "test_call_id"

    # Fourth message: Final AI response
    assert messages[3].type == "ai"
    assert messages[3].content == "The answer is 42"

    # Fifth message: Custom data end marker
    assert messages[4].type == "custom"
    assert messages[4].custom_data == {"time": "end"}


async def static_messages(state: MessagesState, writer: StreamWriter) -> MessagesState:
    START_MESSAGE.dispatch(writer)
    return {"messages": STATIC_MESSAGES}


agent = StateGraph(MessagesState)
agent.add_node("static_messages", static_messages)
agent.set_entry_point("static_messages")
agent.add_edge("static_messages", END)
static_agent = agent.compile(checkpointer=MemorySaver())


def test_agent_stream(app, httpx_routed_to_app):
    """End-to-end: AgentClient.stream -> real route -> real AgentExecutor.stream -> static_agent -> SSE -> parse.

    The custom START/END markers (StreamWriter) plus the three returned messages must round-trip
    through the full stack to exactly EXPECTED_OUTPUT_MESSAGES.
    """
    agent_meta = Agent(name="static-agent", description="A static agent.", graph=static_agent)
    agent_meta.observability = EmptyObservability()

    # A REAL AgentExecutor (load/validate patched out) holding only the static agent, so the route
    # invokes the genuine AgentExecutor.stream logic rather than a mock.
    with (
        patch.object(AgentExecutor, "load_agents_from_imports"),
        patch.object(AgentExecutor, "_validate_default_agent_loaded"),
    ):
        executor = AgentExecutor("dummy:dummy")
        executor.agents = {"static-agent": agent_meta}

    # message_generator resolves the executor via request.app.state.agent_executor (set by lifespan
    # in production); we inject it directly so the genuine route + executor.stream path runs.
    app.state.agent_executor = executor

    client = AgentClient(agent="static-agent", base_url="http://0.0.0.0:8080", get_info=False, verify=False)
    messages = [r for r in client.stream({"message": "Test message"}, stream_tokens=True) if isinstance(r, ChatMessage)]

    # Guard against the old failure mode where an empty stream made the comparison loop a no-op.
    assert len(messages) == len(EXPECTED_OUTPUT_MESSAGES)
    for expected, actual in zip(EXPECTED_OUTPUT_MESSAGES, messages):
        actual.run_id = None
        assert expected == actual
