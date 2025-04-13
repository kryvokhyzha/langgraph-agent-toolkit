from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import StreamWriter

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.agents.blueprints.bg_task_agent.utils import CustomData
from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.core.memory.types import MemoryBackends
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.utils import langchain_to_chat_message
from langgraph_agent_toolkit.schema.schema import ChatMessage


# Define MockStateSnapshot locally instead of importing from tests
class MockStateSnapshot:
    """Mock state snapshot that mimics the structure of langgraph.pregel.types.StateSnapshot."""

    def __init__(self, values=None, tasks=None):
        self.values = values or {}
        self.tasks = tasks or []


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
def mock_httpx():
    """Mock httpx to avoid making real HTTP requests."""
    with patch("httpx.stream") as mock_stream:
        # Create a mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = []

        # Configure the mock to return our mock response
        mock_stream.return_value.__enter__.return_value = mock_response

        yield mock_stream


@pytest.fixture
def sqlite_db_settings():
    """Temporarily configure settings to use SQLite in-memory database for testing."""
    # Patch the settings instance, not the Settings class
    with patch.object(settings, "MEMORY_BACKEND", MemoryBackends.SQLITE):
        with patch.object(settings, "SQLITE_DB_PATH", ":memory:"):
            # Create mock async context manager
            mock_saver = AsyncMock()
            mock_saver.setup = AsyncMock()
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_saver)
            mock_context.__aexit__ = AsyncMock(return_value=None)

            # Create mock memory backend
            mock_memory_backend = MagicMock()
            mock_memory_backend.get_checkpoint_saver.return_value = mock_context

            # Patch the MemoryFactory.create method
            with patch(
                "langgraph_agent_toolkit.core.memory.factory.MemoryFactory.create", return_value=mock_memory_backend
            ):
                # Patch the ObservabilityFactory.create method
                mock_observability = MagicMock()
                with patch(
                    "langgraph_agent_toolkit.core.observability.factory.ObservabilityFactory.create",
                    return_value=mock_observability,
                ):
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


def test_agent_stream(mock_httpx, sqlite_db_settings):
    """Test that streaming from our static agent works correctly with token streaming."""
    # Create agent meta with observability and name
    agent_meta = Agent(name="static-agent", description="A static agent.", graph=static_agent)
    agent_meta.observability = MagicMock()
    agent_meta.observability.get_callback_handler = MagicMock(return_value=None)

    # Ensure aget_state returns a proper StateSnapshot
    mock_state = MockStateSnapshot(values={"messages": []}, tasks=[])
    agent_meta.graph.aget_state = AsyncMock(return_value=mock_state)

    # Create a mock agent executor that will return our test agent
    mock_executor = MagicMock(spec=AgentExecutor)
    mock_executor.get_agent = MagicMock(return_value=agent_meta)
    mock_executor.agents = {"static-agent": agent_meta, "react-agent": MagicMock()}

    # Properly patch the request dependency
    with patch("langgraph_agent_toolkit.service.routes.get_agent_executor", return_value=mock_executor):
        # Also patch the get_agent function to properly use our agent
        with patch("langgraph_agent_toolkit.service.routes.get_agent", return_value=agent_meta):
            # Initialize client with get_info=False and verify=False to avoid HTTP request
            client = AgentClient(agent="static-agent", base_url="http://0.0.0.0:8080", get_info=False, verify=False)

            # Use stream to get intermediate responses
            messages = []

            for response in client.stream("Test message", stream_tokens=False):
                if isinstance(response, ChatMessage):
                    messages.append(response)

    for expected, actual in zip(EXPECTED_OUTPUT_MESSAGES, messages):
        actual.run_id = None
        assert expected == actual
