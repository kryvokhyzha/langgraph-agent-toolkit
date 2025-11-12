from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from pydantic import BaseModel

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.helper.constants import DEFAULT_AGENT
from langgraph_agent_toolkit.schema import ChatMessage


class MockStateSnapshot:
    """Mock state snapshot."""

    def __init__(self, values=None, tasks=None):
        self.values = values or {}
        self.tasks = tasks or []


class MockInput(BaseModel):
    """Test input model."""

    message: str


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock(spec=Agent)
    agent.name = "test-agent"
    agent.description = "A test agent"

    graph = AsyncMock()
    graph.ainvoke = AsyncMock()
    graph.astream = AsyncMock()
    graph.aget_state = AsyncMock(return_value=MockStateSnapshot(values={"messages": []}, tasks=[]))
    agent.graph = graph

    # Properly mock observability with context manager support
    agent.observability = Mock()
    agent.observability.get_callback_handler = Mock(return_value=None)

    # Create a proper context manager mock
    @contextmanager
    def mock_trace_context(*args, **kwargs):
        yield MagicMock()

    agent.observability.trace_context = mock_trace_context

    return agent


@pytest.fixture
def agent_executor(mock_agent):
    """Create an AgentExecutor with a mock agent."""
    default_agent = Mock(spec=Agent)
    default_agent.name = DEFAULT_AGENT
    default_agent.description = "Default test agent"

    with patch.object(AgentExecutor, "load_agents_from_imports"):
        with patch.object(AgentExecutor, "_validate_default_agent_loaded"):
            executor = AgentExecutor("dummy_import:dummy_agent")
            executor.agents = {
                "test-agent": mock_agent,
                DEFAULT_AGENT: default_agent,
            }
            return executor


@pytest.mark.asyncio
async def test_invoke_basic_flow(agent_executor, mock_agent):
    """Test basic invoke flow with successful response."""
    mock_response = [("values", {"messages": [AIMessage(content="Test response")]})]
    mock_agent.graph.ainvoke.return_value = mock_response

    input_obj = MockInput(message="Hello, agent!")
    result = await agent_executor.invoke(
        agent_id="test-agent",
        input=input_obj,
        thread_id="test-thread",
        user_id="test-user",
    )

    assert isinstance(result, ChatMessage)
    assert result.type == "ai"
    assert result.content == "Test response"
    assert result.run_id is not None

    call_args = mock_agent.graph.ainvoke.call_args[1]
    config = call_args["config"]
    assert config["configurable"]["thread_id"] == "test-thread"
    assert config["configurable"]["user_id"] == "test-user"


@pytest.mark.asyncio
async def test_invoke_with_interrupt_handling(agent_executor, mock_agent):
    """Test invoke correctly handles interrupts."""
    interrupt_task = Mock()
    interrupt_task.interrupts = [Mock()]
    mock_agent.graph.aget_state.return_value = MockStateSnapshot(values={"messages": []}, tasks=[interrupt_task])

    mock_response = [("updates", {"__interrupt__": [Mock(value="Need more info")]})]
    mock_agent.graph.ainvoke.return_value = mock_response

    user_input = MockInput(message="Continue")
    result = await agent_executor.invoke(agent_id="test-agent", input=user_input)

    assert result.content == "Need more info"

    call_args = mock_agent.graph.ainvoke.call_args[1]
    assert isinstance(call_args["input"], Command)
    assert call_args["input"].resume == user_input.model_dump()


@pytest.mark.asyncio
async def test_error_handling_with_recursion_error(agent_executor, mock_agent):
    """Test error handling decorator catches GraphRecursionError."""
    mock_agent.graph.ainvoke.side_effect = GraphRecursionError("Recursion limit exceeded")

    with pytest.raises(GraphRecursionError, match="Recursion limit exceeded"):
        await agent_executor.invoke(agent_id="test-agent", input=MockInput(message="Test"))


@pytest.mark.asyncio
async def test_setup_agent_execution_configuration(agent_executor, mock_agent):
    """Test _setup_agent_execution correctly configures the agent."""
    input_obj = MockInput(message="Hello")

    agent, input_data, config, run_id = await agent_executor._setup_agent_execution(
        agent_id="test-agent",
        input=input_obj,
        thread_id="test-thread",
        user_id="test-user",
        model_name="test-model",
        agent_config={"temperature": 0.7},
        recursion_limit=50,
    )

    assert agent is mock_agent
    assert isinstance(input_data, dict)
    assert "messages" in input_data
    assert isinstance(input_data["messages"][0], HumanMessage)

    assert config["configurable"]["thread_id"] == "test-thread"
    assert config["configurable"]["user_id"] == "test-user"
    assert config["configurable"]["model_name"] == "test-model"
    assert config["configurable"]["temperature"] == 0.7
    assert config["recursion_limit"] == 50
    assert isinstance(run_id, UUID)


@pytest.mark.asyncio
async def test_trace_context_integration(agent_executor, mock_agent):
    """Test that trace_context is properly integrated with invoke."""
    mock_response = [("values", {"messages": [AIMessage(content="Response")]})]
    mock_agent.graph.ainvoke.return_value = mock_response

    # Track if trace_context was called by wrapping it
    trace_context_called = False
    original_trace_context = mock_agent.observability.trace_context

    @contextmanager
    def tracked_trace_context(*args, **kwargs):
        nonlocal trace_context_called
        trace_context_called = True
        with original_trace_context(*args, **kwargs) as span:
            yield span

    mock_agent.observability.trace_context = tracked_trace_context

    input_obj = MockInput(message="Test")
    await agent_executor.invoke(
        agent_id="test-agent",
        input=input_obj,
        thread_id="test-thread",
        user_id="test-user",
    )

    # Verify trace_context was called
    assert trace_context_called


def test_agent_management_operations(mock_agent):
    """Test agent add/get operations."""
    with patch.object(AgentExecutor, "load_agents_from_imports"):
        with patch.object(AgentExecutor, "_validate_default_agent_loaded"):
            executor = AgentExecutor("dummy_import:dummy_agent")

            default_agent = Mock(spec=Agent)
            default_agent.name = DEFAULT_AGENT
            default_agent.description = "Default agent"
            executor.agents = {DEFAULT_AGENT: default_agent}

            executor.add_agent("test-agent", mock_agent)
            assert "test-agent" in executor.agents

            agent = executor.get_agent("test-agent")
            assert agent is mock_agent

            agent_info = executor.get_all_agent_info()
            agent_keys = [info.key for info in agent_info]
            assert "test-agent" in agent_keys
            assert DEFAULT_AGENT in agent_keys

            with pytest.raises(KeyError):
                executor.get_agent("nonexistent-agent")
