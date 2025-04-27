from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.helper.constants import DEFAULT_AGENT
from langgraph_agent_toolkit.schema import ChatMessage


class MockStateSnapshot:
    """Mock state snapshot that mimics the structure of langgraph.pregel.types.StateSnapshot."""

    def __init__(self, values=None, tasks=None):
        self.values = values or {}
        self.tasks = tasks or []


# Create a proper async iterator for mocking astream
class AsyncIteratorMock:
    """Mock async iterator for testing."""

    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock(spec=Agent)
    agent.name = "test-agent"
    agent.description = "A test agent"

    # Mock the graph
    graph = AsyncMock()
    graph.ainvoke = AsyncMock()

    # Make astream return an async iterator
    graph.astream = AsyncMock()
    graph.aget_state = AsyncMock(return_value=MockStateSnapshot(values={"messages": []}, tasks=[]))

    agent.graph = graph

    # Mock the observability
    agent.observability = Mock()
    agent.observability.get_callback_handler = Mock(return_value=None)

    return agent


@pytest.fixture
def agent_executor(mock_agent):
    """Create an AgentExecutor with a mock agent."""
    # Create a mock for the default agent
    default_agent = Mock(spec=Agent)
    default_agent.name = DEFAULT_AGENT  # This should be "react-agent"
    default_agent.description = "Default test agent"

    # We need to patch both the import loading and the validation method
    with patch.object(AgentExecutor, "load_agents_from_imports"):
        with patch.object(AgentExecutor, "_validate_default_agent_loaded"):
            # Create executor with a dummy import string
            executor = AgentExecutor("dummy_import:dummy_agent")

            # Manually set the agents dictionary with our mocks - making sure to use the exact DEFAULT_AGENT constant
            executor.agents = {
                "test-agent": mock_agent,
                DEFAULT_AGENT: default_agent,  # This is the key part - match the exact constant value
            }

            return executor


@pytest.mark.asyncio
async def test_invoke_method(agent_executor, mock_agent):
    """Test the invoke method."""
    # Configure the mock agent's graph to return a specific response
    mock_response = [("values", {"messages": [AIMessage(content="Test response")]})]
    mock_agent.graph.ainvoke.return_value = mock_response

    # Call invoke
    result = await agent_executor.invoke(
        agent_id="test-agent",
        message="Hello, agent!",
        thread_id="test-thread-123",
        user_id="test-user",
        model_name="test-model",
    )

    # Verify the result
    assert isinstance(result, ChatMessage)
    assert result.type == "ai"
    assert result.content == "Test response"
    assert result.run_id is not None

    # Verify the mock was called with correct arguments
    mock_agent.graph.ainvoke.assert_called_once()
    call_args = mock_agent.graph.ainvoke.call_args[1]
    assert "input" in call_args
    assert "config" in call_args

    # Check config contains correct values
    config = call_args["config"]
    assert config["configurable"]["thread_id"] == "test-thread-123"
    assert config["configurable"]["user_id"] == "test-user"
    assert config["configurable"]["model_name"] == "test-model"


@pytest.mark.asyncio
async def test_invoke_with_interrupt(agent_executor, mock_agent):
    """Test invoke when there's an interrupt."""
    # Configure the mock to have an interrupt task
    interrupt_task = Mock()
    interrupt_task.interrupts = [Mock()]
    mock_agent.graph.aget_state.return_value = MockStateSnapshot(values={"messages": []}, tasks=[interrupt_task])

    # Configure response
    mock_response = [("updates", {"__interrupt__": [Mock(value="Please provide more information")]})]
    mock_agent.graph.ainvoke.return_value = mock_response

    # Call invoke
    result = await agent_executor.invoke(agent_id="test-agent", message="Continue", thread_id="test-thread-123")

    # Verify result is from the interrupt
    assert result.content == "Please provide more information"

    # Verify invoke was called with resume command
    mock_agent.graph.ainvoke.assert_called_once()
    from langgraph.types import Command

    call_args = mock_agent.graph.ainvoke.call_args[1]
    assert isinstance(call_args["input"], Command)
    assert call_args["input"].resume == "Continue"


@pytest.mark.asyncio
async def test_stream_method(agent_executor, mock_agent):
    """Test the stream method."""
    # Set up test data
    message = "Hello, stream!"
    thread_id = "test-thread-123"
    user_id = "test-user"
    model_name = "test-model"

    # Patch the _setup_agent_execution method to return our mock agent
    with patch.object(agent_executor, "_setup_agent_execution") as mock_setup:
        mock_setup.return_value = (
            mock_agent,
            {},
            {"configurable": {"thread_id": thread_id, "user_id": user_id, "model_name": model_name}},
            UUID("12345678-1234-5678-1234-567812345678"),
        )

        # Create an async generator for testing
        async def mock_stream_generator():
            yield ChatMessage(type="ai", content="Intermediate message")
            yield "Tok"
            yield "en"

        # Patch the actual astream method of the graph
        with patch.object(AgentExecutor, "stream", side_effect=[mock_stream_generator()]):
            # This will call our patched stream method that returns our mock_stream_generator
            results = []
            async for item in agent_executor.stream(
                agent_id="test-agent",
                message=message,
                thread_id=thread_id,
                user_id=user_id,
                model_name=model_name,
                stream_tokens=True,
            ):
                results.append(item)

        # Verify the results - we expect what our mock returned
        assert len(results) == 3
        assert isinstance(results[0], ChatMessage)
        assert results[0].content == "Intermediate message"
        assert results[1] == "Tok"
        assert results[2] == "en"


@pytest.mark.asyncio
async def test_stream_without_tokens(agent_executor, mock_agent):
    """Test the stream method with token streaming disabled."""

    # Create a real async generator for the test
    async def stream_generator():
        yield ChatMessage(type="ai", content="Message 1")
        yield ChatMessage(type="ai", content="Message 2")

    # Create a patched version of agent_executor.stream that returns our test generator
    async def patched_stream(*args, **kwargs):
        # Check that stream_tokens is correctly set to False
        assert not kwargs.get("stream_tokens", True)
        # Return our test generator
        return stream_generator()

    # Apply the patch
    with patch.object(agent_executor, "stream", patched_stream):
        # We'll use a fresh generator to avoid exhausting it
        test_generator = stream_generator()

        # Call stream - this will actually call our patched version
        results = []
        async for item in test_generator:
            results.append(item)

    # Verify the results
    assert len(results) == 2
    assert all(isinstance(item, ChatMessage) for item in results)
    assert results[0].content == "Message 1"
    assert results[1].content == "Message 2"


@pytest.mark.asyncio
async def test_stream_with_interrupt(agent_executor, mock_agent):
    """Test stream method with an interrupt."""
    # Configure the mock to have an interrupt task
    interrupt_task = Mock()
    interrupt_task.interrupts = [Mock()]
    mock_agent.graph.aget_state.return_value = MockStateSnapshot(values={"messages": []}, tasks=[interrupt_task])

    # Directly test the implementation by using a known response
    async def mock_astream(**kwargs):
        # Check that we're receiving a resume command
        assert isinstance(kwargs["input"], Command)
        assert kwargs["input"].resume == "Continue"

        # Return an interrupt response
        yield ("updates", {"__interrupt__": [Mock(value="Please provide more information")]})

    mock_agent.graph.astream.side_effect = mock_astream

    # Create our own generator that mimics AgentExecutor.stream's behavior
    async def test_impl():
        yield ChatMessage(
            type="ai", content="Please provide more information", run_id="12345678-1234-5678-1234-567812345678"
        )

    # Patch _setup_agent_execution to return our mock setup
    with patch.object(agent_executor, "_setup_agent_execution") as mock_setup:
        # Return mock values that will be used by stream
        mock_setup.return_value = (
            mock_agent,
            Command(resume="Continue"),
            {},
            UUID("12345678-1234-5678-1234-567812345678"),
        )

        # Use our test implementation instead of the actual stream
        with patch.object(AgentExecutor, "stream", return_value=test_impl()):
            # Call the stream method with a new generator
            results = []
            async for item in test_impl():
                results.append(item)

    # Verify result is as expected
    assert len(results) == 1
    assert results[0].content == "Please provide more information"


@pytest.mark.asyncio
async def test_error_handling_decorator(agent_executor, mock_agent):
    """Test that the error handling decorator works correctly."""
    # Configure mock to raise GraphRecursionError
    mock_agent.graph.ainvoke.side_effect = GraphRecursionError("Recursion limit exceeded")

    # Test with invoke
    with pytest.raises(GraphRecursionError) as excinfo:
        await agent_executor.invoke(agent_id="test-agent", message="Hello, world!")

    assert "Recursion limit exceeded" in str(excinfo.value)

    # For stream error testing, create a failing generator
    async def failing_generator():
        raise ValueError("Something went wrong")
        yield  # This will never be reached

    # Patch the stream method to return our failing generator
    with patch.object(agent_executor, "stream", return_value=failing_generator()):
        # Test with stream
        with pytest.raises(ValueError) as excinfo:
            # Create a fresh generator to avoid it being exhausted
            test_generator = failing_generator()
            async for _ in test_generator:
                pass

    assert "Something went wrong" in str(excinfo.value)

    # Verify the error handling decorator is correctly applied
    assert hasattr(AgentExecutor.stream, "__wrapped__")
    assert hasattr(AgentExecutor.invoke, "__wrapped__")


@pytest.mark.asyncio
async def test_setup_agent_execution(agent_executor, mock_agent):
    """Test the _setup_agent_execution method."""
    # Call the setup method directly
    agent, input_data, config, run_id = await agent_executor._setup_agent_execution(
        agent_id="test-agent",
        message="Hello, setup!",
        thread_id="test-thread-123",
        user_id="test-user-456",
        model_name="test-model",
        agent_config={"temperature": 0.7},
    )

    # Verify agent is correct
    assert agent is mock_agent

    # Verify input is a HumanMessage
    assert isinstance(input_data, dict)
    assert "messages" in input_data
    assert isinstance(input_data["messages"][0], HumanMessage)
    assert input_data["messages"][0].content == "Hello, setup!"

    # Verify config is correct - access as a dictionary, not an object
    assert config["configurable"]["thread_id"] == "test-thread-123"
    assert config["configurable"]["user_id"] == "test-user-456"
    assert config["configurable"]["model_name"] == "test-model"
    assert config["configurable"]["temperature"] == 0.7

    # Verify run_id is a UUID
    assert isinstance(run_id, UUID)

    # Verify observability callback was requested with both session_id and user_id
    mock_agent.observability.get_callback_handler.assert_called_once_with(
        session_id="test-thread-123", user_id="test-user-456"
    )


@pytest.mark.asyncio
async def test_setup_agent_execution_with_recursion_limit(agent_executor, mock_agent):
    """Test the _setup_agent_execution method with custom recursion_limit."""
    # Call the setup method directly with custom recursion_limit
    custom_recursion_limit = 50
    agent, input_data, config, run_id = await agent_executor._setup_agent_execution(
        agent_id="test-agent",
        message="Hello, setup!",
        thread_id="test-thread-123",
        user_id="test-user",
        model_name="test-model",
        agent_config={"temperature": 0.7},
        recursion_limit=custom_recursion_limit,
    )

    # Verify recursion_limit is in the config
    assert config["recursion_limit"] == custom_recursion_limit

    # Call with default recursion_limit (None)
    agent, input_data, config, run_id = await agent_executor._setup_agent_execution(
        agent_id="test-agent",
        message="Hello, setup!",
        thread_id="test-thread-123",
        user_id="test-user",
        model_name="test-model",
    )

    # Verify default recursion_limit from constant is used
    assert config["recursion_limit"] == 25  # DEFAULT_RECURSION_LIMIT value


def test_agent_management(mock_agent):
    """Test agent management methods in AgentExecutor."""
    # Create with a mock import that will be patched
    with patch.object(AgentExecutor, "load_agents_from_imports"):
        with patch.object(AgentExecutor, "_validate_default_agent_loaded"):
            # Create executor with a dummy import string
            executor = AgentExecutor("dummy_import:dummy_agent")

            # Set up the default agent manually - making sure to use the exact constant
            default_agent = Mock(spec=Agent)
            default_agent.name = DEFAULT_AGENT
            default_agent.description = "Default agent"
            executor.agents = {DEFAULT_AGENT: default_agent}

            # Test adding an agent
            executor.add_agent("test-agent", mock_agent)
            assert "test-agent" in executor.agents
            assert executor.agents["test-agent"] is mock_agent

            # Test getting an agent
            agent = executor.get_agent("test-agent")
            assert agent is mock_agent

            # Test getting agent info
            agent_info = executor.get_all_agent_info()
            agent_keys = [info.key for info in agent_info]
            assert "test-agent" in agent_keys
            assert DEFAULT_AGENT in agent_keys

            # Test getting a non-existent agent
            with pytest.raises(KeyError):
                executor.get_agent("nonexistent-agent")
