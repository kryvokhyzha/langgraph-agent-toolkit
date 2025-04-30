from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.helper.constants import DEFAULT_AGENT
from langgraph_agent_toolkit.schema.schema import ChatMessage
from langgraph_agent_toolkit.service.factory import ServiceRunner


class MockStateSnapshot:
    """Mock state snapshot that mimics the structure of langgraph.pregel.types.StateSnapshot."""

    def __init__(self, values=None, tasks=None):
        self.values = values or {}
        self.tasks = tasks or []


@pytest.fixture
def app():
    """Fixture to create a FastAPI app for testing."""
    service_runner = ServiceRunner()
    return service_runner.app


@pytest.fixture
def mock_agent_executor():
    """Fixture to create a mock agent executor with the default agent."""
    # Create mock agent
    agent_mock = Mock()
    agent_mock.name = DEFAULT_AGENT
    agent_mock.description = "A mock agent for testing"
    agent_mock.graph = Mock()

    # Configure async methods with AsyncMock
    agent_mock.graph.ainvoke = AsyncMock(return_value=[("values", {"messages": [AIMessage(content="Test response")]})])

    # Create a proper StateSnapshot for aget_state
    mock_state = MockStateSnapshot(values={"messages": []}, tasks=[])
    agent_mock.graph.aget_state = AsyncMock(return_value=mock_state)

    # Configure the astream method to work as an async generator
    async def mock_astream(*args, **kwargs):
        for item in [("values", {"messages": [AIMessage(content="Test response")]})]:
            yield item

    agent_mock.graph.astream = mock_astream
    agent_mock.graph.get_state = Mock(return_value=mock_state)

    agent_mock.observability = Mock()
    agent_mock.observability.get_callback_handler = Mock(return_value=None)

    # Create the executor with our agent
    executor = Mock(spec=AgentExecutor)
    executor.agents = {DEFAULT_AGENT: agent_mock}
    executor.get_agent = Mock(return_value=agent_mock)
    executor.get_all_agent_info = Mock(return_value=[{"key": DEFAULT_AGENT, "description": "A mock agent for testing"}])

    # We'll capture all args that are passed to these methods
    async def mock_invoke(**kwargs):
        return ChatMessage(type="ai", content="Test response")

    executor.invoke = AsyncMock(side_effect=mock_invoke)

    async def mock_stream_gen(**kwargs):
        yield ChatMessage(type="ai", content="Test response")

    # Don't use side_effect here as it complicates working with async generators
    executor.stream = mock_stream_gen

    return executor


@pytest.fixture
def mock_agent(mock_agent_executor):
    """Fixture to get the mock agent from the executor."""
    return mock_agent_executor.get_agent(DEFAULT_AGENT)


@pytest.fixture
def test_client(mock_agent_executor, app):
    """Fixture to create a FastAPI test client."""
    # Setup mocks for the app lifecycle
    mock_observability = Mock()
    mock_memory_backend = Mock()
    mock_saver = AsyncMock()
    mock_saver.setup = AsyncMock()

    # Create context managers
    mock_context = AsyncMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_saver)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    mock_memory_backend.get_checkpoint_saver.return_value = mock_context

    # Store original verify_bearer to restore later
    from langgraph_agent_toolkit.service import utils

    original_verify_bearer = utils.verify_bearer

    # Apply patches for app initialization
    with patch("langgraph_agent_toolkit.core.memory.factory.MemoryFactory.create", return_value=mock_memory_backend):
        with patch(
            "langgraph_agent_toolkit.core.observability.factory.ObservabilityFactory.create",
            return_value=mock_observability,
        ):
            with patch("langgraph_agent_toolkit.service.routes.get_agent_executor", return_value=mock_agent_executor):
                with patch(
                    "langgraph_agent_toolkit.service.routes.get_agent",
                    side_effect=lambda req, agent_id: mock_agent_executor.get_agent(agent_id),
                ):
                    with patch(
                        "langgraph_agent_toolkit.service.routes.get_all_agent_info",
                        return_value=mock_agent_executor.get_all_agent_info(),
                    ):
                        # Create a test client
                        client = TestClient(app)

                        # Set up app state with our mock executor
                        app.state.agent_executor = mock_agent_executor

                        yield client

                        # Restore the original verify_bearer
                        utils.verify_bearer = original_verify_bearer


@pytest.fixture
def mock_settings(monkeypatch):
    """Fixture to ensure settings are clean for each test."""
    # Patch settings in the modules where it's actually used
    with patch("langgraph_agent_toolkit.core.settings.settings") as mock_settings:
        with patch("langgraph_agent_toolkit.agents.agent_executor.settings", mock_settings):
            with patch("langgraph_agent_toolkit.agents.blueprints.bg_task_agent.agent.settings", mock_settings):
                with patch("langgraph_agent_toolkit.agents.blueprints.chatbot.agent.settings", mock_settings):
                    # Set up the mock settings
                    mock_settings.AUTH_SECRET = None
                    mock_settings.MODEL_CONFIGS = {}
                    yield mock_settings


@pytest.fixture
def mock_httpx(app):
    """Patch httpx.stream and httpx.get to use our test client."""
    with TestClient(app) as client:

        def mock_stream(method: str, url: str, **kwargs):
            # Strip the base URL since TestClient expects just the path
            path = url.replace("http://0.0.0.0:8080", "")
            return client.stream(method, path, **kwargs)

        def mock_get(url: str, **kwargs):
            # Strip the base URL since TestClient expects just the path
            path = url.replace("http://0.0.0.0:8080", "")
            return client.get(path, **kwargs)

        def mock_post(url: str, **kwargs):
            # Strip the base URL since TestClient expects just the path
            path = url.replace("http://0.0.0.0:8080", "")
            return client.post(path, **kwargs)

        with patch("httpx.stream", mock_stream):
            with patch("httpx.get", mock_get):
                with patch("httpx.post", mock_post):
                    yield
