from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from langgraph_agent_toolkit.service.service import app


class MockStateSnapshot:
    """Mock state snapshot that mimics the structure of langgraph.pregel.types.StateSnapshot."""

    def __init__(self, values=None, tasks=None):
        self.values = values or {}
        self.tasks = tasks or []


@pytest.fixture
def test_client():
    """Fixture to create a FastAPI test client."""
    # Setup mocks for the app lifecycle
    mock_observability = Mock()
    mock_memory_backend = Mock()
    mock_saver = AsyncMock()
    mock_saver.setup = AsyncMock()

    # Mock agents setup
    mock_agent = Mock()
    mock_agent.graph = Mock()
    mock_agent.observability = mock_observability

    # Create context managers
    mock_context = AsyncMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_saver)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    mock_memory_backend.get_checkpoint_saver.return_value = mock_context

    # Apply patches for app initialization
    with patch("langgraph_agent_toolkit.memory.factory.MemoryFactory.create", return_value=mock_memory_backend):
        with patch(
            "langgraph_agent_toolkit.observability.factory.ObservabilityFactory.create", return_value=mock_observability
        ):
            with patch("langgraph_agent_toolkit.service.handler.get_all_agent_info", return_value=[]):
                with patch("langgraph_agent_toolkit.service.handler.get_agent", return_value=mock_agent):
                    client = TestClient(app)
                    yield client


@pytest.fixture
def mock_agent():
    """Fixture to create a mock agent that can be configured for different test scenarios."""
    agent_mock = Mock()
    agent_mock.graph = Mock()

    # Configure async methods with AsyncMock
    agent_mock.graph.ainvoke = AsyncMock(return_value=[("values", {"messages": [AIMessage(content="Test response")]})])
    agent_mock.graph.aget_state = AsyncMock(return_value=MockStateSnapshot(values={"messages": []}, tasks=[]))

    # Configure the astream method to work as an async generator
    async def mock_astream(*args, **kwargs):
        for item in [("values", {"messages": [AIMessage(content="Test response")]})]:
            yield item

    agent_mock.graph.astream = mock_astream
    agent_mock.graph.get_state = Mock()

    agent_mock.observability = Mock()
    agent_mock.observability.get_callback_handler = Mock(return_value=None)

    with patch("langgraph_agent_toolkit.service.routes.get_agent", return_value=agent_mock):
        yield agent_mock


@pytest.fixture
def mock_settings(monkeypatch):
    """Fixture to ensure settings are clean for each test."""
    with patch("langgraph_agent_toolkit.service.routes.settings") as mock_settings:
        mock_settings.AUTH_SECRET = None
        yield mock_settings


@pytest.fixture
def mock_httpx():
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
