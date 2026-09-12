from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.memory.concurrency import ConversationCoordinator
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.schema.schema import ChatMessage
from langgraph_agent_toolkit.service.handler import create_app


@pytest.fixture
def app():
    """Fixture to create a FastAPI app for testing."""
    return create_app()


@pytest.fixture
def mock_agent_executor(mock_state_snapshot):
    """Mock AgentExecutor with a single default agent.

    invoke/stream use return_value-style mocks so individual tests can override the
    response via `mock_agent_executor.invoke.return_value = ...`. (A previous duplicate
    fixture used side_effect, which silently ignored such overrides.)
    """
    agent_mock = Mock()
    agent_mock.name = settings.DEFAULT_AGENT
    agent_mock.description = "A mock agent for testing"

    graph = AsyncMock()
    graph.ainvoke = AsyncMock()
    graph.aget_state = AsyncMock(return_value=mock_state_snapshot(values={"messages": []}, tasks=[]))
    graph.get_state = Mock(return_value=mock_state_snapshot(values={"messages": []}, tasks=[]))

    async def mock_astream(*args, **kwargs):
        for item in [("values", {"messages": [AIMessage(content="Test response")]})]:
            yield item

    graph.astream = mock_astream
    agent_mock.graph = graph

    agent_mock.observability = Mock()
    agent_mock.observability.get_callback_handler = Mock(return_value=None)

    executor = Mock(spec=AgentExecutor)
    executor.concurrency = ConversationCoordinator()
    executor.agents = {settings.DEFAULT_AGENT: agent_mock}
    executor.get_agent = Mock(return_value=agent_mock)
    executor.get_all_agent_info = Mock(
        return_value=[{"key": settings.DEFAULT_AGENT, "description": "A mock agent for testing"}]
    )

    executor.invoke = AsyncMock(return_value=ChatMessage(type="ai", content="Default test response"))

    async def mock_stream_gen(*args, **kwargs):
        yield ChatMessage(type="ai", content="Default test response")

    executor.stream = mock_stream_gen

    return executor


@pytest.fixture
def mock_agent(mock_agent_executor):
    """Fixture to get the mock agent from the executor."""
    return mock_agent_executor.get_agent(settings.DEFAULT_AGENT)


@pytest.fixture
def test_client(mock_agent_executor, app):
    """Test HTTP routing and serialization with an injected executor.

    Lifespan does not run here. Integration tests cover resource initialization.
    """
    app.state.agent_executor = mock_agent_executor
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client
    finally:
        client.close()
