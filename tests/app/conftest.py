from unittest.mock import patch

import pytest

from langgraph_agent_toolkit.schema import AgentInfo, ServiceMetadata


@pytest.fixture
def mock_agent_client(mock_env):
    """Fixture for creating a mock AgentClient with a clean environment."""
    mock_info = ServiceMetadata(
        default_agent="test-agent",
        agents=[
            AgentInfo(key="test-agent", description="Test agent"),
            AgentInfo(key="chatbot", description="Chatbot"),
        ],
    )

    with patch("langgraph_agent_toolkit.client.AgentClient") as mock_agent_client:
        mock_agent_client_instance = mock_agent_client.return_value
        mock_agent_client_instance.info = mock_info
        yield mock_agent_client_instance
