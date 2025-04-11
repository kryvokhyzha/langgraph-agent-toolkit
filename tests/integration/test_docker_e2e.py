import pytest
from streamlit.testing.v1 import AppTest

from langgraph_agent_toolkit.client import AgentClient


@pytest.mark.docker
def test_service_with_fake_model():
    """Test the service using the fake model.

    This test requires the service container to be running with USE_FAKE_MODEL=true
    """
    client = AgentClient("http://0.0.0.0", agent="chatbot-agent")
    response = client.invoke("Tell me a joke?", model="fake")
    assert response.type == "ai"
    assert response.content == "This is a test response from the fake model."


@pytest.mark.docker
def test_service_with_app():
    """Test the service using the app.

    This test requires the service container to be running with USE_FAKE_MODEL=true
    """
    at = AppTest.from_file("../../langgraph_agent_toolkit/streamlit_app.py").run()

    # First check for the welcome message that appears when the app first loads
    assert len(at.chat_message) >= 1
    assert at.chat_message[0].avatar == "assistant"
