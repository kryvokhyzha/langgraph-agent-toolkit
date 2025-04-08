import pytest
from streamlit.testing.v1 import AppTest

from langgraph_agent_toolkit.client import AgentClient


@pytest.mark.docker
def test_service_with_fake_model():
    """Test the service using the fake model.

    This test requires the service container to be running with USE_FAKE_MODEL=true
    """
    client = AgentClient("http://0.0.0.0", agent="chatbot")
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
    assert "Hello! I'm a simple chatbot. Ask me anything!" in at.chat_message[0].markdown[0].value

    # Now set the agent and input a message
    at.sidebar.selectbox[1].set_value("chatbot")
    at.chat_input[0].set_value("What is the weather in Tokyo?").run()

    # Check all messages to verify correct order
    assert len(at.chat_message) == 3

    # First message should still be the welcome message
    assert at.chat_message[0].avatar == "assistant"
    assert "Hello! I'm a simple chatbot. Ask me anything!" in at.chat_message[0].markdown[0].value

    # Second message should be the user input
    assert at.chat_message[1].avatar == "user"
    assert at.chat_message[1].markdown[0].value == "What is the weather in Tokyo?"

    # Third message should be the assistant response
    assert at.chat_message[2].avatar == "assistant"
    assert "This is a test response from the fake model." in at.chat_message[2].markdown[0].value

    assert not at.exception
