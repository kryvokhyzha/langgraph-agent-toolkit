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

    at.sidebar.selectbox[1].set_value("chatbot")
    at.chat_input[0].set_value("What is the weather in Tokyo?").run()

    # Check all messages to verify correct order
    assert len(at.chat_message) == 3

    # First message should be the user input
    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == "What is the weather in Tokyo?"

    # Second and third messages are from the assistant
    assert at.chat_message[1].avatar == "assistant"
    assert at.chat_message[2].avatar == "assistant"

    # One of the assistant messages should be the welcome message
    # and one should be the response
    assistant_messages = [at.chat_message[1].markdown[0].value, at.chat_message[2].markdown[0].value]

    assert "Hello! I'm a simple chatbot. Ask me anything!" in assistant_messages
    assert "This is a test response from the fake model." in assistant_messages

    assert not at.exception
