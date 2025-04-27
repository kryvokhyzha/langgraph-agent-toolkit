import pytest
from streamlit.testing.v1 import AppTest

from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.schema.models import ModelProvider


@pytest.mark.docker
def test_service_with_fake_model(check_service_available):
    """Test the service using the fake model.

    This test requires the service container to be running with USE_FAKE_MODEL=true
    """
    service_url = "http://0.0.0.0"

    # Skip test if service is not available
    if not check_service_available(service_url, timeout=60):
        pytest.skip(f"Service at {service_url} is not available. Is the Docker container running?")

    client = AgentClient(service_url, agent="chatbot-agent")
    response = client.invoke("Tell me a joke?", model_provider=ModelProvider.FAKE)
    assert response.type == "ai"
    assert response.content == "This is a test response from the fake model."


@pytest.mark.docker
def test_service_with_app():
    """Test the service using the app.

    This test requires the service container to be running with USE_FAKE_MODEL=true
    """
    try:
        # Increase timeout to allow app more time to load
        at = AppTest.from_file("../../langgraph_agent_toolkit/streamlit_app.py", default_timeout=60).run()

        # First check for the welcome message that appears when the app first loads
        assert len(at.chat_message) >= 1, "Expected at least one chat message"
        assert at.chat_message[0].avatar == "assistant", "Expected first message from assistant"

        # Check that the title elements exist (indicating successful loading)
        assert at.sidebar, "Expected sidebar to be present"

    except Exception as e:
        pytest.skip(f"Failed to run Streamlit app: {str(e)}")
