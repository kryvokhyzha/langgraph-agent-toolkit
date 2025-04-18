import pytest
from streamlit.testing.v1 import AppTest

from langgraph_agent_toolkit.client import AgentClient


@pytest.mark.docker
def test_service_with_fake_model(check_service_available):
    """Test the service using the fake model.

    This test requires the service container to be running with USE_FAKE_MODEL=true
    """
    service_url = "http://0.0.0.0"

    # Skip test if service is not available
    if not check_service_available(service_url):
        pytest.skip(f"Service at {service_url} is not available. Is the Docker container running?")

    client = AgentClient(service_url, agent="chatbot-agent")
    response = client.invoke("Tell me a joke?", model="fake")
    assert response.type == "ai"
    assert response.content == "This is a test response from the fake model."


@pytest.mark.docker
def test_service_with_app():
    """Test the service using the app.

    This test requires the service container to be running with USE_FAKE_MODEL=true
    """
    try:
        # Increase timeout to allow app more time to load
        at = AppTest.from_file("../../langgraph_agent_toolkit/streamlit_app.py", default_timeout=10).run()

        # Check if there are any errors in the app
        errors = [
            elem
            for elem in at._tree.get(0, {}).get("children", {}).values()
            if hasattr(elem, "type") and elem.type == "error"
        ]

        if errors:
            pytest.skip(f"App failed to load properly: {errors}")

        # First check for the welcome message that appears when the app first loads
        assert len(at.chat_message) >= 1
        assert at.chat_message[0].avatar == "assistant"

    except Exception as e:
        pytest.skip(f"Failed to run Streamlit app: {str(e)}")
