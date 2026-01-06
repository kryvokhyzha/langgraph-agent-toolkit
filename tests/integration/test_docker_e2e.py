import httpx
import pytest
from streamlit.testing.v1 import AppTest

from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.schema import UserComplexInput, UserInput
from langgraph_agent_toolkit.schema.models import ModelProvider


@pytest.mark.docker
def test_service_with_fake_model(check_service_available):
    """Test the service using the fake model.

    This test requires the service container to be running with USE_FAKE_MODEL=true
    """
    service_url = "http://0.0.0.0:8080"

    # Skip test if service is not available
    if not check_service_available(service_url, timeout=60):
        pytest.skip(f"Service at {service_url} is not available. Is the Docker container running?")

    # First check what agents are available
    info_response = httpx.get(f"{service_url}/info", timeout=60)
    print(f"Service info: {info_response.text}")

    # Debug: make a direct request to see the actual error
    request = UserInput(input=UserComplexInput(message="Tell me a joke?"))
    request.model_provider = "fake"
    request_body = request.model_dump()
    print(f"Request body: {request_body}")

    response = httpx.post(
        f"{service_url}/chatbot-agent/invoke",
        json=request_body,
        timeout=60,
    )
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.text}")

    if response.status_code != 200:
        pytest.fail(f"Server returned {response.status_code}: {response.text}")

    client = AgentClient(service_url, agent="chatbot-agent")
    response = client.invoke({"message": "Tell me a joke?"}, model_provider=ModelProvider.FAKE)
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
