"""Check the running service and app containers with client-only dependencies."""

import os
from uuid import uuid4

import httpx
import pytest

from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.schema import ChatMessage
from langgraph_agent_toolkit.schema.models import ModelProvider


pytestmark = pytest.mark.docker


@pytest.fixture
def docker_service_url():
    url = os.environ.get("AGENT_URL", "http://127.0.0.1:8080").rstrip("/")
    try:
        response = httpx.get(f"{url}/health/ready", timeout=10, trust_env=False)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.fail(f"The requested Docker service is not ready at {url}: {exc}")
    return url


def test_docker_service_invokes_streams_and_retains_conversation(docker_service_url):
    expected = "This is a test response from the fake model."
    thread_id = f"docker-{uuid4()}"
    with httpx.Client(trust_env=False) as http:
        with AgentClient(docker_service_url, agent="chatbot-agent", http_client=http) as client:
            response = client.invoke(
                {"message": "First message."}, model_provider=ModelProvider.FAKE, thread_id=thread_id
            )
            assert response.type == "ai"
            assert response.content == expected
            events = list(
                client.stream_jsonl(
                    {"message": "Second message."}, model_provider=ModelProvider.FAKE, thread_id=thread_id
                )
            )
            messages = [event for event in events if isinstance(event, ChatMessage)]
            assert messages[-1].content == expected
            assert messages[-1].thread_id == thread_id
            assert [message.content for message in client.get_history(thread_id).messages] == [
                "First message.",
                expected,
                "Second message.",
                expected,
            ]
            client.clear_history(thread_id)
            assert client.get_history(thread_id).messages == []


def test_docker_app_serves_health_and_html():
    url = os.environ.get("APP_URL", "http://127.0.0.1:8501").rstrip("/")
    try:
        with httpx.Client(base_url=url, timeout=10, trust_env=False) as client:
            health = client.get("/_stcore/health")
            health.raise_for_status()
            page = client.get("/")
            page.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.fail(f"The requested Docker app is not available at {url}: {exc}")
    assert health.text.strip() == "ok"
    assert page.headers["content-type"].startswith("text/html")
    assert "<title>" in page.text
