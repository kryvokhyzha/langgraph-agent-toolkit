"""Check feedback serialization with the installed LangSmith SDK."""

import json
from uuid import uuid4

import pytest
import requests

from langgraph_agent_toolkit.core.observability import langsmith as integration


sdk = pytest.importorskip("langsmith")


@pytest.mark.parametrize("include_session", [False, True])
def test_feedback_preserves_user_and_project_metadata(monkeypatch, include_session):
    requests_sent = []
    session = requests.Session()

    def send(request, **kwargs):
        requests_sent.append(request)
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response.request = request
        return response

    monkeypatch.setattr(session, "send", send)
    client = sdk.Client(
        api_url="https://langsmith.test",
        api_key="test-credential",
        session=session,
        auto_batch_tracing=False,
        info={},
    )
    monkeypatch.setattr(integration, "LangsmithClient", lambda: client)
    for name, value in {
        "LANGSMITH_TRACING": "false",
        "LANGSMITH_API_KEY": "test-credential",
        "LANGSMITH_PROJECT": "test-project",
        "LANGSMITH_ENDPOINT": "https://langsmith.test",
    }.items():
        monkeypatch.setenv(name, value)
    run_id = str(uuid4())
    session_id = str(uuid4())
    options = {"session_id": session_id} if include_session else {}
    try:
        integration.LangsmithObservability().record_feedback(
            run_id,
            "helpful",
            1,
            user_id="user-1",
            extra={"source": "chat-ui"},
            comment="The answer was useful.",
            **options,
        )
    finally:
        client.close()

    assert len(requests_sent) == 1
    request = requests_sent[0]
    assert request.method == "POST"
    assert request.url == "https://langsmith.test/feedback"
    payload = json.loads(request.body)
    assert payload["run_id"] == run_id
    assert payload["key"] == "helpful"
    assert payload["score"] == 1
    assert payload["comment"] == "The answer was useful."
    assert payload["extra"] == {"source": "chat-ui", "user_id": "user-1"}
    assert payload.get("session_id") == (session_id if include_session else None)
