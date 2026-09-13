"""Check feedback ownership through real graphs and HTTP routes."""

import json
import sys
from types import ModuleType
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import SecretStr, ValidationError

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.core._base_settings import Settings
from langgraph_agent_toolkit.core.memory.types import MemoryBackends
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper import constants
from langgraph_agent_toolkit.helper.types import EnvironmentMode
from langgraph_agent_toolkit.service.auth import validate_auth_configuration
from langgraph_agent_toolkit.service.handler import create_app


SIGNING_SECRET = "test-feedback-proof-server-key-0000000000"
ALICE = {"Authorization": "Bearer feedback-alice-token"}
BOB = {"Authorization": "Bearer feedback-bob-token"}


@pytest.fixture
def feedback_service(monkeypatch, tmp_path, mock_env):
    recorded = []
    defaults = Settings(_env_file=None)

    class Recorder(EmptyObservability):
        def record_feedback(self, run_id, key, score, **kwargs):
            recorded.append({"run_id": run_id, "key": key, "score": score, **kwargs})

    for name, value in {
        "ENV_MODE": EnvironmentMode.PRODUCTION,
        "AUTH_MODE": defaults.AUTH_MODE,
        "AUTH_SECRET": None,
        "AUTH_USERS": {"alice": SecretStr("feedback-alice-token"), "bob": SecretStr("feedback-bob-token")},
        "FEEDBACK_SIGNING_SECRET": SecretStr(SIGNING_SECRET),
        "MEMORY_BACKEND": MemoryBackends.SQLITE,
        "SQLITE_DB_PATH": str(tmp_path / "feedback.sqlite"),
        "OBSERVABILITY_BACKEND": ObservabilityBackend.EMPTY,
        "DEFAULT_AGENT": "feedback-agent",
        "MCP_SERVERS": {},
        "MCP_AGENT_SERVERS": {},
    }.items():
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(constants, "_runtime_default_agent", "feedback-agent")

    def make_app():
        module = ModuleType("lat_feedback_agent_" + uuid4().hex)
        paths = []
        for attribute, agent_id in (("agent", "feedback-agent"), ("other", "other-agent")):
            graph = StateGraph(MessagesState)
            graph.add_node("reply", lambda state: {"messages": [AIMessage("An offline answer.")]})
            graph.add_edge(START, "reply")
            graph.add_edge("reply", END)
            setattr(module, attribute, Agent(agent_id, "An offline feedback agent.", graph.compile(), Recorder()))
            paths.append(f"{module.__name__}:{attribute}")
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(settings, "AGENT_PATHS", paths)
        return create_app()

    return make_app, recorded


def _run(http, endpoint="invoke"):
    response = http.post(
        f"/feedback-agent/{endpoint}",
        headers=ALICE,
        json={"input": {"message": "Give an offline answer."}, "thread_id": "feedback-thread"},
    )
    assert response.status_code == 200, response.text
    if endpoint == "invoke":
        return response.json()
    events = []
    for line in response.text.splitlines():
        data = line.removeprefix("data: ")
        if data and data != "[DONE]":
            events.append(json.loads(data))
    return [event["content"] for event in events if event["type"] == "message"][-1]


def _feedback(message, **updates):
    return {
        "run_id": message["run_id"],
        "feedback_token": message["feedback_token"],
        "key": "helpfulness",
        "score": 1.0,
        **updates,
    }


@pytest.mark.parametrize("endpoint", ["invoke", "stream", "stream/jsonl"])
def test_run_feedback_binds_user_agent_and_run(feedback_service, endpoint):
    make_app, recorded = feedback_service
    with TestClient(make_app()) as http:
        message = _run(http, endpoint)
        assert message["feedback_token"].startswith("v1.")
        assert message["thread_id"] == "feedback-thread"
        kwargs = {"comment": "Useful answer", "tags": ["offline"], "metadata": {"category": "test"}}
        saved = http.post("/feedback", headers=ALICE, json=_feedback(message, kwargs=kwargs))
        assert saved.status_code == 201, saved.text
        assert recorded == [
            {"run_id": message["run_id"], "key": "helpfulness", "score": 1.0, "user_id": "alice", **kwargs}
        ]

        cases = [
            (BOB, "/feedback-agent/feedback", _feedback(message)),
            (ALICE, "/other-agent/feedback", _feedback(message)),
            (ALICE, "/feedback", _feedback(message, run_id=str(uuid4()))),
            (ALICE, "/feedback", _feedback(message, feedback_token="v1." + "0" * 64)),
            (ALICE, "/feedback", _feedback(message, feedback_token="invalid-\N{SNOWMAN}")),
            (ALICE, "/feedback", _feedback(message, feedback_token=None)),
            (ALICE, "/feedback", _feedback(message, user_id="bob")),
        ]
        for headers, path, body in cases:
            rejected = http.post(path, headers=headers, json=body)
            assert rejected.status_code == 403, rejected.text
        assert len(recorded) == 1


def test_feedback_proof_survives_new_app_and_sqlite_restart(feedback_service):
    make_app, recorded = feedback_service
    with TestClient(make_app()) as http:
        message = _run(http)
    with TestClient(make_app()) as http:
        saved = http.post("/feedback-agent/feedback", headers=ALICE, json=_feedback(message))
        assert saved.status_code == 201, saved.text
        history = http.get("/feedback-agent/history", headers=ALICE, params={"thread_id": "feedback-thread"})
        assert history.status_code == 200
        assert all(item["run_id"] is None and item["feedback_token"] is None for item in history.json()["messages"])
    assert len(recorded) == 1


def test_imported_history_cannot_issue_feedback_proof(feedback_service):
    make_app, recorded = feedback_service
    with TestClient(make_app()) as http:
        alice_message = _run(http)
        # A caller can know another user's run ID and proof without owning the run.
        forged = {
            "type": "ai",
            "content": "Imported text",
            "run_id": alice_message["run_id"],
            "feedback_token": alice_message["feedback_token"],
            "response_metadata": {
                "run_id": alice_message["run_id"],
                "feedback_token": alice_message["feedback_token"],
            },
        }
        imported = http.post(
            "/feedback-agent/history/add_messages",
            headers=BOB,
            json={"thread_id": "imported-thread", "messages": [forged]},
        )
        assert imported.status_code == 201, imported.text
        history = http.get("/feedback-agent/history", headers=BOB, params={"thread_id": "imported-thread"})
        saved = history.json()["messages"][-1]
        assert saved["run_id"] is None
        assert saved["feedback_token"] is None
        rejected = http.post("/feedback", headers=BOB, json=_feedback(alice_message))
        assert rejected.status_code == 403
    assert recorded == []


@pytest.mark.parametrize("endpoint", ["invoke", "stream", "stream/jsonl"])
def test_missing_signing_secret_only_disables_token_feedback(feedback_service, monkeypatch, endpoint):
    make_app, recorded = feedback_service
    monkeypatch.setattr(settings, "FEEDBACK_SIGNING_SECRET", None)
    with TestClient(make_app()) as http:
        message = _run(http, endpoint)
        assert message["feedback_token"] is None
        rejected = http.post("/feedback", headers=ALICE, json=_feedback(message))
        assert rejected.status_code == 503
        assert "FEEDBACK_SIGNING_SECRET" in rejected.json()["detail"]
    assert recorded == []


@pytest.mark.parametrize("user_id", [None, "client-user"])
def test_default_shared_token_keeps_legacy_feedback_but_user_tokens_require_proof(
    feedback_service, monkeypatch, user_id
):
    make_app, recorded = feedback_service
    monkeypatch.setattr(settings, "AUTH_SECRET", SecretStr("feedback-shared-token"))
    monkeypatch.setattr(settings, "FEEDBACK_SIGNING_SECRET", None)
    body = {"run_id": str(uuid4()), "key": "helpfulness", "score": 1.0}
    if user_id is not None:
        body["user_id"] = user_id
    with TestClient(make_app()) as http:
        saved = http.post("/feedback", headers={"Authorization": "Bearer feedback-shared-token"}, json=body)
        assert saved.status_code == 201, saved.text
        assert recorded[-1]["user_id"] == (user_id or settings.AUTH_SERVICE_USER_ID)
        rejected = http.post("/feedback", headers=ALICE, json={**body, "user_id": "alice"})
        assert rejected.status_code == 503
    assert len(recorded) == 1


@pytest.mark.parametrize(
    "override",
    [
        {"run_id": "other"},
        {"trace_id": "other"},
        {"observation_id": "other"},
        {"session_id": "other"},
        {"project_id": "other"},
        {"dataset_run_id": "other"},
        {"user_id": "bob"},
        {"key": "other"},
        {"name": "other"},
        {"score": 0},
        {"value": 0},
        {"id": "existing-score"},
        {"score_id": "existing-score"},
        {"feedback_id": "existing-feedback"},
        {"source_run_id": "other"},
        {"source_info": {"__run": {"run_id": "other"}}},
    ],
)
def test_feedback_rejects_provider_target_overrides_before_submission(feedback_service, override):
    make_app, recorded = feedback_service
    with TestClient(make_app()) as http:
        message = _run(http)
        rejected = http.post("/feedback", headers=ALICE, json=_feedback(message, kwargs=override))
        assert rejected.status_code == 422, rejected.text
    assert recorded == []


@pytest.mark.parametrize("credential", ["shared", "user"])
def test_signing_secret_cannot_be_a_client_bearer_credential(feedback_service, monkeypatch, credential):
    if credential == "shared":
        monkeypatch.setattr(settings, "AUTH_SECRET", SecretStr(SIGNING_SECRET))
    else:
        monkeypatch.setattr(settings, "AUTH_USERS", {"alice": SecretStr(SIGNING_SECRET)})
    with pytest.raises(ValueError, match="must differ from all client bearer tokens"):
        validate_auth_configuration()


def test_signing_secret_requires_at_least_32_characters():
    with pytest.raises(ValidationError, match="FEEDBACK_SIGNING_SECRET"):
        Settings(_env_file=None, FEEDBACK_SIGNING_SECRET="too-short")
