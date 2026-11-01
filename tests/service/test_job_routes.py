"""Verify optional job routes, authentication, and submission limits."""

import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from langgraph_agent_toolkit.core.background import JobConflict
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper import constants
from langgraph_agent_toolkit.schema import BackgroundJob
from langgraph_agent_toolkit.service.handler import create_app


AUTH = {"Authorization": "Bearer job-test-shared"}
HEADERS = {**AUTH, "Idempotency-Key": "operation-1"}
BODY = {"thread_id": "public-thread", "input": {"value": 1}}


@pytest.fixture
def job_service(monkeypatch):
    for key, value in {
        "BACKGROUND_ENABLED": True,
        "BACKGROUND_EXECUTION_TIMEOUT": 120.0,
        "AUTH_MODE": "trusted",
        "AUTH_SECRET": SecretStr("job-test-shared"),
        "AUTH_SERVICE_USER_ID": "service",
        "AUTH_USERS": {"owner": SecretStr("job-test-owner")},
        "DEFAULT_RECURSION_LIMIT": 20,
    }.items():
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(constants, "_runtime_default_agent", "example")
    now = datetime.now(timezone.utc)
    job = BackgroundJob(
        job_id=str(uuid4()),
        agent_id="example",
        thread_id="public-thread",
        status="queued",
        created_at=now,
        updated_at=now,
    )
    queue = SimpleNamespace(
        submit=AsyncMock(return_value=job), get=AsyncMock(return_value=job), cancel=AsyncMock(return_value=job)
    )
    app = create_app()
    app.state.ready = True
    app.state.agent_executor = SimpleNamespace(get_agent={"example": object()}.__getitem__)
    app.state.background_jobs = queue
    # Route contract checks use the real middleware and auth. Process tests own startup and storage checks.
    client = TestClient(app)
    yield SimpleNamespace(client=client, app=app, queue=queue, job=job)
    client.close()


def test_jobs_are_absent_unless_enabled(monkeypatch):
    monkeypatch.setattr(settings, "BACKGROUND_ENABLED", False)
    monkeypatch.setitem(sys.modules, "procrastinate", None)
    paths = create_app().openapi()["paths"]
    assert "/invoke" in paths
    assert not any("jobs" in path for path in paths)


@pytest.mark.parametrize("path", ["/jobs", "/example/jobs"])
def test_submission_returns_location_and_applies_server_defaults(job_service, path):
    response = job_service.client.post(path, headers=HEADERS, json=BODY)
    assert response.status_code == 202, response.text
    assert response.headers["location"].endswith("/jobs/" + job_service.job.job_id)
    agent, owner, value, key = job_service.queue.submit.call_args.args
    assert (agent, owner, key) == ("example", "service", "operation-1")
    assert value.user_id == "service"
    assert value.timeout_seconds == 120
    assert value.input == {"value": 1}
    assert "request" not in response.json()
    assert "owner_id" not in response.json()


def test_trusted_user_scope_is_required_again_when_reading_or_cancelling(job_service):
    client = job_service.client
    submitted = client.post("/jobs", headers=HEADERS, json={**BODY, "user_id": "member"})
    assert submitted.status_code == 202
    assert submitted.headers["location"].endswith("?user_id=member")
    assert job_service.queue.submit.call_args.args[1] == "member"
    job_path = "/jobs/" + job_service.job.job_id
    assert client.get(submitted.headers["location"], headers=AUTH).status_code == 200
    job_service.queue.get.assert_awaited_once_with(job_service.job.job_id, "member")
    assert client.post(job_path + "/cancel", headers=AUTH, params={"user_id": "member"}).status_code == 200
    job_service.queue.cancel.assert_awaited_once_with(job_service.job.job_id, "member")
    job_service.queue.get.return_value = None
    assert client.get(job_path, headers=AUTH).status_code == 404
    job_service.queue.get.assert_awaited_with(job_service.job.job_id, "service")


def test_auth_rejects_calls_before_job_storage(job_service):
    client = job_service.client
    assert client.post("/jobs", headers={"Idempotency-Key": "x"}, json=BODY).status_code == 401
    owner_headers = {"Authorization": "Bearer job-test-owner", "Idempotency-Key": "x"}
    assert client.post("/jobs", headers=owner_headers, json={**BODY, "user_id": "other"}).status_code == 403
    path = "/jobs/" + job_service.job.job_id
    for method, suffix in (("GET", ""), ("POST", "/cancel")):
        response = client.request(method, path + suffix, headers=owner_headers, params={"user_id": "other"})
        assert response.status_code == 403
    job_service.queue.submit.assert_not_awaited()
    job_service.queue.get.assert_not_awaited()
    job_service.queue.cancel.assert_not_awaited()


@pytest.mark.parametrize(
    "changes",
    [
        {"timeout_seconds": 121},
        {"timeout_seconds": 0},
        {"recursion_limit": 21},
        {"operation": "resume", "resume": {"answer": True}},
        {"agent_config": {"thread_id": "foreign"}},
    ],
)
def test_invalid_execution_is_not_persisted(job_service, changes):
    response = job_service.client.post("/jobs", headers=HEADERS, json={**BODY, **changes})
    assert response.status_code == 422, response.text
    job_service.queue.submit.assert_not_awaited()


def test_key_agent_and_storage_errors_are_explicit(job_service):
    client = job_service.client
    assert client.post("/jobs", headers=AUTH, json=BODY).status_code == 422
    assert client.post("/missing/jobs", headers=HEADERS, json=BODY).status_code == 404
    job_service.queue.submit.assert_not_awaited()
    job_service.queue.submit.side_effect = JobConflict("Idempotency key was already used for a different request")
    assert client.post("/jobs", headers=HEADERS, json=BODY).status_code == 409
    job_service.queue.submit.side_effect = psycopg.OperationalError("private database credentials")
    response = client.post("/jobs", headers=HEADERS, json=BODY)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    assert "private" not in response.text
    job_service.app.state.ready = False
    assert client.get("/jobs/" + job_service.job.job_id, headers=AUTH).status_code == 503
