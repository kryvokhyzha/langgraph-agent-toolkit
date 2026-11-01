"""Check real model calls through immediate and queued API execution.

Use --run-e2e --run-postgres --run-llm with explicit LAT_TEST_* configuration.
These two tests normally make two model requests. No SDK retry is enabled.
Neither collection nor fixtures load credentials from a local environment file.
"""

import json
import os

import pytest
from test_background_jobs import TEST_DSN, background_service  # noqa: F401


pytestmark = [
    pytest.mark.llm,
    pytest.mark.postgres,
    pytest.mark.skipif(not TEST_DSN, reason="Set LAT_TEST_POSTGRES_DSN to a disposable PostgreSQL database"),
]
PROMPT = "What is 3 plus 4? Include the numeral in your answer."


@pytest.fixture
def live_execution_service(request):
    if not request.config.getoption("--run-llm"):
        pytest.skip("Use --run-llm for explicit live model execution.")
    for name in ("LAT_TEST_OPENAI_API_KEY", "LAT_TEST_OPENAI_MODEL"):
        if not os.environ.get(name):
            pytest.fail(f"Set {name} explicitly.")
    service = request.getfixturevalue("background_service")
    service.api.stop()
    service.api.env.update(
        AGENT_PATHS=json.dumps(["live_execution_agent:live_execution_agent"]),
        DEFAULT_AGENT="live-execution",
        USE_FAKE_MODEL="false",
        MODEL_CONFIGS="{}",
        LAT_TEST_LIVE_EXECUTION="yes",
        LAT_TEST_LIVE_EXECUTION_EVENTS=str(service.api.directory / "live-execution.events"),
        LAT_TEST_OPENAI_API_KEY=os.environ["LAT_TEST_OPENAI_API_KEY"],
        LAT_TEST_OPENAI_MODEL=os.environ["LAT_TEST_OPENAI_MODEL"],
        LAT_TEST_OPENAI_BASE_URL=os.environ.get("LAT_TEST_OPENAI_BASE_URL", "https://api.openai.com/v1"),
        LAT_TEST_OPENAI_MODEL_KWARGS=os.environ.get("LAT_TEST_OPENAI_MODEL_KWARGS", "{}"),
        LLM_HTTP_ASYNC_TRANSPORT="httpx",
        LLM_HTTP_MAX_RETRIES="0",
        LLM_HTTP_CONNECT_TIMEOUT="5",
        LLM_HTTP_READ_TIMEOUT="20",
        REQUEST_TIMEOUT="40",
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
        LANGFUSE_TRACING_ENABLED="false",
    )
    for name in (
        "LAT_TEST_OPENAI_PROVIDER",
        "LAT_TEST_AZURE_OPENAI_ENDPOINT",
        "LAT_TEST_AZURE_OPENAI_API_VERSION",
        "LAT_TEST_AZURE_OPENAI_DEPLOYMENT",
    ):
        if name in os.environ:
            service.api.env[name] = os.environ[name]
    service.worker.env = service.api.env.copy()
    service.api.start()
    return service


def recorded_calls(service) -> list[str]:
    path = service.api.directory / "live-execution.events"
    return path.read_text().splitlines() if path.exists() else []


def assert_saved_reply(service, thread_id: str):
    history = service.http.get("/live-execution/history", params={"thread_id": thread_id, "user_id": "owner-a"})
    assert history.status_code == 200
    messages = history.json()["messages"]
    assert len(messages) == 2
    assert messages[0]["content"] == PROMPT
    assert messages[1]["type"] == "ai"
    assert "7" in messages[1]["content"]
    return messages[1]["content"]


def test_real_model_immediate_result_survives_api_restart(live_execution_service):
    service = live_execution_service
    response = service.http.post(
        "/live-execution/invoke",
        json={"thread_id": "live-immediate", "user_id": "owner-a", "input": {"message": PROMPT}},
        timeout=45,
    )
    assert response.status_code == 200, {
        "http_status": response.status_code,
        "provider_events": recorded_calls(service),
    }
    message = response.json()
    assert message["type"] == "ai"
    assert message["thread_id"] == "live-immediate"
    assert "7" in message["content"]
    assert message["usage_metadata"]["total_tokens"] > 0
    service.api.restart()
    assert assert_saved_reply(service, "live-immediate") == message["content"]
    assert recorded_calls(service) == ["attempted", "completed"]


def test_real_model_queued_approval_resume_and_duplicate_use_saved_result(live_execution_service):
    service = live_execution_service
    payload = {
        "thread_id": "live-queued",
        "user_id": "owner-a",
        "input": {"messages": [{"role": "user", "content": PROMPT}], "review_required": True},
    }
    response = service.http.post("/live-execution/jobs", headers={"Idempotency-Key": "live-input"}, json=payload)
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert response.json()["status"] == "queued"
    service.worker.start()
    paused = service.wait_job(job_id, "interrupted")
    assert recorded_calls(service) == []
    service.worker.restart()
    result = paused["result"]
    resume = {
        "operation": "resume",
        "thread_id": "live-queued",
        "user_id": "owner-a",
        "expected_checkpoint_id": result["checkpoint_id"],
        "resume": {result["interrupts"][0]["id"]: True},
    }
    response = service.http.post("/live-execution/jobs", headers={"Idempotency-Key": "live-answer"}, json=resume)
    assert response.status_code == 202
    done = service.wait_job(response.json()["job_id"], "completed", timeout=45)
    answer = done["result"]["output"]["messages"][-1]
    assert answer["type"] == "ai"
    assert "7" in answer["content"]
    assert answer["usage_metadata"]["total_tokens"] > 0
    repeated = service.http.post("/live-execution/jobs", headers={"Idempotency-Key": "live-answer"}, json=resume)
    assert repeated.status_code == 202
    assert repeated.json()["job_id"] == done["job_id"]
    assert repeated.json()["result"] == done["result"]
    service.api.restart()
    assert assert_saved_reply(service, "live-queued") == answer["content"]
    assert recorded_calls(service) == ["attempted", "completed"]
