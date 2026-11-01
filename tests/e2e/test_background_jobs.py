"""Test a real API and separate queue worker with disposable PostgreSQL schemas."""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from uuid import uuid4

import httpx
import psycopg
import pytest
from api_process import BOOTSTRAP, ApiProcess
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict


TEST_DSN = os.environ.get("LAT_TEST_POSTGRES_DSN")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not TEST_DSN, reason="Set LAT_TEST_POSTGRES_DSN to a disposable PostgreSQL database"),
]
WORKER_BOOTSTRAP = BOOTSTRAP.replace("langgraph_agent_toolkit.run_api", "langgraph_agent_toolkit.run_worker")
TERMINAL = {"completed", "interrupted", "cancelled", "failed"}


class WorkerProcess(ApiProcess):
    """Use the API test's clean environment and bounded process cleanup."""

    def __init__(self, api: ApiProcess, *, name: str = "worker"):
        self.directory = api.directory
        self.env = api.env.copy()
        self.logfile = api.directory / f"{name}.log"
        self.process = None
        self.log = None

    def start(self):
        self.log = self.logfile.open("ab")
        self.process = subprocess.Popen(
            [sys.executable, "-c", WORKER_BOOTSTRAP, "run", "--recovery_interval=1", "--stalled_after=3"],
            cwd=self.directory,
            env=self.env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )


@dataclass
class BackgroundService:
    api: ApiProcess
    worker: WorkerProcess
    http: httpx.Client

    def assert_worker_alive(self):
        if self.worker.process is not None:
            assert self.worker.process.poll() is None, self.worker.logfile.read_text()

    def wait_job(self, job_id: str, status: str, *, owner: str = "owner-a", timeout: float = 25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.assert_worker_alive()
            response = self.http.get(f"/jobs/{job_id}", params={"user_id": owner})
            assert response.status_code == 200, response.text
            job = response.json()
            if job["status"] == status:
                return job
            assert job["status"] not in TERMINAL, job
            time.sleep(0.05)
        pytest.fail(f"Job did not reach {status}: {job}\n{self.worker.logfile.read_text()}")

    def events(self, marker: str) -> list[str]:
        path = self.api.directory / f"{marker}.events"
        return path.read_text().splitlines() if path.exists() else []

    def wait_event(self, marker: str, event: str):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            self.assert_worker_alive()
            if event in self.events(marker):
                return
            time.sleep(0.05)
        pytest.fail(f"Missing {event} event for {marker}.\n{self.worker.logfile.read_text()}")

    def submit(
        self,
        thread: str,
        *,
        mode: str = "complete",
        key: str | None = None,
        owner: str = "owner-a",
        marker: str | None = None,
    ):
        response = self.http.post(
            "/background-agent/jobs",
            headers={"Idempotency-Key": key or thread},
            json={
                "thread_id": thread,
                "user_id": owner,
                "input": {"mode": mode, "marker": marker or thread, "value": marker or thread},
            },
        )
        assert response.status_code == 202, response.text
        location = httpx.URL(response.headers["location"])
        assert location.path == f"/jobs/{response.json()['job_id']}"
        assert location.params["user_id"] == owner
        return response.json()


@pytest.fixture
def background_service(tmp_path):
    suffix = uuid4().hex
    checkpoint_schema = "lat_e2e_checkpoints_" + suffix
    queue_schema = "lat_e2e_jobs_" + suffix
    connection = conninfo_to_dict(TEST_DSN)
    api = ApiProcess(tmp_path)
    api.env.update(
        AGENT_PATHS=json.dumps(["background_agent:background_agent"]),
        DEFAULT_AGENT="background-agent",
        AUTH_MODE="trusted",
        MEMORY_BACKEND="postgres",
        POSTGRES_USER=connection.get("user", "postgres"),
        POSTGRES_PASSWORD=connection.get("password") or "unused-test-password",
        POSTGRES_HOST=connection.get("host", "127.0.0.1"),
        POSTGRES_PORT=connection.get("port", "5432"),
        POSTGRES_DB=connection.get("dbname", "postgres"),
        POSTGRES_SCHEMA=checkpoint_schema,
        POSTGRES_APPLICATION_NAME="lat-background-e2e",
        POSTGRES_MIN_SIZE="1",
        POSTGRES_POOL_SIZE="3",
        POSTGRES_LOCK_POOL_SIZE="2",
        BACKGROUND_ENABLED="true",
        BACKGROUND_POSTGRES_SCHEMA=queue_schema,
        BACKGROUND_POSTGRES_POOL_SIZE="3",
        BACKGROUND_WORKER_CONCURRENCY="1",
        CHECKPOINT_DURABILITY="sync",
        LAT_TEST_BACKGROUND_DIRECTORY=str(tmp_path),
    )
    worker = WorkerProcess(api)
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(checkpoint_schema)))
    try:
        initialized = subprocess.run(
            [sys.executable, "-c", WORKER_BOOTSTRAP, "initialize"],
            cwd=tmp_path,
            env=api.env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert initialized.returncode == 0, initialized.stdout + initialized.stderr
        api.start()
        with httpx.Client(base_url=api.url, headers=api.headers, timeout=10, trust_env=False) as http:
            yield BackgroundService(api=api, worker=worker, http=http)
    finally:
        worker.stop()
        api.stop()
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            for schema in (queue_schema, checkpoint_schema):
                conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


def test_submission_survives_api_restart_and_deduplicates_by_owner(background_service):
    service = background_service
    job = service.submit("durable-input", key="event-1")
    assert job["status"] == "queued"
    assert job["attempts"] == 0
    assert service.events("durable-input") == []
    assert service.submit("durable-input", key="event-1")["job_id"] == job["job_id"]
    conflict = service.http.post(
        "/background-agent/jobs",
        headers={"Idempotency-Key": "event-1"},
        json={"thread_id": "different-input", "user_id": "owner-a", "input": {}},
    )
    assert conflict.status_code == 409
    other = service.submit("other-owner", key="event-1", owner="owner-b")
    assert other["job_id"] != job["job_id"]
    for method, suffix in ((service.http.get, ""), (service.http.post, "/cancel")):
        assert method(f"/jobs/{job['job_id']}{suffix}", params={"user_id": "owner-b"}).status_code == 404

    service.api.restart()
    persisted = service.wait_job(job["job_id"], "queued")
    assert persisted["created_at"] == job["created_at"]
    service.worker.start()
    done = service.wait_job(job["job_id"], "completed")
    assert done["attempts"] == 1
    assert done["result"]["output"]["output"] == "prepared:durable-input"
    assert done["result"]["checkpoint_id"]
    assert service.wait_job(other["job_id"], "completed", owner="owner-b")["attempts"] == 1
    assert service.events("durable-input") == ["prepared", "completed"]
    assert service.submit("durable-input", key="event-1")["job_id"] == done["job_id"]


def test_interrupt_releases_worker_capacity_and_resumes_after_restart(background_service):
    service = background_service
    service.worker.start()
    paused = service.wait_job(service.submit("approval", mode="approval")["job_id"], "interrupted")
    result = paused["result"]
    assert result["checkpoint_id"]
    assert result["interrupts"][0]["value"]["question"] == "Approve this operation?"
    # The worker has one slot. A saved interrupt must release it for another thread.
    other = service.wait_job(service.submit("unrelated")["job_id"], "completed")
    assert other["result"]["output"]["output"] == "prepared:unrelated"
    service.worker.restart()
    resume = {
        "operation": "resume",
        "thread_id": "approval",
        "user_id": "owner-a",
        "expected_checkpoint_id": result["checkpoint_id"],
        "resume": {result["interrupts"][0]["id"]: "approved"},
    }
    response = service.http.post("/background-agent/jobs", headers={"Idempotency-Key": "answer-1"}, json=resume)
    assert response.status_code == 202, response.text
    done = service.wait_job(response.json()["job_id"], "completed")
    assert done["result"]["output"]["output"] == "prepared:approval; answer=approved"
    assert done["result"]["checkpoint_id"] != result["checkpoint_id"]
    assert service.events("approval") == ["prepared"]
    stale = service.http.post("/background-agent/jobs", headers={"Idempotency-Key": "stale-answer"}, json=resume)
    assert stale.status_code == 202, stale.text
    failed = service.wait_job(stale.json()["job_id"], "failed")
    assert failed["error"] == "Execution failed (ExecutionConflictError)"
    assert failed["result"] is None


def test_queued_and_running_cancellation_release_execution_capacity(background_service):
    service = background_service
    queued = service.submit("cancel-queued")
    cancelled = service.http.post(f"/jobs/{queued['job_id']}/cancel", params={"user_id": "owner-a"})
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    service.worker.start()
    running = service.submit("cancel-running", mode="blocked")
    service.wait_event("cancel-running", "waiting")
    response = service.http.post(f"/jobs/{running['job_id']}/cancel", params={"user_id": "owner-a"})
    assert response.status_code == 200
    assert response.json()["status"] in {"cancelling", "cancelled"}
    stopped = service.wait_job(running["job_id"], "cancelled")
    assert stopped["result"] is None
    service.wait_event("cancel-running", "cancelled")
    service.wait_job(service.submit("after-cancel")["job_id"], "completed")
    assert service.events("cancel-queued") == []
    assert service.events("cancel-running") == ["prepared", "waiting", "cancelled"]
    assert service.wait_job(queued["job_id"], "cancelled")["attempts"] == 0


def test_killed_worker_recovers_checkpoint_without_repeating_completed_node(background_service):
    service = background_service
    service.worker.start()
    job = service.submit("crash-recovery", mode="blocked")
    # With sync durability, entry to the second node proves the first node was saved.
    service.wait_event("crash-recovery", "waiting")
    service.worker.stop(crash=True)
    assert service.events("crash-recovery") == ["prepared", "waiting"]
    (service.api.directory / "crash-recovery.release").touch()
    service.worker.start()
    done = service.wait_job(job["job_id"], "completed", timeout=30)
    assert done["attempts"] == 2
    assert done["result"]["output"]["output"] == "prepared:crash-recovery"
    assert service.events("crash-recovery") == ["prepared", "waiting", "waiting", "completed"]
    assert done["result"]["checkpoint_id"]


def test_two_workers_serialize_one_thread_without_blocking_unrelated_work(background_service):
    service = background_service
    other_worker = WorkerProcess(service.api, name="second-worker")
    release = service.api.directory / "first-operation.release"
    service.worker.start()
    first = service.submit("shared-thread", mode="blocked", key="first", marker="first-operation")
    service.wait_event("first-operation", "waiting")
    try:
        second = service.submit("shared-thread", key="second", marker="second-operation")
        other = service.submit("unrelated-thread")
        other_worker.start()
        # Both workers have one slot. The second must skip the locked thread.
        progressed = service.wait_job(other["job_id"], "completed")
        assert progressed["result"]["output"]["output"] == "prepared:unrelated-thread"
        assert other_worker.process.poll() is None, other_worker.logfile.read_text()
        queued = service.wait_job(second["job_id"], "queued")
        assert queued["attempts"] == 0
        assert service.events("second-operation") == []

        release.touch()
        completed_first = service.wait_job(first["job_id"], "completed")
        completed_second = service.wait_job(second["job_id"], "completed")
        assert completed_first["result"]["output"]["output"] == "prepared:first-operation"
        assert completed_second["result"]["output"]["output"] == "prepared:second-operation"
    finally:
        release.touch()
        other_worker.stop()
    assert other_worker.process.returncode == 0, other_worker.logfile.read_text()
