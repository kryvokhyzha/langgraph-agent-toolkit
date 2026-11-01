"""Protect recovery checkpoints when HTTP clients request history deletion."""

import os
from types import SimpleNamespace
from typing import TypedDict
from uuid import uuid4

import httpx
import psycopg
import pytest
from langgraph.graph import END, START, StateGraph
from psycopg import sql
from psycopg.types.json import Jsonb
from pydantic import SecretStr

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.memory.concurrency import ConversationCoordinator
from langgraph_agent_toolkit.core.memory.coordinated_sqlite import CoordinatedSqliteSaver
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.schema.execution import ExecutionRequest
from langgraph_agent_toolkit.service.handler import create_app


pytest.importorskip("procrastinate")
from langgraph_agent_toolkit.core.background import PostgresJobQueue, initialize_background  # noqa: E402


TEST_DSN = os.environ.get("LAT_TEST_POSTGRES_DSN")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not TEST_DSN, reason="Set LAT_TEST_POSTGRES_DSN to a disposable database"),
]


class Counter(TypedDict):
    value: int


@pytest.fixture
async def service(monkeypatch, tmp_path):
    schema = "lat_history_jobs_" + uuid4().hex
    for name, value in {
        "BACKGROUND_ENABLED": True,
        "AUTH_MODE": "trusted",
        "AUTH_SECRET": SecretStr("history-job-test-token"),
        "AUTH_SERVICE_USER_ID": "owner",
        "AUTH_USERS": {},
    }.items():
        monkeypatch.setattr(settings, name, value)
    calls = []
    builder = StateGraph(Counter)
    builder.add_node("step", lambda state: calls.append(state["value"]) or {"value": state["value"] + 1})
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    try:
        await initialize_background(TEST_DSN, schema=schema)
        async with (
            PostgresJobQueue(TEST_DSN, schema=schema) as queue,
            CoordinatedSqliteSaver.from_conn_string(str(tmp_path / "history.sqlite")) as saver,
        ):
            executor = AgentExecutor.__new__(AgentExecutor)
            executor.agents = {"test": Agent("test", "History recovery test", builder.compile(checkpointer=saver))}
            executor.concurrency = ConversationCoordinator(timeout=1)
            app = create_app()
            app.state.agent_executor = executor
            app.state.background_jobs = queue
            app.state.ready = True
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://history.test",
                headers={"Authorization": "Bearer history-job-test-token"},
            ) as client:
                yield SimpleNamespace(queue=queue, executor=executor, client=client, calls=calls)
            await app.state.blocking_executor.aclose(1)
    finally:
        async with await psycopg.AsyncConnection.connect(TEST_DSN, autocommit=True) as conn:
            await conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


async def test_history_clear_preserves_uncertain_job_until_completion(service):
    queue, executor = service.queue, service.executor
    request = ExecutionRequest(thread_id="work", input={"value": 1})
    job = await queue.submit("test", "owner", request, "first")

    async def started(checkpoint_id):
        assert checkpoint_id is None
        async with queue.pool.connection() as conn:
            await conn.execute("UPDATE lat_jobs SET status='running',started=true WHERE job_id=%s", (job.job_id,))

    result = await executor.execute("test", request, owner_id="owner", operation_id=job.job_id, on_start=started)
    # Simulate worker loss after durable graph output, before the job result write.
    for active_status in ("queued", "running", "cancelling"):
        async with queue.pool.connection() as conn:
            await conn.execute("UPDATE lat_jobs SET status=%s WHERE job_id=%s", (active_status, job.job_id))
        response = await service.client.request("DELETE", "/test/history/clear", json={"thread_id": "work"})
        assert response.status_code == 409, response.text
        assert (
            await executor.inspect_execution("test", "work", owner_id="owner")
        ).checkpoint_id == result.checkpoint_id

    recovered = await executor.execute("test", request, owner_id="owner", operation_id=job.job_id, retry=True)
    assert recovered == result
    assert service.calls == [1]
    async with queue.pool.connection() as conn:
        await conn.execute(
            "UPDATE lat_jobs SET status='completed',result=%s WHERE job_id=%s",
            (Jsonb(recovered.model_dump(mode="json")), job.job_id),
        )
    # Active jobs belonging to another owner, agent, or thread must not block it.
    await queue.submit("test", "other-owner", request, "other-owner")
    await queue.submit("other-agent", "owner", request, "other-agent")
    await queue.submit("test", "owner", request.model_copy(update={"thread_id": "other-thread"}), "other-thread")
    response = await service.client.request("DELETE", "/test/history/clear", json={"thread_id": "work"})
    assert response.status_code == 200, response.text
    assert (await executor.inspect_execution("test", "work", owner_id="owner")).status == "empty"


async def test_history_clear_keeps_checkpoints_when_queue_check_fails(service, monkeypatch):
    result = await service.executor.execute(
        "test", ExecutionRequest(thread_id="work", input={"value": 1}), owner_id="owner"
    )

    async def unavailable(**kwargs):
        raise psycopg.OperationalError("private connection details")

    monkeypatch.setattr(service.queue, "has_active_jobs", unavailable)
    response = await service.client.request("DELETE", "/test/history/clear", json={"thread_id": "work"})
    assert response.status_code == 503, response.text
    assert "private" not in response.text
    snapshot = await service.executor.inspect_execution("test", "work", owner_id="owner")
    assert snapshot.checkpoint_id == result.checkpoint_id
