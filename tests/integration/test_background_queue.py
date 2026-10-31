"""Exercise durable job intake and execution with a disposable PostgreSQL database."""

import asyncio
import os
from contextlib import nullcontext, suppress
from types import SimpleNamespace
from typing import TypedDict
from uuid import uuid4

import psycopg
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from psycopg import sql

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.memory.concurrency import ConversationCoordinator
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.schema.execution import ExecutionRequest


pytest.importorskip("procrastinate")
from langgraph_agent_toolkit.core.background import (  # noqa: E402
    BackgroundUnavailable,
    JobConflict,
    PostgresJobQueue,
    initialize_background,
)


TEST_DSN = os.environ.get("LAT_TEST_POSTGRES_DSN")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not TEST_DSN, reason="Set LAT_TEST_POSTGRES_DSN to a disposable database"),
]


@pytest.fixture
async def queue_schema():
    name = "lat_jobs_test_" + uuid4().hex
    try:
        yield name
    finally:
        async with await psycopg.AsyncConnection.connect(TEST_DSN, autocommit=True) as conn:
            await conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(name)))


@pytest.fixture
async def queue(queue_schema):
    await initialize_background(TEST_DSN, schema=queue_schema)
    async with PostgresJobQueue(TEST_DSN, schema=queue_schema) as queue:
        yield queue


async def test_initialization_is_explicit_and_repeatable(queue_schema):
    with pytest.raises(BackgroundUnavailable, match="Initialize"):
        async with PostgresJobQueue(TEST_DSN, schema=queue_schema):
            pass
    await asyncio.gather(*(initialize_background(TEST_DSN, schema=queue_schema) for _ in range(2)))
    async with PostgresJobQueue(TEST_DSN, schema=queue_schema) as queue:
        first = await queue.submit("worker", "owner", ExecutionRequest(thread_id="thread"), "first")
    await initialize_background(TEST_DSN, schema=queue_schema)
    async with PostgresJobQueue(TEST_DSN, schema=queue_schema) as reopened:
        assert (await reopened.get(first.job_id, "owner")).status == "queued"


@pytest.mark.parametrize("previous_version", [1, 2])
async def test_explicit_initialization_upgrades_prior_schema_without_losing_jobs(queue_schema, previous_version):
    await initialize_background(TEST_DSN, schema=queue_schema)
    async with PostgresJobQueue(TEST_DSN, schema=queue_schema) as queue:
        job = await queue.submit("worker", "owner", ExecutionRequest(thread_id="thread"), "retained")
        async with queue.pool.connection() as conn:
            if previous_version == 1:
                await conn.execute("DROP INDEX lat_jobs_recovery_active,lat_queue_recovery_doing")
            await conn.execute("DROP INDEX lat_jobs_active_thread")
            await conn.execute("UPDATE lat_background_version SET version=%s", (previous_version,))
    with pytest.raises(BackgroundUnavailable, match="schema version differs"):
        async with PostgresJobQueue(TEST_DSN, schema=queue_schema):
            pass
    await asyncio.gather(*(initialize_background(TEST_DSN, schema=queue_schema) for _ in range(2)))
    async with PostgresJobQueue(TEST_DSN, schema=queue_schema) as queue:
        assert (await queue.get(job.job_id, "owner")).status == "queued"
        async with queue.pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT version,to_regclass('lat_jobs_recovery_active') IS NOT NULL AS active_index,"
                    "to_regclass('lat_queue_recovery_doing') IS NOT NULL AS doing_index,"
                    "to_regclass('lat_jobs_active_thread') IS NOT NULL AS thread_index "
                    "FROM lat_background_version"
                )
            ).fetchone()
        assert row == {"version": 3, "active_index": True, "doing_index": True, "thread_index": True}


async def test_duplicate_submission_is_atomic_and_owner_scoped(queue):
    request = ExecutionRequest(thread_id="thread", input={"value": 1})
    jobs = await asyncio.gather(*(queue.submit("worker", "owner", request, "same-key") for _ in range(12)))
    assert len({job.job_id for job in jobs}) == 1
    with pytest.raises(JobConflict):
        await queue.submit("worker", "owner", request.model_copy(update={"input": {"value": 2}}), "same-key")
    assert await queue.get(jobs[0].job_id, "other-owner") is None
    assert await queue.cancel(jobs[0].job_id, "other-owner") is None
    assert await queue.get("invalid-uuid", "owner") is None
    other = await queue.submit("worker", "other-owner", request, "same-key")
    assert other.job_id != jobs[0].job_id
    async with queue.pool.connection() as conn:
        counts = await (
            await conn.execute(
                "SELECT (SELECT count(*) FROM lat_jobs) AS ledger, (SELECT count(*) FROM procrastinate_jobs) AS queue"
            )
        ).fetchone()
    assert counts == {"ledger": 2, "queue": 2}


async def test_enqueue_failure_rolls_back_job_record(queue, monkeypatch):
    async def fail(**kwargs):
        raise RuntimeError("Queue insertion failed")

    monkeypatch.setattr(queue.task, "configure", lambda **kwargs: SimpleNamespace(defer_async=fail))
    with pytest.raises(RuntimeError, match="Queue insertion"):
        await queue.submit("worker", "owner", ExecutionRequest(thread_id="thread"), "failed-insert")
    async with queue.pool.connection() as conn:
        row = await (await conn.execute("SELECT count(*) AS count FROM lat_jobs")).fetchone()
    assert row["count"] == 0


async def test_queued_cancellation_is_durable_and_idempotent(queue):
    job = await queue.submit("worker", "owner", ExecutionRequest(thread_id="thread"), "cancel")
    cancelled = await queue.cancel(job.job_id, "owner")
    assert cancelled.status == "cancelled"
    assert (await queue.cancel(job.job_id, "owner")).status == "cancelled"
    async with queue.pool.connection() as conn:
        row = await (await conn.execute("SELECT status FROM procrastinate_jobs")).fetchone()
    assert row["status"] == "cancelled"


async def test_concurrent_recovery_does_not_requeue_a_replacement_worker(queue):
    job = await queue.submit("worker", "owner", ExecutionRequest(thread_id="thread"), "stalled")
    async with queue.pool.connection() as conn:
        row = await (
            await conn.execute(
                "UPDATE lat_jobs SET status='running' WHERE job_id=%s RETURNING queue_job_id", (job.job_id,)
            )
        ).fetchone()
        queue_id = row["queue_job_id"]
        await conn.execute("UPDATE procrastinate_jobs SET status='doing',worker_id=NULL WHERE id=%s", (queue_id,))
    await asyncio.gather(*(queue.recover_stalled(stalled_after=1) for _ in range(12)))
    async with queue.pool.connection() as conn:
        row = await (
            await conn.execute("SELECT status,attempts FROM procrastinate_jobs WHERE id=%s", (queue_id,))
        ).fetchone()
    assert row == {"status": "todo", "attempts": 1}
    worker_id = await queue.app.job_manager.register_worker()
    try:
        async with queue.pool.connection() as conn:
            await conn.execute(
                "UPDATE procrastinate_jobs SET status='doing',worker_id=%s WHERE id=%s", (worker_id, queue_id)
            )
        await asyncio.gather(*(queue.recover_stalled(stalled_after=60) for _ in range(6)))
        async with queue.pool.connection() as conn:
            row = await (
                await conn.execute("SELECT status,attempts FROM procrastinate_jobs WHERE id=%s", (queue_id,))
            ).fetchone()
        assert row == {"status": "doing", "attempts": 1}
    finally:
        await queue.app.job_manager.unregister_worker(worker_id)


async def test_recovery_skips_a_locked_queue_row_and_recovers_other_jobs(queue):
    jobs = [
        await queue.submit("worker", "owner", ExecutionRequest(thread_id=f"thread-{index}"), f"locked-{index}")
        for index in range(2)
    ]
    async with queue.pool.connection() as conn:
        await conn.execute("UPDATE lat_jobs SET status='running'")
        await conn.execute("UPDATE procrastinate_jobs SET status='doing',worker_id=NULL")
        rows = await (await conn.execute("SELECT id FROM procrastinate_jobs ORDER BY id")).fetchall()
    async with queue.pool.connection() as held, held.transaction():
        await held.execute("SELECT id FROM procrastinate_jobs WHERE id=%s FOR UPDATE", (rows[0]["id"],))
        await asyncio.wait_for(queue.recover_stalled(stalled_after=1), timeout=2)
        async with queue.pool.connection() as conn:
            states = await (await conn.execute("SELECT status FROM procrastinate_jobs ORDER BY id")).fetchall()
        assert states == [{"status": "doing"}, {"status": "todo"}]
        assert (await queue.get(jobs[0].job_id, "owner")).attempts == 0
    await queue.recover_stalled(stalled_after=1)
    async with queue.pool.connection() as conn:
        states = await (await conn.execute("SELECT status FROM procrastinate_jobs ORDER BY id")).fetchall()
    assert states == [{"status": "todo"}, {"status": "todo"}]


async def test_recovery_candidates_preserve_terminal_jobs_and_repair_active_deliveries(queue):
    cases = [
        ("failed", "running", "todo", "running"),
        ("aborted", "running", "aborted", "failed"),
        ("failed", "completed", "failed", "completed"),
        ("doing", "completed", "todo", "completed"),
    ]
    jobs = []
    for index, (queue_status, ledger_status, _, _) in enumerate(cases):
        job = await queue.submit("worker", "owner", ExecutionRequest(thread_id=f"candidate-{index}"), str(index))
        jobs.append(job)
        async with queue.pool.connection() as conn:
            await conn.execute("UPDATE lat_jobs SET status=%s WHERE job_id=%s", (ledger_status, job.job_id))
            await conn.execute(
                "UPDATE procrastinate_jobs SET status=%s,worker_id=NULL WHERE id="
                "(SELECT queue_job_id FROM lat_jobs WHERE job_id=%s)",
                (queue_status, job.job_id),
            )
    await queue.recover_stalled(stalled_after=1)
    for job, (_, _, queue_status, ledger_status) in zip(jobs, cases):
        assert (await queue.get(job.job_id, "owner")).status == ledger_status
        async with queue.pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT q.status FROM procrastinate_jobs q JOIN lat_jobs j ON j.queue_job_id=q.id "
                    "WHERE j.job_id=%s",
                    (job.job_id,),
                )
            ).fetchone()
        assert row["status"] == queue_status


async def test_failed_delivery_before_execution_has_bounded_retries(queue):
    job = await queue.submit("worker", "owner", ExecutionRequest(thread_id="thread"), "failed-delivery")
    async with queue.pool.connection() as conn:
        await conn.execute(
            "UPDATE procrastinate_jobs SET status='failed',attempts=%s WHERE id="
            "(SELECT queue_job_id FROM lat_jobs WHERE job_id=%s)",
            (queue.max_attempts, job.job_id),
        )
    await queue.recover_stalled()
    failed = await queue.get(job.job_id, "owner")
    assert failed.status == "failed"
    assert failed.error == "Worker recovery attempts exhausted"
    await queue.recover_stalled()
    assert (await queue.get(job.job_id, "owner")).status == "failed"


async def test_cancelling_a_lost_worker_releases_its_queue_lock(queue):
    job = await queue.submit("worker", "owner", ExecutionRequest(thread_id="thread"), "lost-cancel")
    async with queue.pool.connection() as conn:
        await conn.execute("UPDATE lat_jobs SET status='running' WHERE job_id=%s", (job.job_id,))
        await conn.execute("UPDATE procrastinate_jobs SET status='doing',worker_id=NULL")
    assert (await queue.cancel(job.job_id, "owner")).status == "cancelling"
    await queue.recover_stalled()
    assert (await queue.get(job.job_id, "owner")).status == "cancelled"
    async with queue.pool.connection() as conn:
        row = await (await conn.execute("SELECT status FROM procrastinate_jobs")).fetchone()
    assert row["status"] == "failed"
    # Procrastinate forbids retrying a failed queue row with abort_requested.
    await queue.recover_stalled()
    assert (await queue.get(job.job_id, "owner")).status == "cancelled"


class State(TypedDict):
    value: int


def make_runtime(node):
    builder = StateGraph(State)
    builder.add_node("work", node)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    executor = AgentExecutor.__new__(AgentExecutor)
    executor.agents = {"worker": Agent("worker", "Generic worker", graph, EmptyObservability())}
    executor.concurrency = ConversationCoordinator()
    return SimpleNamespace(executor=executor, llm_transport_manager=SimpleNamespace(bind=nullcontext))


async def wait_job(queue, job_id, statuses):
    async with asyncio.timeout(10):
        while True:
            job = await queue.get(job_id, "owner")
            if job.status in statuses:
                return job
            await asyncio.sleep(0.02)


async def wait_queue_status(queue, job_id, status):
    async with asyncio.timeout(10):
        while True:
            async with queue.pool.connection() as conn:
                row = await (
                    await conn.execute(
                        "SELECT q.status FROM procrastinate_jobs q JOIN lat_jobs j ON j.queue_job_id=q.id "
                        "WHERE j.job_id=%s",
                        (job_id,),
                    )
                ).fetchone()
            if row["status"] == status:
                return
            await asyncio.sleep(0.02)


async def test_real_queue_worker_returns_structured_result_and_recovers_lost_delivery(queue):
    calls = []

    def work(state):
        calls.append(state["value"])
        return {"value": state["value"] + 1}

    runtime = make_runtime(work)
    job = await queue.submit("worker", "owner", ExecutionRequest(thread_id="thread", input={"value": 3}), "run")
    worker = asyncio.create_task(queue.run_worker(runtime, concurrency=2, install_signal_handlers=False))
    try:
        result = await wait_job(queue, job.job_id, {"completed", "failed"})
        assert result.status == "completed", result.error
        assert result.result.output == {"value": 4}
        assert result.attempts == 1
        assert calls == [3]
        await wait_queue_status(queue, job.job_id, "succeeded")
        # Simulate a lost result write after the completed checkpoint was saved.
        async with queue.pool.connection() as conn:
            row = await (
                await conn.execute(
                    "UPDATE lat_jobs SET status='running',result=NULL WHERE job_id=%s RETURNING queue_job_id",
                    (job.job_id,),
                )
            ).fetchone()
            await conn.execute("UPDATE procrastinate_jobs SET status='failed' WHERE id=%s", (row["queue_job_id"],))
        await queue.recover_stalled()
        recovered = await wait_job(queue, job.job_id, {"completed", "failed"})
        assert recovered.status == "completed", recovered.error
        assert recovered.result.output == {"value": 4}
        assert recovered.attempts == 2
        assert calls == [3]
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker


async def test_active_cancellation_waits_for_graph_cleanup(queue):
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    async def work(state):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    runtime = make_runtime(work)
    job = await queue.submit("worker", "owner", ExecutionRequest(thread_id="thread", input={"value": 3}), "cancel-run")
    worker = asyncio.create_task(queue.run_worker(runtime, concurrency=2, install_signal_handlers=False))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert (await queue.cancel(job.job_id, "owner")).status == "cancelling"
        cancelled = await wait_job(queue, job.job_id, {"cancelled", "failed"})
        assert cancelled.status == "cancelled"
        assert cleaned.is_set()
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker


async def test_recovery_runs_with_full_capacity_and_an_orphaned_legacy_maintenance_lock(queue):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def work(state):
        calls.append(state["value"])
        if state["value"] == 1:
            entered.set()
            await release.wait()
        return {"value": state["value"] + 1}

    # Old deployments scheduled recovery as a locked queue job. Its orphan
    # must not block recovery after a worker restarts with the current code.
    async def legacy_recover(timestamp):
        raise AssertionError("Legacy maintenance must not execute")

    legacy_task = queue.app.task(
        name="lat.recover", queue="lat-maintenance", lock="lat-recovery", queueing_lock="lat-recovery"
    )(legacy_recover)
    await legacy_task.defer_async(timestamp=1)
    old_worker = await queue.app.job_manager.register_worker()
    await queue.app.job_manager.fetch_job(queues=["lat-maintenance"], worker_id=old_worker)
    await queue.app.job_manager.unregister_worker(old_worker)

    runtime = make_runtime(work)
    busy = await queue.submit("worker", "owner", ExecutionRequest(thread_id="busy", input={"value": 1}), "busy")
    worker = asyncio.create_task(
        queue.run_worker(runtime, concurrency=1, recovery_interval=1, install_signal_handlers=False)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        lost = await queue.submit("worker", "owner", ExecutionRequest(thread_id="lost", input={"value": 2}), "lost")
        async with queue.pool.connection() as conn:
            row = await (
                await conn.execute("SELECT queue_job_id FROM lat_jobs WHERE job_id=%s", (lost.job_id,))
            ).fetchone()
            await conn.execute(
                "UPDATE procrastinate_jobs SET status='doing',worker_id=NULL WHERE id=%s", (row["queue_job_id"],)
            )
        await wait_queue_status(queue, lost.job_id, "todo")
        assert calls == [1]
        assert not release.is_set()
        async with queue.pool.connection() as conn:
            maintenance = await (
                await conn.execute("SELECT status FROM procrastinate_jobs WHERE queue_name='lat-maintenance'")
            ).fetchall()
        assert maintenance == [{"status": "doing"}]
        release.set()
        assert (await wait_job(queue, busy.job_id, {"completed", "failed"})).status == "completed"
        assert (await wait_job(queue, lost.job_id, {"completed", "failed"})).status == "completed"
        assert calls == [1, 2]
    finally:
        release.set()
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker


async def test_failed_acknowledgement_stops_worker_and_replacement_unblocks_thread(queue, monkeypatch):
    calls = []

    def work(state):
        calls.append(state["value"])
        return {"value": state["value"] + 1}

    runtime = make_runtime(work)
    first = await queue.submit("worker", "owner", ExecutionRequest(thread_id="same", input={"value": 1}), "first")
    finish = queue.app.job_manager.finish_job
    failed = False

    async def fail_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise psycopg.OperationalError("Synthetic acknowledgement failure")
        return await finish(*args, **kwargs)

    monkeypatch.setattr(queue.app.job_manager, "finish_job", fail_once)
    with pytest.raises(BackgroundUnavailable, match="acknowledgement failed"):
        await asyncio.wait_for(queue.run_worker(runtime, concurrency=1, install_signal_handlers=False), timeout=5)
    assert queue.runtime is None
    assert (await queue.get(first.job_id, "owner")).status == "completed"
    await wait_queue_status(queue, first.job_id, "doing")

    second = await queue.submit("worker", "owner", ExecutionRequest(thread_id="same", input={"value": 2}), "second")
    replacement = asyncio.create_task(queue.run_worker(runtime, concurrency=1, install_signal_handlers=False))
    try:
        assert (await wait_job(queue, second.job_id, {"completed", "failed"})).status == "completed"
        await wait_queue_status(queue, first.job_id, "succeeded")
        await wait_queue_status(queue, second.job_id, "succeeded")
        assert calls == [1, 2]
        assert (await queue.get(first.job_id, "owner")).attempts == 1
    finally:
        replacement.cancel()
        with suppress(asyncio.CancelledError):
            await replacement


async def test_recovery_failure_stops_and_drains_worker(queue, monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()
    original = queue.recover_stalled
    calls = 0

    async def work(state):
        entered.set()
        try:
            await release.wait()
            return state
        finally:
            cleaned.set()

    async def recover(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return await original(**kwargs)
        await asyncio.wait_for(entered.wait(), timeout=5)
        release.set()
        raise psycopg.OperationalError("Synthetic recovery connection failure")

    monkeypatch.setattr(queue, "recover_stalled", recover)
    runtime = make_runtime(work)
    job = await queue.submit("worker", "owner", ExecutionRequest(thread_id="thread", input={"value": 1}), "run")
    try:
        with pytest.raises(psycopg.OperationalError, match="Synthetic recovery"):
            await asyncio.wait_for(
                queue.run_worker(runtime, recovery_interval=1, install_signal_handlers=False), timeout=5
            )
        assert cleaned.is_set()
        assert queue.runtime is None
        assert (await queue.get(job.job_id, "owner")).status == "completed"
        async with queue.pool.connection() as conn:
            row = await (await conn.execute("SELECT count(*) AS workers FROM procrastinate_workers")).fetchone()
        assert row["workers"] == 0
    finally:
        release.set()
