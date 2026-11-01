"""Persist job records and enqueue them in one PostgreSQL transaction."""

import asyncio
import hashlib
import importlib.metadata
import json
import re
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from psycopg import AsyncConnection, sql
from psycopg.types.json import Jsonb

from langgraph_agent_toolkit.core.identity import storage_thread_id, validate_conversation_identifiers
from langgraph_agent_toolkit.core.memory.pool import CheckedAsyncConnectionPool as AsyncConnectionPool
from langgraph_agent_toolkit.core.postgres import (
    postgres_connection_kwargs,
    postgres_connection_string,
    postgres_pool_options,
)
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.logging import logger
from langgraph_agent_toolkit.schema.execution import ExecutionRequest
from langgraph_agent_toolkit.schema.jobs import BackgroundJob


if TYPE_CHECKING:
    from procrastinate.job_context import JobContext
    from procrastinate.middleware import AsyncCallNext
    from procrastinate.worker import Worker

    from langgraph_agent_toolkit.core.runtime import AgentRuntime


_TERMINAL = {"completed", "interrupted", "failed", "cancelled"}
_QUEUE = "lat-runs"
_TASK = "lat.execute"
_SCHEMA_VERSION = 3
_HISTORY_GUARD_INDEX = """
CREATE INDEX lat_jobs_active_thread ON lat_jobs(owner_id,agent_id,md5(thread_id))
    WHERE status IN ('queued','running','cancelling');
"""
_RECOVERY_INDEXES = """
CREATE INDEX lat_jobs_recovery_active ON lat_jobs(queue_job_id)
    WHERE status IN ('queued','running','cancelling');
CREATE INDEX lat_queue_recovery_doing ON procrastinate_jobs(id)
    WHERE status = 'doing';
"""
_DDL = """
CREATE TABLE lat_background_version (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    version integer NOT NULL,
    queue_version text NOT NULL
);
CREATE TABLE lat_jobs (
    job_id uuid PRIMARY KEY,
    queue_job_id bigint UNIQUE REFERENCES procrastinate_jobs(id),
    owner_id text NOT NULL,
    agent_id text NOT NULL,
    thread_id text NOT NULL,
    idempotency_key text NOT NULL,
    fingerprint text NOT NULL,
    request jsonb NOT NULL,
    status text NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued','running','cancelling','cancelled','completed','interrupted','failed')),
    started boolean NOT NULL DEFAULT false,
    baseline_checkpoint_id text,
    attempts integer NOT NULL DEFAULT 0,
    result jsonb,
    error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (owner_id, agent_id, idempotency_key)
);
CREATE INDEX lat_jobs_owner_created ON lat_jobs(owner_id, created_at DESC);
"""


class JobConflict(Exception):
    """The operation conflicts with an existing durable job."""


class BackgroundUnavailable(Exception):
    """The optional queue has not been configured or initialized."""


def _procrastinate():
    try:
        import procrastinate
    except ImportError as exc:
        raise BackgroundUnavailable("Install langgraph-agent-toolkit[background] to use durable jobs") from exc
    return procrastinate


def _schema_name(value: str) -> str:
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", value):
        raise ValueError("Background schema must be a lowercase PostgreSQL identifier")
    return value


def _job(row: dict[str, Any]) -> BackgroundJob:
    return BackgroundJob.model_validate({**row, "job_id": str(row["job_id"])})


def _valid_job_id(value: str) -> bool:
    try:
        UUID(value)
    except (ValueError, TypeError, AttributeError):
        return False
    return True


async def _configure_connection(conn: AsyncConnection) -> None:
    # Procrastinate sets autocommit itself on its separate LISTEN connection.
    await conn.set_autocommit(True)


class PostgresJobQueue:
    """Own a bounded pool shared by job records and the optional task queue.

    Opening the queue checks its schema. It never creates or migrates tables.
    Use ``initialize_background`` once with a deployment migration identity.
    """

    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "lat_background",
        pool_size: int = 5,
        pool_max_size: int | None = None,
        max_attempts: int = 3,
        execution_timeout: float = 300,
    ):
        procrastinate = _procrastinate()
        self.schema = _schema_name(schema)
        if pool_max_size is not None:
            if pool_size != 5 and pool_size != pool_max_size:
                raise ValueError("Use only one background pool size")
            pool_size = pool_max_size
        if pool_size < 2 or max_attempts < 1 or execution_timeout <= 0:
            raise ValueError("Background pool size must be at least 2; attempts and timeout must be positive")
        self.max_attempts = max_attempts
        self.execution_timeout = execution_timeout
        conninfo = postgres_connection_string(settings, dsn=dsn)
        pool_options = postgres_pool_options(
            settings,
            min_size=1,
            max_size=pool_size,
            schema=self.schema,
            app_prefix="background",
            conninfo=conninfo,
            force_schema=True,
        )
        # Procrastinate sets this keyword on its separate LISTEN connection.
        # The configure callback enables it for pooled connections as well.
        pool_options["kwargs"].pop("autocommit")
        self.pool = AsyncConnectionPool(
            conninfo,
            **pool_options,
            configure=_configure_connection,
        )
        self.app = procrastinate.App(connector=procrastinate.PsycopgConnector())
        self.task = self.app.task(
            name=_TASK,
            queue=_QUEUE,
            pass_context=True,
            retry=procrastinate.RetryStrategy(max_attempts=max_attempts, exponential_wait=2),
        )(self._execute_job)
        self.runtime: AgentRuntime | None = None
        self._resources: AsyncExitStack | None = None

    async def __aenter__(self) -> "PostgresJobQueue":
        resources = AsyncExitStack()
        try:
            await resources.enter_async_context(self.pool)
            await self.pool.wait(timeout=settings.POSTGRES_POOL_TIMEOUT)
            await resources.enter_async_context(self.app.open_async(pool=self.pool))
            async with self.pool.connection() as conn:
                exists = await (await conn.execute("SELECT to_regclass('lat_background_version') AS name")).fetchone()
                if exists["name"] is None:
                    raise BackgroundUnavailable("Initialize the background database schema before starting the service")
                version = await (
                    await conn.execute("SELECT version, queue_version FROM lat_background_version")
                ).fetchone()
                if (
                    not version
                    or version["version"] != _SCHEMA_VERSION
                    or version["queue_version"] != importlib.metadata.version("procrastinate")
                ):
                    raise BackgroundUnavailable(
                        "Background schema version differs; apply a reviewed database migration"
                    )
            self._resources = resources
            return self
        except BaseException:
            await resources.aclose()
            raise

    async def __aexit__(self, *exc):
        if self._resources is not None:
            await self._resources.__aexit__(*exc)
            self._resources = None

    async def submit(
        self, agent_id: str, owner_id: str, request: ExecutionRequest, idempotency_key: str
    ) -> BackgroundJob:
        """Atomically save a request and enqueue it, or return its prior job."""
        validate_conversation_identifiers(owner_id, agent_id, request.thread_id)
        if request.user_id is not None and request.user_id != owner_id:
            raise ValueError("user_id must match the execution owner")
        if not idempotency_key or len(idempotency_key) > 128 or "\x00" in idempotency_key:
            raise ValueError("Idempotency key must contain 1 to 128 characters without NUL")
        if request.timeout_seconds > self.execution_timeout:
            raise ValueError("Execution timeout exceeds the configured background limit")
        payload = request.model_dump(mode="json")
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        job_id = uuid4()
        async with self.pool.connection() as conn, conn.transaction():
            inserted = await (
                await conn.execute(
                    "INSERT INTO lat_jobs(job_id,owner_id,agent_id,thread_id,idempotency_key,fingerprint,request) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (owner_id,agent_id,idempotency_key) DO NOTHING "
                    "RETURNING job_id",
                    (job_id, owner_id, agent_id, request.thread_id, idempotency_key, fingerprint, Jsonb(payload)),
                )
            ).fetchone()
            if inserted:
                queue_id = await self.task.configure(
                    connection=conn, lock=storage_thread_id(owner_id, agent_id, request.thread_id)
                ).defer_async(job_id=str(job_id))
                await conn.execute("UPDATE lat_jobs SET queue_job_id=%s WHERE job_id=%s", (queue_id, job_id))
            row = await (
                await conn.execute(
                    "SELECT * FROM lat_jobs WHERE owner_id=%s AND agent_id=%s AND idempotency_key=%s",
                    (owner_id, agent_id, idempotency_key),
                )
            ).fetchone()
            if row["fingerprint"] != fingerprint:
                raise JobConflict("Idempotency key was already used for a different request")
            return _job(row)

    async def get(self, job_id: str, owner_id: str) -> BackgroundJob | None:
        """Read only a job owned by the authenticated identity."""
        if not _valid_job_id(job_id):
            return None
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute("SELECT * FROM lat_jobs WHERE job_id=%s AND owner_id=%s", (job_id, owner_id))
            ).fetchone()
        return _job(row) if row else None

    async def has_active_jobs(self, agent_id: str, owner_id: str, thread_id: str) -> bool:
        """Check for recoverable work before deleting a conversation's checkpoints.

        The caller holds the conversation lock. This read does not acquire job
        row locks. A newly submitted job cannot start until that lock is released.
        """
        validate_conversation_identifiers(owner_id, agent_id, thread_id)
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT EXISTS (SELECT 1 FROM lat_jobs WHERE owner_id=%s AND agent_id=%s AND thread_id=%s "
                    "AND md5(thread_id)=md5(%s) "
                    "AND status IN ('queued','running','cancelling')) AS active",
                    # Hash the index key to bound its UTF-8 size. The original
                    # thread comparison prevents hash collisions from matching.
                    (owner_id, agent_id, thread_id, thread_id),
                )
            ).fetchone()
        return row["active"]

    async def cancel(self, job_id: str, owner_id: str) -> BackgroundJob | None:
        """Persist cancellation and notify the worker in the same transaction."""
        if not _valid_job_id(job_id):
            return None
        async with self.pool.connection() as conn, conn.transaction():
            row = await (
                await conn.execute(
                    "SELECT * FROM lat_jobs WHERE job_id=%s AND owner_id=%s FOR UPDATE", (job_id, owner_id)
                )
            ).fetchone()
            if not row:
                return None
            if row["status"] not in _TERMINAL:
                queue_status = await self.app.job_manager.get_job_status_async(row["queue_job_id"], connection=conn)
                status = "cancelling" if queue_status.value == "doing" else "cancelled"
                await self.app.job_manager.cancel_job_by_id_async(row["queue_job_id"], abort=True, connection=conn)
                row = await (
                    await conn.execute(
                        "UPDATE lat_jobs SET status=%s,updated_at=now() WHERE job_id=%s RETURNING *", (status, job_id)
                    )
                ).fetchone()
            return _job(row)

    async def _execute_job(self, context: "JobContext", *, job_id: str) -> None:
        if self.runtime is None:
            raise BackgroundUnavailable("A worker runtime is required to execute background jobs")
        async with self.pool.connection() as conn, conn.transaction():
            row = await (await conn.execute("SELECT * FROM lat_jobs WHERE job_id=%s FOR UPDATE", (job_id,))).fetchone()
            if row is None or row["status"] in _TERMINAL:
                return
            if row["status"] == "cancelling":
                await conn.execute("UPDATE lat_jobs SET status='cancelled',updated_at=now() WHERE job_id=%s", (job_id,))
                return
            if row["attempts"] >= self.max_attempts:
                await conn.execute(
                    "UPDATE lat_jobs SET status='failed',error='Worker recovery attempts exhausted',updated_at=now() "
                    "WHERE job_id=%s",
                    (job_id,),
                )
                return
            await conn.execute(
                "UPDATE lat_jobs SET status='running',attempts=attempts+1,updated_at=now() WHERE job_id=%s", (job_id,)
            )

        async def record_start(checkpoint_id: str | None) -> None:
            async with self.pool.connection() as conn, conn.transaction():
                current = await (
                    await conn.execute("SELECT status FROM lat_jobs WHERE job_id=%s FOR UPDATE", (job_id,))
                ).fetchone()
                if current["status"] != "running":
                    raise asyncio.CancelledError
                await conn.execute(
                    "UPDATE lat_jobs SET started=true,baseline_checkpoint_id=%s,updated_at=now() WHERE job_id=%s",
                    (checkpoint_id, job_id),
                )

        try:
            result = await self.runtime.executor.execute(
                row["agent_id"],
                ExecutionRequest.model_validate(row["request"]),
                owner_id=row["owner_id"],
                operation_id=job_id,
                retry=row["started"],
                baseline_checkpoint_id=row["baseline_checkpoint_id"],
                on_start=record_start,
            )
        except asyncio.CancelledError:
            async with self.pool.connection() as conn:
                await conn.execute(
                    "UPDATE lat_jobs SET status=CASE WHEN status='cancelling' THEN 'cancelled' ELSE 'running' END, "
                    "updated_at=now() WHERE job_id=%s AND status IN ('running','cancelling')",
                    (job_id,),
                )
            raise
        except Exception as exc:
            # Exception messages can contain model input or provider credentials.
            logger.warning("Background operation {} failed ({})", job_id, type(exc).__name__)
            async with self.pool.connection() as conn:
                await conn.execute(
                    "UPDATE lat_jobs SET status=CASE WHEN status='cancelling' THEN 'cancelled' ELSE 'failed' END, "
                    "error=%s,updated_at=now() WHERE job_id=%s AND status IN ('running','cancelling')",
                    (f"Execution failed ({type(exc).__name__})", job_id),
                )
            return
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE lat_jobs SET status=%s,result=%s,updated_at=now() WHERE job_id=%s "
                "AND status IN ('running','cancelling')",
                (result.status, Jsonb(result.model_dump(mode="json")), job_id),
            )

    async def recover_stalled(self, *, stalled_after: float = 60) -> None:
        """Recover lost workers and failed result persistence without replaying input."""
        procrastinate = _procrastinate()
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT candidates.job_id FROM ("
                    "SELECT j.job_id,q.id FROM procrastinate_jobs q JOIN lat_jobs j ON j.queue_job_id=q.id "
                    "WHERE q.status='doing' AND (q.worker_id IS NULL OR EXISTS ("
                    "SELECT 1 FROM procrastinate_workers w WHERE w.id=q.worker_id "
                    "AND w.last_heartbeat < now() - %s * interval '1 second')) "
                    "UNION ALL "
                    "SELECT j.job_id,q.id FROM lat_jobs j JOIN procrastinate_jobs q ON q.id=j.queue_job_id "
                    "WHERE j.status IN ('queued','running','cancelling') AND q.status IN ('failed','aborted')"
                    ") candidates ORDER BY candidates.id LIMIT 100",
                    (stalled_after,),
                )
            ).fetchall()
        for row in rows:
            async with self.pool.connection() as conn, conn.transaction():
                # Match cancellation's ledger-before-queue lock order. Recheck the
                # queue after locking; another worker can already have recovered it.
                job = await (
                    await conn.execute(
                        "SELECT * FROM lat_jobs WHERE job_id=%s FOR UPDATE SKIP LOCKED", (row["job_id"],)
                    )
                ).fetchone()
                if job is None:
                    continue
                candidate = await (
                    await conn.execute(
                        "SELECT q.status,q.attempts,q.abort_requested FROM procrastinate_jobs q WHERE q.id=%s AND ("
                        "q.status IN ('failed','aborted') OR (q.status='doing' AND (q.worker_id IS NULL OR EXISTS ("
                        "SELECT 1 FROM procrastinate_workers w WHERE w.id=q.worker_id "
                        "AND w.last_heartbeat < now() - %s * interval '1 second')))) "
                        "FOR UPDATE OF q SKIP LOCKED",
                        (job["queue_job_id"], stalled_after),
                    )
                ).fetchone()
                if candidate is None:
                    continue
                if candidate["status"] == "aborted" or (
                    candidate["status"] == "failed"
                    and (candidate["attempts"] >= self.max_attempts or candidate["abort_requested"])
                ):
                    if job["status"] not in _TERMINAL:
                        await conn.execute(
                            "UPDATE lat_jobs SET status="
                            "CASE WHEN status='cancelling' THEN 'cancelled' ELSE 'failed' END, "
                            "error='Worker recovery attempts exhausted',updated_at=now() WHERE job_id=%s",
                            (job["job_id"],),
                        )
                    continue
                if candidate["status"] == "failed" and job["status"] in _TERMINAL:
                    continue
                if job["status"] == "cancelling":
                    await conn.execute(
                        "UPDATE lat_jobs SET status='cancelled',updated_at=now() WHERE job_id=%s", (job["job_id"],)
                    )
                    if candidate["status"] == "failed":
                        continue
                    await self.app.job_manager.cancel_job_by_id_async(job["queue_job_id"], abort=True, connection=conn)
                # Reuse the library's transition in this transaction. Its public
                # retry helper does not accept an external connection.
                await self.app.connector.execute_query_one_async_with_connection(
                    conn,
                    query=procrastinate.sql.queries["retry_job"],
                    job_id=job["queue_job_id"],
                    retry_at=datetime.now(timezone.utc),
                    new_priority=None,
                    new_queue_name=None,
                    new_lock=None,
                )

    async def run_worker(
        self,
        runtime: "AgentRuntime",
        *,
        concurrency: int = 4,
        recovery_interval: int = 10,
        stalled_after: float = 60,
        install_signal_handlers: bool = True,
    ) -> None:
        """Supervise the library worker and recovery outside execution slots."""
        if recovery_interval < 1 or stalled_after <= 0 or concurrency < 1:
            raise ValueError("Recovery interval, stalled timeout, and concurrency must be positive")
        if self.runtime is not None:
            raise BackgroundUnavailable("Use a separate queue instance for each worker runtime")
        self.runtime = runtime
        bookkeeping_failure: BaseException | None = None

        async def supervise_job(call_next: "AsyncCallNext", context: "JobContext", worker: "Worker") -> Any:
            # Middleware runs inside the library's delivery task. Its completion
            # includes the final queue acknowledgement, after middleware returns.
            def completed(task: asyncio.Task) -> None:
                nonlocal bookkeeping_failure
                if not task.cancelled() and (error := task.exception()) is not None:
                    bookkeeping_failure = bookkeeping_failure or error
                    worker.stop()

            task = asyncio.current_task()
            assert task is not None
            task.add_done_callback(completed)
            return await call_next()

        async def recover() -> None:
            # A queued recovery task can strand its own lock after a worker dies.
            # This loop has no queue lock and consumes no execution slot or row.
            while True:
                await asyncio.sleep(recovery_interval)
                await self.recover_stalled(stalled_after=stalled_after)

        try:
            await self.recover_stalled(stalled_after=stalled_after)
            with runtime.llm_transport_manager.bind():
                worker_task = asyncio.create_task(
                    self.app.run_worker_async(
                        queues=[_QUEUE],
                        concurrency=concurrency,
                        install_signal_handlers=install_signal_handlers,
                        shutdown_graceful_timeout=10,
                        update_heartbeat_interval=min(10, stalled_after / 3),
                        stalled_worker_timeout=stalled_after,
                        fetch_job_polling_interval=1,
                        abort_job_polling_interval=1,
                        worker_middleware=[supervise_job],
                    ),
                    name="background-worker",
                )
                recovery_task = asyncio.create_task(recover(), name="background-recovery")
                tasks = (worker_task, recovery_task)
                try:
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    # The library drains running jobs before its task exits.
                    await asyncio.gather(*tasks, return_exceptions=True)
                if bookkeeping_failure is not None:
                    raise BackgroundUnavailable("Background job acknowledgement failed") from bookkeeping_failure
        finally:
            self.runtime = None


async def initialize_background(dsn: str, *, schema: str = "lat_background") -> None:
    """Create the dedicated schema or apply a supported toolkit schema upgrade."""
    procrastinate = _procrastinate()
    schema = _schema_name(schema)
    conninfo = postgres_connection_string(settings, dsn=dsn)
    connection_kwargs = postgres_connection_kwargs(
        settings, schema=schema, app_prefix="background-initialize", conninfo=conninfo, force_schema=True
    )
    # Schema migrations contain multiple statements. Do not prepare them.
    connection_kwargs["prepare_threshold"] = None
    async with await AsyncConnection.connect(
        conninfo,
        **connection_kwargs,
    ) as conn:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (f"lat-background:{schema}",))
            await conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
            await conn.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
            found = await (await conn.execute("SELECT to_regclass('lat_background_version') AS name")).fetchone()
            if found["name"] is None:
                queue_table = await (await conn.execute("SELECT to_regclass('procrastinate_jobs') AS name")).fetchone()
                if queue_table["name"] is not None:
                    raise BackgroundUnavailable("Use a fresh dedicated schema; an unrelated queue already exists")
                await conn.execute(procrastinate.schema.SchemaManager.get_schema())
                await conn.execute(_DDL)
                await conn.execute(_RECOVERY_INDEXES)
                await conn.execute(_HISTORY_GUARD_INDEX)
                await conn.execute(
                    "INSERT INTO lat_background_version(version,queue_version) VALUES (%s,%s)",
                    (_SCHEMA_VERSION, importlib.metadata.version("procrastinate")),
                )
            else:
                version = await (
                    await conn.execute("SELECT version,queue_version FROM lat_background_version")
                ).fetchone()
                queue_version = importlib.metadata.version("procrastinate")
                if version and version["queue_version"] == queue_version and version["version"] in {1, 2}:
                    # Only the explicit initializer applies this additive migration.
                    if version["version"] == 1:
                        await conn.execute(_RECOVERY_INDEXES)
                    await conn.execute(_HISTORY_GUARD_INDEX)
                    await conn.execute("UPDATE lat_background_version SET version=%s", (_SCHEMA_VERSION,))
                elif version != {
                    "version": _SCHEMA_VERSION,
                    "queue_version": queue_version,
                }:
                    raise BackgroundUnavailable(
                        "Background schema version differs; apply a reviewed database migration"
                    )


def queue_from_settings() -> PostgresJobQueue:
    """Build the optional queue without exposing its connection credentials."""
    return PostgresJobQueue(
        postgres_connection_string(
            settings,
            dsn=settings.BACKGROUND_POSTGRES_DSN.get_secret_value() if settings.BACKGROUND_POSTGRES_DSN else None,
        ),
        schema=settings.BACKGROUND_POSTGRES_SCHEMA,
        pool_size=settings.BACKGROUND_POSTGRES_POOL_SIZE,
        max_attempts=settings.BACKGROUND_MAX_ATTEMPTS,
        execution_timeout=settings.BACKGROUND_EXECUTION_TIMEOUT,
    )
