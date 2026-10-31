"""Verify queue configuration and checkpoint deletion guards on PostgreSQL."""

import os
import random
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from pydantic import SecretStr

from langgraph_agent_toolkit.core.memory.pool import CheckedAsyncConnectionPool
from langgraph_agent_toolkit.core.postgres import postgres_pool_options
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.schema.execution import ExecutionRequest


pytest.importorskip("procrastinate")
from langgraph_agent_toolkit.core.background import initialize_background, queue_from_settings  # noqa: E402


TEST_DSN = os.environ.get("LAT_TEST_POSTGRES_DSN")
pytestmark = [pytest.mark.postgres, pytest.mark.skipif(not TEST_DSN, reason="Set LAT_TEST_POSTGRES_DSN")]


@pytest.fixture(params=["shared", "override"])
async def configured_queue(request, monkeypatch):
    schema = "lat_queue_config_" + uuid4().hex
    connection = conninfo_to_dict(TEST_DSN)
    for name, value in {
        "POSTGRES_USER": connection.get("user", "postgres"),
        "POSTGRES_PASSWORD": SecretStr(connection.get("password") or "unused-test-password"),
        "POSTGRES_HOST": connection.get("host", "127.0.0.1"),
        "POSTGRES_PORT": int(connection.get("port", "5432")),
        "POSTGRES_DB": connection.get("dbname", "postgres"),
        "POSTGRES_APPLICATION_NAME": "lat-queue-config",
        "POSTGRES_STATEMENT_TIMEOUT": 1543,
        "POSTGRES_LOCK_TIMEOUT": 234,
        "POSTGRES_IDLE_IN_TRANSACTION_SESSION_TIMEOUT": 4321,
        "POSTGRES_POOL_TIMEOUT": 5,
        "BACKGROUND_POSTGRES_SCHEMA": schema,
        "BACKGROUND_POSTGRES_POOL_SIZE": 2,
        "BACKGROUND_POSTGRES_DSN": None,
    }.items():
        monkeypatch.setattr(settings, name, value)
    if request.param == "override":
        monkeypatch.setattr(settings, "POSTGRES_HOST", None)
        monkeypatch.setattr(settings, "POSTGRES_PASSWORD", None)
        monkeypatch.setattr(
            settings,
            "BACKGROUND_POSTGRES_DSN",
            SecretStr(
                make_conninfo(TEST_DSN, options="-c search_path=public -c statement_timeout=9000 -c timezone=UTC")
            ),
        )
    queue = queue_from_settings()
    try:
        await initialize_background(queue.pool.conninfo, schema=schema)
        async with queue:
            yield queue
    finally:
        async with await psycopg.AsyncConnection.connect(TEST_DSN, autocommit=True) as conn:
            await conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


async def test_queue_uses_shared_controls_and_replaces_closed_connections(configured_queue):
    queue = configured_queue
    assert isinstance(queue.pool, CheckedAsyncConnectionPool)
    assert queue.pool.max_size == 2
    async with queue.pool.connection() as conn:
        row = await (
            await conn.execute(
                "SELECT current_schema() AS schema, current_setting('application_name') AS application_name, "
                "current_setting('statement_timeout') AS statement_timeout, "
                "current_setting('lock_timeout') AS lock_timeout, "
                "current_setting('idle_in_transaction_session_timeout') AS idle_timeout"
            )
        ).fetchone()
        assert row == {
            "schema": queue.schema,
            "application_name": "lat-queue-config-background",
            "statement_timeout": "1543ms",
            "lock_timeout": "234ms",
            "idle_timeout": "4321ms",
        }
        await conn.close()
    job = await queue.submit("agent", "owner", ExecutionRequest(thread_id="replacement"), "key")
    assert (await queue.get(job.job_id, "owner")).status == "queued"


async def test_active_job_guard_is_scoped_and_releases_after_cancellation(configured_queue):
    queue = configured_queue
    job = await queue.submit("agent", "owner", ExecutionRequest(thread_id="thread"), "key")
    assert await queue.has_active_jobs("agent", "owner", "thread")
    assert not await queue.has_active_jobs("agent", "other-owner", "thread")
    assert not await queue.has_active_jobs("other-agent", "owner", "thread")
    assert not await queue.has_active_jobs("agent", "owner", "other-thread")
    await queue.cancel(job.job_id, "owner")
    assert not await queue.has_active_jobs("agent", "owner", "thread")


async def test_active_job_index_accepts_maximum_unicode_identities(configured_queue):
    generator = random.Random(41)
    owner, agent, thread = [
        "".join(chr(value) for value in generator.sample(range(0x10000, 0x10FFFF), 256)) for _ in range(3)
    ]
    queue = configured_queue
    job = await queue.submit(agent, owner, ExecutionRequest(thread_id=thread), "unicode-identity")
    assert await queue.has_active_jobs(agent, owner, thread)
    await queue.cancel(job.job_id, owner)
    assert not await queue.has_active_jobs(agent, owner, thread)


async def test_explicit_public_queue_schema_overrides_dsn_search_path():
    conninfo = make_conninfo(TEST_DSN, options="-c search_path=pg_catalog")
    options = postgres_pool_options(
        settings, min_size=1, max_size=2, schema="public", app_prefix="background", conninfo=conninfo, force_schema=True
    )
    async with CheckedAsyncConnectionPool(conninfo, **options) as pool:
        async with pool.connection() as conn:
            row = await (await conn.execute("SELECT current_setting('search_path') AS search_path")).fetchone()
        assert row == {"search_path": "public"}
