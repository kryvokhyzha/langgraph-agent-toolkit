"""Run PostgreSQL reliability checks only with an explicit test DSN."""

import asyncio
import os
import sys
from contextlib import aclosing, asynccontextmanager, suppress
from uuid import uuid4

import psycopg
import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage
from langgraph.graph import END, MessagesState, StateGraph
from psycopg import sql
from psycopg_pool import PoolTimeout, TooManyRequests
from pydantic import BaseModel, SecretStr

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.memory.concurrency import (
    ConversationCoordinator,
    ConversationLockLostError,
    PostgresConversationCoordinator,
)
from langgraph_agent_toolkit.core.memory.postgres import PostgresMemoryBackend
from langgraph_agent_toolkit.core.memory.schema_lock import schema_setup_lock
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.core.settings import settings


TEST_DSN = os.environ.get("LAT_TEST_POSTGRES_DSN")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.asyncio,
    pytest.mark.skipif(not TEST_DSN, reason="Set LAT_TEST_POSTGRES_DSN to a disposable PostgreSQL database"),
]


class Input(BaseModel):
    message: str


def make_executor(graph, coordinator):
    executor = AgentExecutor.__new__(AgentExecutor)
    executor.agents = {"test": Agent("test", "Local test", graph, EmptyObservability())}
    executor.concurrency = coordinator
    return executor


def message_graph(saver, node):
    builder = StateGraph(MessagesState)
    builder.add_node("reply", node)
    builder.set_entry_point("reply")
    builder.add_edge("reply", END)
    return builder.compile(checkpointer=saver)


@pytest_asyncio.fixture
async def backend(monkeypatch):
    schema = "lat_test_" + uuid4().hex
    async with await psycopg.AsyncConnection.connect(TEST_DSN, autocommit=True) as conn:
        await conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    monkeypatch.setattr(PostgresMemoryBackend, "get_connection_string", staticmethod(lambda: TEST_DSN))
    for name, value in {
        "POSTGRES_USER": "test-only",
        "POSTGRES_PASSWORD": SecretStr("unused-local-test-password"),
        "POSTGRES_HOST": "test-only",
        "POSTGRES_PORT": 5432,
        "POSTGRES_DB": "test-only",
        "POSTGRES_APPLICATION_NAME": "lat-test-" + uuid4().hex,
        "POSTGRES_SCHEMA": schema,
        "POSTGRES_MIN_SIZE": 1,
        "POSTGRES_POOL_SIZE": 3,
        "POSTGRES_LOCK_POOL_SIZE": 3,
        "POSTGRES_POOL_TIMEOUT": 3,
        "POSTGRES_RECONNECT_TIMEOUT": 3,
        "THREAD_LOCK_HEARTBEAT": 0.03,
        "THREAD_LOCK_HEARTBEAT_TIMEOUT": 0.5,
    }.items():
        monkeypatch.setattr(settings, name, value)
    try:
        yield PostgresMemoryBackend()
    finally:
        async with await psycopg.AsyncConnection.connect(TEST_DSN, autocommit=True) as conn:
            await conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


async def test_workers_serialize_same_thread_and_keep_both_turns(backend):
    async def reply(state):
        await asyncio.sleep(0.05)
        return {"messages": [AIMessage("reply to " + state["messages"][-1].content)]}

    async with backend.get_checkpoint_saver() as saver, backend.get_lock_pool() as lock_pool:
        await saver.setup()
        first = make_executor(message_graph(saver, reply), PostgresConversationCoordinator(lock_pool, timeout=2))
        second = make_executor(message_graph(saver, reply), PostgresConversationCoordinator(lock_pool, timeout=2))
        thread = uuid4().hex
        await asyncio.gather(
            first.invoke("test", Input(message="one"), thread_id=thread),
            second.invoke("test", Input(message="two"), thread_id=thread),
        )
        state = await first.agents["test"].graph.aget_state({"configurable": {"thread_id": thread}})
        assert {m.content for m in state.values["messages"]} == {"one", "two", "reply to one", "reply to two"}


async def test_workers_can_initialize_the_same_checkpoint_schema(backend):
    async with backend.get_checkpoint_saver() as first, backend.get_checkpoint_saver() as second:
        async with asyncio.TaskGroup() as startup:
            startup.create_task(first.setup())
            startup.create_task(second.setup())
        config = {"configurable": {"thread_id": uuid4().hex}}
        assert await first.aget_tuple(config) is None
        assert await second.aget_tuple(config) is None


async def test_workers_can_initialize_the_same_store_schema(backend):
    async with backend.get_memory_store() as first, backend.get_memory_store() as second:
        async with asyncio.TaskGroup() as startup:
            startup.create_task(first.setup())
            startup.create_task(second.setup())
        await first.aput(("tenant", "user"), "preference", {"language": "English"})
        item = await second.aget(("tenant", "user"), "preference")
        assert item.value == {"language": "English"}


async def test_schema_lock_wait_expires_and_later_setup_succeeds(backend, monkeypatch):
    async with backend.get_checkpoint_saver() as first, backend.get_checkpoint_saver() as second:
        monkeypatch.setattr(settings, "POSTGRES_POOL_TIMEOUT", 0.1)
        async with first._cursor() as cursor:
            async with schema_setup_lock(cursor, "langgraph-agent-toolkit:checkpoint-setup"):
                async with asyncio.timeout(1):
                    with pytest.raises(TimeoutError):
                        await second.setup()
        monkeypatch.setattr(settings, "POSTGRES_POOL_TIMEOUT", 3)
        await second.setup()
        assert await second.aget_tuple({"configurable": {"thread_id": uuid4().hex}}) is None


async def test_different_threads_execute_concurrently(backend):
    both_entered = asyncio.Event()
    entered = 0

    async def reply(state):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=2)
        return {"messages": [AIMessage("done")]}

    async with backend.get_checkpoint_saver() as saver, backend.get_lock_pool() as lock_pool:
        await saver.setup()
        executor = make_executor(message_graph(saver, reply), PostgresConversationCoordinator(lock_pool, timeout=2))
        await asyncio.gather(
            executor.invoke("test", Input(message="one"), thread_id=uuid4().hex),
            executor.invoke("test", Input(message="two"), thread_id=uuid4().hex),
        )
    assert entered == 2


async def test_cancellation_releases_advisory_lock(backend):
    entered = asyncio.Event()
    key = uuid4().hex
    async with backend.get_lock_pool() as pool:

        async def hold():
            async with PostgresConversationCoordinator(pool, timeout=2).lock(key):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(hold())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with PostgresConversationCoordinator(pool, timeout=1).lock(key):
            pass


async def test_session_loss_cancels_active_holder_and_releases_lock(backend):
    entered = asyncio.Event()
    stopped = asyncio.Event()
    key = uuid4().hex
    async with backend.get_lock_pool() as pool:
        coordinator = PostgresConversationCoordinator(pool, timeout=2)

        async def hold():
            async with coordinator.lock(key):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

        task = asyncio.create_task(hold())
        try:
            await entered.wait()
            async with await psycopg.AsyncConnection.connect(TEST_DSN, autocommit=True) as conn:
                cursor = await conn.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_locks "
                    "WHERE locktype = 'advisory' AND granted AND pid <> pg_backend_pid() "
                    "AND pid IN (SELECT pid FROM pg_stat_activity WHERE application_name = %s)",
                    (f"{settings.POSTGRES_APPLICATION_NAME}-locks",),
                )
                assert await cursor.fetchall(), "No lock-holding database session was found"
            with pytest.raises(ConversationLockLostError):
                await asyncio.wait_for(task, timeout=2)
            assert stopped.is_set()
            async with PostgresConversationCoordinator(pool, timeout=2).lock(key):
                pass
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


async def test_checkpoints_survive_graph_and_pool_restart(backend):
    async def reply(state):
        return {"messages": [AIMessage("reply to " + state["messages"][-1].content)]}

    thread = uuid4().hex
    async with backend.get_checkpoint_saver() as saver, backend.get_lock_pool() as pool:
        await saver.setup()
        executor = make_executor(message_graph(saver, reply), PostgresConversationCoordinator(pool, timeout=2))
        await executor.invoke("test", Input(message="before restart"), thread_id=thread)
    async with backend.get_checkpoint_saver() as saver, backend.get_lock_pool() as pool:
        await saver.setup()
        executor = make_executor(message_graph(saver, reply), PostgresConversationCoordinator(pool, timeout=2))
        await executor.invoke("test", Input(message="after restart"), thread_id=thread)
        state = await executor.agents["test"].graph.aget_state({"configurable": {"thread_id": thread}})
        assert [m.content for m in state.values["messages"]] == [
            "before restart",
            "reply to before restart",
            "after restart",
            "reply to after restart",
        ]


async def test_pool_replaces_terminated_idle_connection(backend):
    async with backend.get_lock_pool() as pool:
        async with pool.connection() as conn:
            first_pid = conn.info.backend_pid
        async with await psycopg.AsyncConnection.connect(TEST_DSN, autocommit=True) as admin:
            await admin.execute("SELECT pg_terminate_backend(%s)", (first_pid,))
        async with pool.connection(timeout=3) as conn:
            cursor = await conn.execute("SELECT 1 AS healthy")
            assert (await cursor.fetchone())["healthy"] == 1
            assert conn.info.backend_pid != first_pid


async def test_independent_checkpoint_reads_use_available_pool_connections(backend):
    async with backend.get_checkpoint_saver() as saver:
        await saver.setup()
        async with saver._cursor() as cursor:
            await cursor.execute("SELECT 1")
            # One caller must not hold the saver-wide lock while a second waits.
            async with asyncio.timeout(1):
                assert await saver.aget_tuple({"configurable": {"thread_id": uuid4().hex}}) is None


async def test_checkpoint_pool_rejects_excess_waiters_and_recovers_after_cancellation(backend, monkeypatch):
    monkeypatch.setattr(settings, "POSTGRES_POOL_SIZE", 1)
    monkeypatch.setattr(settings, "POSTGRES_POOL_MAX_WAITING", 1)
    async with backend.get_checkpoint_saver() as saver:
        await saver.setup()
        config = {"configurable": {"thread_id": uuid4().hex}}
        async with saver._cursor():
            waiter = asyncio.create_task(saver.aget_tuple(config))
            try:
                async with asyncio.timeout(1):
                    while not saver.conn.get_stats().get("requests_waiting"):
                        await asyncio.sleep(0.01)
                with pytest.raises(TooManyRequests):
                    async with asyncio.timeout(1):
                        await saver.aget_tuple(config)
            finally:
                waiter.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiter
        async with asyncio.timeout(1):
            assert await saver.aget_tuple(config) is None


async def test_cancelled_checkpoint_query_returns_a_usable_connection(backend):
    async with backend.get_checkpoint_saver() as saver:
        await saver.setup()
        started = asyncio.Event()

        async def slow_read():
            async with saver._cursor() as cursor:
                started.set()
                await cursor.execute("SELECT pg_sleep(30)")

        task = asyncio.create_task(slow_read())
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        async with asyncio.timeout(1):
            assert await saver.aget_tuple({"configurable": {"thread_id": uuid4().hex}}) is None


async def test_stalled_health_probe_respects_checkout_deadline_and_recovers(backend, monkeypatch):
    monkeypatch.setattr(settings, "POSTGRES_POOL_TIMEOUT", 0.3)
    monkeypatch.setattr(settings, "POSTGRES_HEALTH_CHECK_TIMEOUT", 0.05)
    connections = []
    async with backend.get_lock_pool() as pool:
        original_check = pool.check_connection

        async def stalled_check(connection):
            connections.append(connection)
            await connection.execute("SELECT pg_sleep(30)")

        monkeypatch.setattr(pool, "check_connection", stalled_check)
        with pytest.raises(PoolTimeout):
            async with asyncio.timeout(1):
                async with pool.connection():
                    pytest.fail("A stalled probe returned a connection")
        assert connections
        assert all(connection.closed for connection in connections)
        monkeypatch.setattr(pool, "check_connection", original_check)
        async with pool.connection(timeout=1) as connection:
            assert (await (await connection.execute("SELECT 1 AS healthy")).fetchone())["healthy"] == 1


async def test_cancelled_health_probe_closes_connection_and_preserves_cancellation(backend, monkeypatch):
    started = asyncio.Event()
    connections = []
    async with backend.get_lock_pool() as pool:
        original_check = pool.check_connection

        async def stalled_check(connection):
            connections.append(connection)
            started.set()
            await connection.execute("SELECT pg_sleep(30)")

        monkeypatch.setattr(pool, "check_connection", stalled_check)

        async def acquire():
            async with pool.connection():
                pytest.fail("A cancelled probe returned a connection")

        task = asyncio.create_task(acquire())
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert len(connections) == 1
        assert connections[0].closed
        monkeypatch.setattr(pool, "check_connection", original_check)
        async with pool.connection(timeout=1) as connection:
            assert not connection.closed


async def test_process_termination_releases_advisory_lock(backend):
    key = uuid4().hex
    code = """
import asyncio, os, sys, dotenv
dotenv.find_dotenv = lambda *args, **kwargs: ''
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph_agent_toolkit.core.memory.concurrency import PostgresConversationCoordinator
async def main():
    async with AsyncConnectionPool(os.environ['LAT_TEST_POSTGRES_DSN'], min_size=1, max_size=1,
            kwargs={'autocommit': True, 'row_factory': dict_row}, open=False) as pool:
        await pool.open(wait=True)
        async with PostgresConversationCoordinator(pool, timeout=2).lock(sys.argv[1]):
            print('READY', flush=True)
            await asyncio.Event().wait()
asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c", code, key, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        assert await asyncio.wait_for(process.stdout.readline(), timeout=10) == b"READY\n"
        process.kill()
        await asyncio.wait_for(process.wait(), timeout=5)
        async with backend.get_lock_pool() as pool:
            async with PostgresConversationCoordinator(pool, timeout=2).lock(key):
                pass
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()


async def test_lost_lock_holder_cannot_overwrite_newer_turn_before_heartbeat(backend, monkeypatch):
    monkeypatch.setattr(settings, "THREAD_LOCK_HEARTBEAT", 60)
    entered = asyncio.Event()
    release_old = asyncio.Event()
    thread = uuid4().hex

    async def reply(state):
        message = state["messages"][-1].content
        if message == "stale request":
            entered.set()
            await release_old.wait()
        return {"messages": [AIMessage("reply to " + message)]}

    async with backend.get_checkpoint_saver() as saver, backend.get_lock_pool() as pool:
        await saver.setup()
        first = make_executor(message_graph(saver, reply), PostgresConversationCoordinator(pool, timeout=2))
        second = make_executor(message_graph(saver, reply), PostgresConversationCoordinator(pool, timeout=2))
        old_task = asyncio.create_task(first.invoke("test", Input(message="stale request"), thread_id=thread))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            # Let the pre-node checkpoint finish before the simulated connection loss.
            await asyncio.sleep(0.05)
            async with await psycopg.AsyncConnection.connect(TEST_DSN, autocommit=True) as conn:
                cursor = await conn.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_locks "
                    "WHERE locktype = 'advisory' AND granted AND pid <> pg_backend_pid() "
                    "AND pid IN (SELECT pid FROM pg_stat_activity WHERE application_name = %s)",
                    (f"{settings.POSTGRES_APPLICATION_NAME}-locks",),
                )
                assert await cursor.fetchall()
            assert not old_task.done(), "The old node must remain active until after the competing commit"
            result = await second.invoke("test", Input(message="newer request"), thread_id=thread)
            assert result.content == "reply to newer request"
            config = {"configurable": {"thread_id": thread}}
            before = await second.agents["test"].graph.aget_state(config)
            release_old.set()
            with pytest.raises((ConversationLockLostError, psycopg.OperationalError)):
                await asyncio.wait_for(old_task, timeout=2)
            after = await second.agents["test"].graph.aget_state(config)
            assert after.values == before.values
            assert after.values["messages"][-1].content == "reply to newer request"
            assert "reply to stale request" not in [message.content for message in after.values["messages"]]
        finally:
            release_old.set()
            if not old_task.done():
                old_task.cancel()
                with suppress(asyncio.CancelledError, ConversationLockLostError, psycopg.OperationalError):
                    await old_task


async def test_independent_saver_keeps_its_schema_inside_another_backend_lock(backend, monkeypatch):
    other_schema = "lat_other_" + uuid4().hex
    async with await psycopg.AsyncConnection.connect(TEST_DSN, autocommit=True) as connection:
        await connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(other_schema)))

    async def reply(state):
        return {"messages": [AIMessage("second database reply")]}

    try:
        async with backend.get_checkpoint_saver() as first_saver, backend.get_lock_pool() as first_pool:
            await first_saver.setup()
            monkeypatch.setattr(settings, "POSTGRES_SCHEMA", other_schema)
            async with backend.get_checkpoint_saver() as other_saver:
                await other_saver.setup()
                graph = message_graph(other_saver, reply)
                executor = make_executor(graph, ConversationCoordinator())
                thread = uuid4().hex
                await executor.invoke("test", Input(message="second database request"), thread_id=thread)
                config = {"configurable": {"thread_id": thread}}
                outside = await graph.aget_state(config)
                async with PostgresConversationCoordinator(first_pool, timeout=2).lock(uuid4().hex):
                    inside = await graph.aget_state(config)
                assert inside.values == outside.values
    finally:
        async with await psycopg.AsyncConnection.connect(TEST_DSN, autocommit=True) as connection:
            await connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(other_schema)))


@pytest.mark.parametrize("locked", [False, True])
async def test_checkpoint_iteration_allows_a_second_query(backend, locked):
    async def reply(state):
        return {"messages": [AIMessage("done")]}

    @asynccontextmanager
    async def no_lock():
        yield

    async with backend.get_checkpoint_saver() as saver, backend.get_lock_pool() as pool:
        await saver.setup()
        thread = uuid4().hex
        executor = make_executor(message_graph(saver, reply), PostgresConversationCoordinator(pool, timeout=1))
        await executor.invoke("test", Input(message="hello"), thread_id=thread)
        config = {"configurable": {"thread_id": thread}}
        async with executor.concurrency.lock(thread) if locked else no_lock():
            async with aclosing(saver.alist(config, limit=2)) as checkpoints:
                async for checkpoint in checkpoints:
                    async with asyncio.timeout(1):
                        assert await saver.aget_tuple(checkpoint.config) is not None


async def test_normal_langgraph_subgraph_reuses_the_parent_lock(backend):
    async def reply(state):
        return {"messages": [AIMessage("subgraph reply")]}

    async with backend.get_checkpoint_saver() as saver, backend.get_lock_pool() as pool:
        await saver.setup()
        child = message_graph(None, reply)
        graph = StateGraph(MessagesState)
        graph.add_node("child", child)
        graph.set_entry_point("child")
        graph.add_edge("child", END)
        thread = uuid4().hex
        executor = make_executor(graph.compile(checkpointer=saver), PostgresConversationCoordinator(pool, timeout=1))
        result = await executor.invoke("test", Input(message="hello"), thread_id=thread)
        assert result.content == "subgraph reply"
        state = await executor.agents["test"].graph.aget_state({"configurable": {"thread_id": thread}})
        assert [value.content for value in state.values["messages"]] == ["hello", "subgraph reply"]


async def test_cancelled_lock_result_cannot_leave_a_session_lock_in_the_pool(backend):
    acquired = asyncio.Event()

    class DelayedConnection:
        def __init__(self, conn):
            self.conn = conn

        async def execute(self, query, params):
            cursor = await self.conn.execute(query, params)

            class DelayedCursor:
                async def fetchone(self):
                    result = await cursor.fetchone()
                    assert result["acquired"] is True
                    acquired.set()
                    await asyncio.Event().wait()
                    return result

            return DelayedCursor()

        async def close(self):
            await self.conn.close()

    async with backend.get_lock_pool() as pool:
        connections = []

        class DelayedPool:
            @asynccontextmanager
            async def connection(self, timeout):
                async with pool.connection(timeout=timeout) as conn:
                    connections.append(conn)
                    yield DelayedConnection(conn)

        thread = uuid4().hex

        async def hold():
            async with PostgresConversationCoordinator(DelayedPool(), timeout=1).lock(thread):
                pytest.fail("The lock result was not delivered")

        task = asyncio.create_task(hold())
        try:
            await asyncio.wait_for(acquired.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert connections[0].closed
            async with PostgresConversationCoordinator(pool, timeout=1).lock(thread):
                pass
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
