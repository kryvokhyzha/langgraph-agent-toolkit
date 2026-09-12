"""Check lock acquisition, inherited tasks, and checkpoint iterator cleanup."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from langgraph_agent_toolkit.core.memory.concurrency import (
    ConversationBusyError,
    ConversationCoordinator,
    ConversationLockLostError,
    PostgresConversationCoordinator,
    PostgresLockSession,
    postgres_lock_session,
)
from langgraph_agent_toolkit.core.memory.coordinated_saver import CoordinatedPostgresSaver
from langgraph_agent_toolkit.core.settings import settings


pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "same_coordinator,same_key,child_task", [(True, True, False), (True, False, True), (False, False, True)]
)
async def test_nested_conversation_fails_before_waiting(same_coordinator, same_key, child_task):
    first = ConversationCoordinator(timeout=1)
    second = first if same_coordinator else ConversationCoordinator(timeout=1)

    async def enter():
        async with second.lock("outer" if same_key else "inner"):
            pytest.fail("A nested operation acquired another conversation lock")

    async with first.lock("outer"):
        with pytest.raises(ConversationBusyError, match="Nested"):
            async with asyncio.timeout(0.1):
                if child_task:
                    await asyncio.create_task(enter())
                else:
                    await enter()
    assert not first._entries
    assert not second._entries


async def test_child_can_start_an_independent_operation_after_parent_releases():
    coordinator = ConversationCoordinator(timeout=0.2)
    released = asyncio.Event()

    async def later():
        await released.wait()
        async with coordinator.lock("outer"):
            return "completed"

    async with coordinator.lock("outer"):
        child = asyncio.create_task(later())
    released.set()
    assert await asyncio.wait_for(child, 0.5) == "completed"
    assert not coordinator._entries


class LockConnection:
    def __init__(self):
        self.closed = False
        self.locked = False
        self.read_started = asyncio.Event()
        self.allow_read = asyncio.Event()
        self.unlock_started = asyncio.Event()
        self.allow_unlock = asyncio.Event()
        self.allow_unlock.set()

    async def execute(self, query, params=None):
        if "pg_try_advisory_lock" in query:
            self.locked = True
        elif "pg_advisory_unlock" in query:
            self.unlock_started.set()
            await self.allow_unlock.wait()
            self.locked = False
        return self

    async def fetchone(self):
        self.read_started.set()
        await self.allow_read.wait()
        return {"acquired": True}

    async def close(self):
        self.closed = True
        self.locked = False


class LockPool:
    def __init__(self, connection):
        self.conn = connection

    @asynccontextmanager
    async def connection(self, timeout=None):
        yield self.conn


async def test_cancel_between_server_acquisition_and_result_closes_session():
    connection = LockConnection()
    coordinator = PostgresConversationCoordinator(LockPool(connection), timeout=1)

    async def hold():
        async with coordinator.lock("conversation"):
            pytest.fail("The result was not read")

    task = asyncio.create_task(hold())
    await connection.read_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert connection.closed
    assert not connection.locked
    assert not coordinator._entries


async def test_repeated_cancellation_does_not_interrupt_unlock(monkeypatch):
    monkeypatch.setattr(settings, "THREAD_LOCK_HEARTBEAT_TIMEOUT", 0.2)
    connection = LockConnection()
    connection.allow_read.set()
    connection.allow_unlock.clear()
    entered = asyncio.Event()
    coordinator = PostgresConversationCoordinator(LockPool(connection), timeout=1)

    async def hold():
        async with coordinator.lock("conversation"):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold())
    await entered.wait()
    task.cancel()
    await connection.unlock_started.wait()
    task.cancel()
    connection.allow_unlock.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 0.5)
    assert not connection.locked
    assert not coordinator._entries


async def test_normal_completion_waits_for_active_heartbeat_without_cancelling_it(monkeypatch):
    monkeypatch.setattr(settings, "THREAD_LOCK_HEARTBEAT", 0.01)
    monkeypatch.setattr(settings, "THREAD_LOCK_HEARTBEAT_TIMEOUT", 0.5)
    heartbeat_started = asyncio.Event()
    release_heartbeat = asyncio.Event()
    body_finished = asyncio.Event()

    class HeartbeatConnection(LockConnection):
        async def execute(self, query, params=None):
            if query == "SELECT 1":
                heartbeat_started.set()
                try:
                    await release_heartbeat.wait()
                except asyncio.CancelledError:
                    await self.close()
                    raise
                return self
            return await super().execute(query, params)

    connection = HeartbeatConnection()
    connection.allow_read.set()
    coordinator = PostgresConversationCoordinator(LockPool(connection), timeout=1)

    async def finish():
        async with coordinator.lock("conversation"):
            await heartbeat_started.wait()
            body_finished.set()

    task = asyncio.create_task(finish())
    try:
        await asyncio.wait_for(body_finished.wait(), 1)
        await asyncio.sleep(0)
    finally:
        release_heartbeat.set()
        await asyncio.wait_for(task, 1)
    assert not connection.closed
    assert not connection.locked
    assert not coordinator._entries


async def test_finalization_from_another_task_still_releases_database_lock():
    connection = LockConnection()
    connection.allow_read.set()
    coordinator = PostgresConversationCoordinator(LockPool(connection), timeout=1)
    manager = coordinator.lock("conversation")
    await asyncio.create_task(manager.__aenter__())
    await manager.__aexit__(None, None, None)
    assert not connection.locked
    assert not coordinator._entries


async def test_checkpoint_list_releases_cursor_before_yield(monkeypatch):
    lock = asyncio.Lock()

    async def upstream_list(self, config, *, filter=None, before=None, limit=None):
        async with lock:
            yield "first"
            yield "second"

    monkeypatch.setattr(AsyncPostgresSaver, "alist", upstream_list)
    saver = CoordinatedPostgresSaver.__new__(CoordinatedPostgresSaver)
    iterator = saver.alist({"configurable": {"thread_id": "one"}})
    try:
        assert await anext(iterator) == "first"
        async with asyncio.timeout(0.1):
            async with lock:
                pass
        assert [value async for value in iterator] == ["second"]
    finally:
        await iterator.aclose()


async def test_heartbeat_does_not_treat_active_checkpoint_io_as_connection_loss(monkeypatch):
    monkeypatch.setattr(settings, "THREAD_LOCK_HEARTBEAT", 0.01)
    monkeypatch.setattr(settings, "THREAD_LOCK_HEARTBEAT_TIMEOUT", 0.02)
    connection = LockConnection()
    connection.allow_read.set()
    coordinator = PostgresConversationCoordinator(LockPool(connection), timeout=1)
    async with coordinator.lock("conversation"):
        session = postgres_lock_session.get()
        async with session.lock:
            await asyncio.sleep(0.06)
    assert not connection.locked


async def test_stale_inherited_checkpoint_session_cannot_use_the_pool():
    connection = LockConnection()
    session = PostgresLockSession(connection, scope="one")
    saver = CoordinatedPostgresSaver.__new__(CoordinatedPostgresSaver)
    saver._scope = "one"
    saver.conn = AsyncMock()
    token = postgres_lock_session.set(session)
    try:
        session.active = False
        with pytest.raises(ConversationLockLostError):
            async with saver._cursor():
                pass
        saver.conn.connection.assert_not_called()
    finally:
        postgres_lock_session.reset(token)
