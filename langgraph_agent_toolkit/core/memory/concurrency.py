"""Bound concurrent conversation operations and release locks after failures."""

import asyncio
import functools
import hashlib
import inspect
import sys
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from langgraph_agent_toolkit.core.execution import execution_timeout
from langgraph_agent_toolkit.core.settings import settings


class ConversationBusyError(Exception):
    """The conversation queue is full or its wait time has expired."""


class NestedConversationError(ConversationBusyError):
    """An active operation tried to acquire another conversation lock."""


class ConversationLockLostError(Exception):
    """The database session that owns the conversation lock was lost."""


@dataclass
class _Entry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


@dataclass
class _ConversationLease:
    active: bool = True


_conversation_lease: ContextVar[_ConversationLease | None] = ContextVar("conversation_lease", default=None)


@dataclass
class PostgresLockSession:
    """Bind checkpoint I/O to the session that owns its advisory lock."""

    connection: object
    scope: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active: bool = True


postgres_lock_session: ContextVar[PostgresLockSession | None] = ContextVar("postgres_lock_session", default=None)


class ConversationCoordinator:
    """Serialize operations on one thread in this process."""

    def __init__(self, timeout: float | None = None, max_waiters: int | None = None):
        self.timeout = timeout if timeout is not None else settings.THREAD_QUEUE_TIMEOUT
        self.max_waiters = max_waiters if max_waiters is not None else settings.THREAD_QUEUE_MAX_WAITERS
        self._entries: dict[str, _Entry] = {}

    @asynccontextmanager
    async def _external_lock(self, key: str, deadline: float):
        yield

    @asynccontextmanager
    async def lock(self, key: str):
        parent = _conversation_lease.get()
        if parent is not None and parent.active:
            raise NestedConversationError(
                "Nested conversation operations are not supported. Compose agents with LangGraph subgraphs."
            )
        entry = self._entries.setdefault(key, _Entry())
        if entry.users >= self.max_waiters + 1:
            raise ConversationBusyError("The conversation queue is full. Retry later.")
        entry.users += 1
        acquired = False
        deadline = asyncio.get_running_loop().time() + self.timeout
        try:
            try:
                await asyncio.wait_for(entry.lock.acquire(), timeout=self.timeout)
                acquired = True
            except TimeoutError as exc:
                raise ConversationBusyError("The conversation queue wait expired. Retry later.") from exc
            lease = _ConversationLease()
            token = _conversation_lease.set(lease)
            try:
                async with self._external_lock(key, deadline):
                    yield
            finally:
                # Child tasks inherit this object. They must see when the run ends.
                lease.active = False
                # Async generator finalization can run in a different context.
                with suppress(ValueError):
                    _conversation_lease.reset(token)
        finally:
            if acquired:
                entry.lock.release()
            entry.users -= 1
            if not entry.users:
                self._entries.pop(key, None)


class SQLiteConversationCoordinator(ConversationCoordinator):
    """Use OS file locks to coordinate workers that share one SQLite file."""

    def __init__(self, database_path: str, **kwargs):
        super().__init__(**kwargs)
        self.directory = Path(database_path).resolve().with_suffix(".conversation-locks")
        self.directory.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def _external_lock(self, key: str, deadline: float):
        from filelock import FileLock, Timeout

        # Fixed stripes bound the number of files. Collisions only add waiting.
        stripe = int.from_bytes(hashlib.sha256(key.encode()).digest()[:2], "big") % 256
        lock = FileLock(str(self.directory / f"{stripe}.lock"), thread_local=False)
        acquired = False
        try:
            try:
                async with asyncio.timeout_at(deadline):
                    while not acquired:
                        try:
                            lock.acquire(timeout=0)
                            acquired = True
                        except Timeout:
                            await asyncio.sleep(0.05)
            except TimeoutError as exc:
                raise ConversationBusyError("The conversation queue wait expired. Retry later.") from exc
            yield
        finally:
            if acquired:
                lock.release()


class PostgresConversationCoordinator(ConversationCoordinator):
    """Hold a session advisory lock in a separate, bounded connection pool."""

    def __init__(self, pool, **kwargs):
        super().__init__(**kwargs)
        self.pool = pool

    @asynccontextmanager
    async def _external_lock(self, key: str, deadline: float):
        lock_id = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big", signed=True)
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise ConversationBusyError("The conversation queue wait expired. Retry later.")
        async with self.pool.connection(timeout=remaining) as conn:
            acquired = False
            acquisition_pending = False
            heartbeat = None
            stop_heartbeat = asyncio.Event()
            lost = None
            owner = asyncio.current_task()
            session = PostgresLockSession(conn, scope=getattr(self.pool, "_lat_checkpoint_scope", None))
            token = None

            async def check_session():
                nonlocal lost
                try:
                    while True:
                        try:
                            await asyncio.wait_for(stop_heartbeat.wait(), settings.THREAD_LOCK_HEARTBEAT)
                            return
                        except TimeoutError:
                            pass
                        # Checkpoint I/O has its own query and request time limits.
                        async with session.lock:
                            if stop_heartbeat.is_set():
                                return
                            async with asyncio.timeout(settings.THREAD_LOCK_HEARTBEAT_TIMEOUT):
                                await conn.execute("SELECT 1")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    lost = exc
                    if not stop_heartbeat.is_set():
                        owner.cancel()

            try:
                try:
                    async with asyncio.timeout_at(deadline):
                        while not acquired:
                            acquisition_pending = True
                            cursor = await conn.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (lock_id,))
                            row = await cursor.fetchone()
                            acquired = row["acquired"]
                            acquisition_pending = False
                            if not acquired:
                                await asyncio.sleep(0.05)
                except TimeoutError as exc:
                    raise ConversationBusyError("The conversation queue wait expired. Retry later.") from exc
                token = postgres_lock_session.set(session)
                heartbeat = asyncio.create_task(check_session())
                try:
                    yield
                except asyncio.CancelledError:
                    if lost is not None:
                        raise ConversationLockLostError("The database conversation lock was lost.") from lost
                    raise
            finally:
                active_error = sys.exception()
                session.active = False
                if token is not None:
                    with suppress(ValueError):
                        postgres_lock_session.reset(token)
                if heartbeat is not None:
                    # Do not cancel a healthy query during normal run cleanup.
                    stop_heartbeat.set()
                    try:
                        async with asyncio.timeout(settings.THREAD_LOCK_HEARTBEAT_TIMEOUT):
                            await heartbeat
                    except TimeoutError as exc:
                        lost = exc
                        await conn.close()
                    except asyncio.CancelledError:
                        # The owner was cancelled again during cleanup.
                        await conn.close()
                        raise
                if acquired:
                    try:
                        async with asyncio.timeout(settings.THREAD_LOCK_HEARTBEAT_TIMEOUT):
                            async with session.lock:
                                await conn.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
                    except BaseException as exc:
                        # Closing the session releases locks even after cancellation.
                        with suppress(Exception):
                            await conn.close()
                        if active_error is None:
                            if isinstance(exc, asyncio.CancelledError):
                                raise
                            raise ConversationLockLostError("The database conversation lock was lost.") from exc
                elif acquisition_pending:
                    # The server can acquire the lock before cancellation reaches the client.
                    # Do not return this session to the pool with an unknown lock state.
                    await conn.close()
                if lost is not None and active_error is None:
                    raise ConversationLockLostError("The database conversation lock was lost.") from lost


def serialize_execution(function):
    """Hold a conversation lock through execution and stream cleanup."""
    signature = inspect.signature(function)

    def prepare(self, args, kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        bound.arguments["thread_id"] = bound.arguments["thread_id"] or str(uuid4())
        return bound

    if inspect.isasyncgenfunction(function):

        @functools.wraps(function)
        async def stream(self, *args, **kwargs):
            bound = prepare(self, args, kwargs)
            async with self.concurrency.lock(bound.arguments["thread_id"]):
                async with execution_timeout(settings.REQUEST_TIMEOUT):
                    generator = function(*bound.args, **bound.kwargs)
                    try:
                        async for item in generator:
                            yield item
                    finally:
                        await generator.aclose()

        return stream

    @functools.wraps(function)
    async def invoke(self, *args, **kwargs):
        bound = prepare(self, args, kwargs)
        async with self.concurrency.lock(bound.arguments["thread_id"]):
            async with execution_timeout(settings.REQUEST_TIMEOUT):
                return await function(*bound.args, **bound.kwargs)

    return invoke
