"""Keep checkpoint operations on the connection that owns the conversation lock."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from langgraph_agent_toolkit.core.memory.concurrency import ConversationLockLostError, postgres_lock_session
from langgraph_agent_toolkit.core.memory.schema_lock import schema_setup_lock


class CoordinatedPostgresSaver(AsyncPostgresSaver):
    """Use the pool outside a run, and the locked session during a run.

    Override the upstream cursor hook so reads, writes, and deletion share the
    same session. A disconnected session cannot write through a replacement
    connection after another worker acquires the conversation lock.
    """

    def __init__(self, conn, pipe=None, serde=None):
        super().__init__(conn, pipe=pipe, serde=serde)
        self._scope = getattr(conn, "_lat_checkpoint_scope", None)

    async def setup(self) -> None:
        """Serialize schema changes when several workers start together."""
        async with self._cursor() as cursor:
            async with schema_setup_lock(cursor, "langgraph-agent-toolkit:checkpoint-setup"):
                scoped = AsyncPostgresSaver(cursor.connection, pipe=self.pipe, serde=self.serde)
                await scoped.setup()

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Release the database cursor before the caller receives a checkpoint."""
        # Upstream fetches all selected rows, but holds its lock across each yield.
        # Collect its results so the caller can safely make another checkpoint query.
        values = [value async for value in super().alist(config, filter=filter, before=before, limit=limit)]
        for value in values:
            yield value

    @asynccontextmanager
    async def _cursor(self, *, pipeline=False):
        session = postgres_lock_session.get()
        if session is None or self._scope is None or self._scope != session.scope:
            if isinstance(self.conn, AsyncConnectionPool):
                # Each operation owns one pool connection. The pool bounds waiting.
                async with self.conn.connection() as connection:
                    scoped = AsyncPostgresSaver(connection, serde=self.serde)
                    async with scoped._cursor(pipeline=pipeline) as cursor:
                        yield cursor
                return
            async with super()._cursor(pipeline=pipeline) as cursor:
                yield cursor
            return
        if not session.active or session.connection.closed:
            raise ConversationLockLostError("The database conversation lock was lost.")
        scoped = AsyncPostgresSaver(session.connection, serde=self.serde)
        scoped.lock = session.lock
        async with scoped._cursor(pipeline=pipeline) as cursor:
            if not session.active:
                raise ConversationLockLostError("The database conversation lock was released.")
            yield cursor
