"""Release SQLite cursors and failed transactions before the next operation."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class _TransactionLock(asyncio.Lock):
    """Roll back a failed operation before another task can use the connection."""

    def __init__(self, connection):
        super().__init__()
        self.connection = connection

    async def __aexit__(self, exc_type, exc, traceback):
        try:
            if exc_type is not None:

                async def rollback():
                    try:
                        # The worker queue runs this after any pending SQLite call.
                        await self.connection.rollback()
                    except BaseException:
                        await self.connection.close()
                        raise

                cleanup = asyncio.create_task(rollback())
                cancelled = False
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        # Keep the lock until rollback finishes after repeated cancellation.
                        cancelled = True
                cleanup.result()
                if cancelled:
                    raise asyncio.CancelledError from exc
        finally:
            await super().__aexit__(exc_type, exc, traceback)


class CoordinatedSqliteSaver(AsyncSqliteSaver):
    """Keep failed writes out of the next operation's transaction."""

    def __init__(self, conn, *, serde=None):
        super().__init__(conn, serde=serde)
        self.lock = _TransactionLock(conn)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Release the database cursor before the caller receives a checkpoint."""
        values = [value async for value in super().alist(config, filter=filter, before=before, limit=limit)]
        for value in values:
            yield value
