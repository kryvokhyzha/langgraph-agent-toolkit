"""Serialize PostgreSQL schema changes without holding a waiting query snapshot."""

import asyncio
from contextlib import asynccontextmanager

from langgraph_agent_toolkit.core.settings import settings


@asynccontextmanager
async def schema_setup_lock(cursor, name: str):
    """Hold a schema lock until setup finishes, or close an uncertain session."""
    try:
        # Two-key advisory locks use a separate space from conversation locks.
        # A blocking SELECT can block CREATE INDEX CONCURRENTLY with its snapshot.
        async with asyncio.timeout(settings.POSTGRES_POOL_TIMEOUT):
            while True:
                await cursor.execute(
                    "SELECT pg_try_advisory_lock(hashtext(%s), hashtext(current_schema())) AS acquired",
                    (name,),
                )
                if (await cursor.fetchone())["acquired"]:
                    break
                await asyncio.sleep(0.05)
        yield
        await cursor.execute(
            "SELECT pg_advisory_unlock(hashtext(%s), hashtext(current_schema()))",
            (name,),
        )
    except BaseException:
        # Acquisition or migration can finish before cancellation is received.
        await cursor.connection.close()
        raise
