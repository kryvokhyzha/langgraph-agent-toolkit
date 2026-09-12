"""Serialize long-term store setup across PostgreSQL workers."""

from langgraph.store.postgres.aio import AsyncPostgresStore

from langgraph_agent_toolkit.core.memory.schema_lock import schema_setup_lock


class CoordinatedPostgresStore(AsyncPostgresStore):
    """Keep concurrent workers from applying the same schema migration."""

    async def setup(self) -> None:
        async with self._cursor() as cursor:
            async with schema_setup_lock(cursor, "langgraph-agent-toolkit:store-setup"):
                scoped = AsyncPostgresStore(
                    cursor.connection,
                    pipe=self.pipe,
                    deserializer=self._deserializer,
                    ttl=self.ttl_config,
                )
                # Reuse the validated index and embedding objects without creating providers.
                scoped.index_config = self.index_config
                scoped.embeddings = self.embeddings
                await scoped.setup()
