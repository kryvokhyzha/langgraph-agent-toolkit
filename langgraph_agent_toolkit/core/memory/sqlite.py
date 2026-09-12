from contextlib import AbstractAsyncContextManager

from langgraph_agent_toolkit.core.memory.base import BaseMemoryBackend
from langgraph_agent_toolkit.core.memory.coordinated_sqlite import CoordinatedSqliteSaver as AsyncSqliteSaver
from langgraph_agent_toolkit.core.settings import settings


class SQLiteMemoryBackend(BaseMemoryBackend):
    """SQLite memory backend."""

    def validate_config(self) -> bool:
        """Validate SQLite configuration."""
        if not getattr(settings, "SQLITE_DB_PATH", None):
            raise ValueError("Missing SQLITE_DB_PATH configuration. This must be set to use SQLite persistence.")
        return True

    def get_checkpoint_saver(self) -> AbstractAsyncContextManager[AsyncSqliteSaver]:
        """Initialize and return a SQLite saver."""
        self.validate_config()
        return AsyncSqliteSaver.from_conn_string(settings.SQLITE_DB_PATH)

    def get_memory_store(self) -> AbstractAsyncContextManager[AsyncSqliteSaver]:
        """Initialize and return a SQLite store."""
        raise NotImplementedError(
            "`SQLiteMemoryBackend` does not support get_memory_store. Use `get_checkpoint_saver` instead."
        )
