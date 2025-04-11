from contextlib import AbstractAsyncContextManager
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langgraph_agent_toolkit.core.settings import settings
from core.memory.base import BaseMemoryBackend


class SQLiteMemoryBackend(BaseMemoryBackend):
    """SQLite implementation of memory backend."""

    def validate_config(self) -> bool:
        """
        Validate that SQLite configuration is present.
        Raises ValueError if configuration is missing.
        """
        if not getattr(settings, "SQLITE_DB_PATH", None):
            raise ValueError("Missing SQLITE_DB_PATH configuration. This must be set to use SQLite persistence.")
        return True

    def get_checkpoint_saver(self) -> AbstractAsyncContextManager[AsyncSqliteSaver]:
        """Initialize and return a SQLite saver instance."""
        self.validate_config()
        return AsyncSqliteSaver.from_conn_string(settings.SQLITE_DB_PATH)
