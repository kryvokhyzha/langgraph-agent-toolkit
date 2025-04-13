from langgraph_agent_toolkit.core.memory.base import BaseMemoryBackend
from langgraph_agent_toolkit.core.memory.postgres import PostgresMemoryBackend
from langgraph_agent_toolkit.core.memory.sqlite import SQLiteMemoryBackend
from langgraph_agent_toolkit.core.memory.types import MemoryBackends


class MemoryFactory:
    """Factory for creating memory backend instances."""

    @staticmethod
    def create(backend: MemoryBackends) -> BaseMemoryBackend:
        """Create and return a memory backend instance.

        Args:
            backend: The memory backend to create

        Returns:
            An instance of the requested memory backend

        Raises:
            ValueError: If the requested backend is not supported

        """
        match backend:
            case MemoryBackends.POSTGRES:
                return PostgresMemoryBackend()
            case MemoryBackends.SQLITE:
                return SQLiteMemoryBackend()
            case _:
                raise ValueError(f"Unsupported memory backend: {backend}")
