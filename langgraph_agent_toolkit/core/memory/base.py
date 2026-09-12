from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from typing import Any, TypeVar


T = TypeVar("T", bound=Any)


class BaseMemoryBackend(ABC):
    """Base class for memory backends."""

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate required configuration.

        Returns:
            `True` when configuration is valid.

        Raises:
            ValueError: If required configuration is missing.

        """
        pass

    @abstractmethod
    def get_checkpoint_saver(self) -> AbstractAsyncContextManager[T]:
        """Get the checkpoint saver for this memory backend.

        Returns:
            Configured checkpoint saver.

        """
        pass

    @abstractmethod
    def get_memory_store(self) -> AbstractAsyncContextManager[T]:
        """Get the memory store for this memory backend.

        Returns:
            Configured memory store.

        """
        pass
