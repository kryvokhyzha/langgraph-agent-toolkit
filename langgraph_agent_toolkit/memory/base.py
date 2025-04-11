from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from typing import Union


class BaseMemoryBackend(ABC):
    """Base class for memory backends."""

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate that all necessary configuration is set.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If required configuration is missing
        """
        pass

    @abstractmethod
    def get_checkpoint_saver(self) -> AbstractAsyncContextManager[Union["AsyncSqliteSaver", "AsyncPostgresSaver"]]:
        """
        Get the checkpoint saver for the memory backend.

        Returns:
            A configured checkpoint saver
        """
        pass
