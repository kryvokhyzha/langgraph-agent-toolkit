from abc import ABC, abstractmethod
import os
from typing import Any, List


class BaseObservabilityPlatform(ABC):
    """Base class for observability platforms."""

    __default_required_vars = []

    def __init__(self):
        """Initialize the observability platform."""
        self._required_vars = self.__default_required_vars.copy()

    @property
    def required_vars(self) -> List[str]:
        """Return the name of the observability platform."""
        return self._required_vars

    @required_vars.setter
    def required_vars(self, value: List[str]) -> None:
        """Set the name of the observability platform."""
        self._required_vars = value

    def validate_environment(self) -> bool:
        """Validate that all necessary environment variables are set."""
        missing_vars = [var for var in self._required_vars if not os.environ.get(var)]

        if missing_vars:
            raise ValueError(f"Missing required environment variables for Langsmith: {', '.join(missing_vars)}")

        return True

    @abstractmethod
    def get_callback_handler(self, **kwargs) -> Any:
        """Get the callback handler for the observability platform."""
        pass

    @abstractmethod
    def before_shutdown(self) -> None:
        """Perform any necessary cleanup before shutdown."""
        pass
