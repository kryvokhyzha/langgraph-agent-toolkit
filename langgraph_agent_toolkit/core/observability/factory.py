from typing import Optional

from core.observability.base import BaseObservabilityPlatform
from core.observability.langfuse import LangfuseObservability
from core.observability.langsmith import LangsmithObservability
from core.observability.empty import EmptyObservability
from core.observability.types import ObservabilityBackend


class ObservabilityFactory:
    """Factory for creating observability platform instances."""

    @staticmethod
    def create(platform: ObservabilityBackend) -> Optional[BaseObservabilityPlatform]:
        """
        Create and return an observability platform instance.

        Args:
            platform: The observability platform to create

        Returns:
            An instance of the requested observability platform

        Raises:
            ValueError: If the requested platform is not supported
        """
        match platform:
            case ObservabilityBackend.LANGFUSE:
                return LangfuseObservability()
            case ObservabilityBackend.LANGSMITH:
                return LangsmithObservability()
            case ObservabilityBackend.EMPTY | None:
                return EmptyObservability()
            case _:
                raise ValueError(f"Unsupported observability platform: {platform}")
