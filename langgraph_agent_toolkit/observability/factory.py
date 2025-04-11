from typing import Optional

from langgraph_agent_toolkit.observability.base import BaseObservabilityPlatform
from langgraph_agent_toolkit.observability.langfuse import LangfuseObservability
from langgraph_agent_toolkit.observability.langsmith import LangsmithObservability
from langgraph_agent_toolkit.observability.empty import EmptyObservability
from langgraph_agent_toolkit.observability.types import ObservabilityBackend


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
