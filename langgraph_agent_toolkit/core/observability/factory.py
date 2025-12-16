from typing import Union

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend


class ObservabilityFactory:
    """Factory for creating observability platform instances."""

    @staticmethod
    def create(
        platform: Union[ObservabilityBackend, str],
        remote_first: bool = False,
        **kwargs,
    ) -> BaseObservabilityPlatform:
        """Create and return an observability platform instance.

        Args:
            platform: The observability platform to create
            remote_first: If True, prioritize remote prompts over local ones
            **kwargs: Additional arguments to pass to the platform constructor

        Returns:
            An instance of the requested observability platform

        Raises:
            ValueError: If the requested platform is not supported

        """
        platform = ObservabilityBackend(platform)

        match platform:
            case ObservabilityBackend.LANGFUSE:
                from langgraph_agent_toolkit.core.observability.langfuse import LangfuseObservability

                return LangfuseObservability(remote_first=remote_first, **kwargs)

            case ObservabilityBackend.LANGSMITH:
                from langgraph_agent_toolkit.core.observability.langsmith import LangsmithObservability

                return LangsmithObservability(remote_first=remote_first, **kwargs)

            case ObservabilityBackend.EMPTY:
                from langgraph_agent_toolkit.core.observability.empty import EmptyObservability

                return EmptyObservability(remote_first=remote_first, **kwargs)

            case _:
                raise ValueError(f"Unsupported ObservabilityBackend: {platform}")
