from typing import Optional, Union

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.core.observability.langfuse import LangfuseObservability
from langgraph_agent_toolkit.core.observability.langsmith import LangsmithObservability
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend


class ObservabilityFactory:
    """Factory for creating observability platform instances."""

    @staticmethod
    def create(
        platform: Union[ObservabilityBackend, str], prompts_dir: Optional[str] = None
    ) -> BaseObservabilityPlatform:
        """Create and return an observability platform instance.

        Args:
            platform: The observability platform to create
            prompts_dir: Optional directory to store prompts locally

        Returns:
            An instance of the requested observability platform

        Raises:
            ValueError: If the requested platform is not supported

        """
        platform = ObservabilityBackend(platform)

        match platform:
            case ObservabilityBackend.LANGFUSE:
                return LangfuseObservability(prompts_dir=prompts_dir)

            case ObservabilityBackend.LANGSMITH:
                return LangsmithObservability(prompts_dir=prompts_dir)

            case ObservabilityBackend.EMPTY:
                return EmptyObservability(prompts_dir=prompts_dir)

            case _:
                raise ValueError(f"Unsupported ObservabilityBackend: {platform}")
