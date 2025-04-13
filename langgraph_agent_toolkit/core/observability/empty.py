from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform


class EmptyObservability(BaseObservabilityPlatform):
    """Empty implementation of observability platform."""

    __default_required_vars = []

    def get_callback_handler(self, **kwargs) -> None:
        """
        Langsmith doesn't require a custom callback handler as it's configured via environment variables.
        This method validates the environment and returns None.

        Args:
            **kwargs: Any keyword arguments (not used for Langsmith)

        Returns:
            None as Langsmith uses environment variables for configuration
        """
        return None

    def before_shutdown(self) -> None:
        """Perform any necessary cleanup before shutdown."""
        pass

    def record_feedback(self, run_id: str, key: str, score: float, **kwargs) -> None:
        """Record feedback for a run with Empty observability platform."""
        raise ValueError("Cannot record feedback: No observability platform is configured.")
