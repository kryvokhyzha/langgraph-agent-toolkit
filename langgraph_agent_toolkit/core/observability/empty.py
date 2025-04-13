from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform


class EmptyObservability(BaseObservabilityPlatform):
    """Empty implementation of observability platform."""

    __default_required_vars = []

    def get_callback_handler(self, **kwargs) -> None:
        """Get the callback handler for the observability platform."""
        return None

    def before_shutdown(self) -> None:
        """Perform any necessary cleanup before shutdown."""
        pass

    def record_feedback(self, run_id: str, key: str, score: float, **kwargs) -> None:
        """Record feedback for a run with Empty observability platform."""
        raise ValueError("Cannot record feedback: No observability platform is configured.")
