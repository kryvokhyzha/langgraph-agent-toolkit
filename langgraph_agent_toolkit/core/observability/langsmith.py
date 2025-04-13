from langsmith import Client as LangsmithClient

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform


class LangsmithObservability(BaseObservabilityPlatform):
    """Langsmith implementation of observability platform."""

    __default_required_vars = ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"]

    def get_callback_handler(self, **kwargs) -> None:
        """Get the callback handler for the observability platform."""
        self.validate_environment()
        return None

    def before_shutdown(self) -> None:
        """Perform any necessary cleanup before shutdown."""
        pass

    def record_feedback(self, run_id: str, key: str, score: float, **kwargs) -> None:
        """Record feedback for a run to LangSmith."""
        self.validate_environment()
        client = LangsmithClient()
        client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            **kwargs,
        )
