from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform


class LangfuseObservability(BaseObservabilityPlatform):
    """Langfuse implementation of observability platform."""

    __default_required_vars = ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_HOST"]

    def get_callback_handler(self, **kwargs) -> CallbackHandler:
        """Get Langfuse callback handler.

        Args:
            **kwargs: Any keyword arguments

        Returns:
            A configured Langfuse CallbackHandler

        """
        self.validate_environment()

        return CallbackHandler(**kwargs)

    def before_shutdown(self) -> None:
        """Perform any necessary cleanup before shutdown."""
        Langfuse().flush()

    def record_feedback(self, run_id: str, key: str, score: float, **kwargs) -> None:
        """Record feedback for a run to Langfuse."""
        self.validate_environment()
        client = Langfuse()
        client.score(
            trace_id=run_id,
            name=key,
            value=score,
            **kwargs,
        )
