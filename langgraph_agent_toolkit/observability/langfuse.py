from langfuse.callback import CallbackHandler
from langfuse import Langfuse

from langgraph_agent_toolkit.observability.base import BaseObservabilityPlatform


class LangfuseObservability(BaseObservabilityPlatform):
    """Langfuse implementation of observability platform."""

    __default_required_vars = ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_HOST"]

    def get_callback_handler(self, **kwargs) -> CallbackHandler:
        """
        Get Langfuse callback handler.

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
