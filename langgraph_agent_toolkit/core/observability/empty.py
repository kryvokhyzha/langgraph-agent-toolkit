from typing import Any, Dict, Literal, Optional

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform
from langgraph_agent_toolkit.core.observability.types import PromptReturnType, PromptTemplateType


class EmptyObservability(BaseObservabilityPlatform):
    """Empty implementation of observability platform."""

    __default_required_vars = []

    def __init__(self, prompts_dir: Optional[str] = None):
        """Initialize EmptyObservability.

        Args:
            prompts_dir: Optional directory to store prompts locally. If None, a system temp directory is used.

        """
        super().__init__(prompts_dir)

    def get_callback_handler(self, **kwargs) -> None:
        """Get the callback handler for the observability platform."""
        return None

    def before_shutdown(self) -> None:
        """Perform any necessary cleanup before shutdown."""
        pass

    def record_feedback(self, run_id: str, key: str, score: float, **kwargs) -> None:
        """Record feedback for a run with Empty observability platform."""
        raise ValueError("Cannot record feedback: No observability platform is configured.")

    def push_prompt(
        self,
        name: str,
        prompt_template: PromptTemplateType,
        metadata: Optional[Dict[str, Any]] = None,
        create_new_version: bool = True,
    ) -> None:
        """Push a prompt using local storage.

        Args:
            name: Name of the prompt
            prompt_template: String template, list of message dicts, or prompt object
            metadata: Additional metadata for the prompt
            create_new_version: If True, overwrite existing prompt with new version

        """
        super().push_prompt(name, prompt_template, metadata, create_new_version)

    def pull_prompt(
        self,
        name: str,
        template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
        **kwargs,
    ) -> PromptReturnType:
        """Pull a prompt from local storage."""
        return super().pull_prompt(name, template_format=template_format, **kwargs)

    def delete_prompt(self, name: str) -> None:
        """Delete a prompt from local storage.

        Args:
            name: Name of the prompt to delete

        """
        super().delete_prompt(name)
