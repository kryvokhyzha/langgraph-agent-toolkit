from typing import Any, Dict, Literal, Optional

from langgraph_agent_toolkit.core.observability.base import (
    BaseObservabilityPlatform,
    PromptReturnType,
    PromptTemplateType,
)
from langgraph_agent_toolkit.helper.logging import logger


class EmptyObservability(BaseObservabilityPlatform):
    """Observability platform with in-memory prompt storage.

    This supports prompt templates without a remote Langfuse or LangSmith backend.
    """

    def __init__(self, remote_first: bool = False):
        """Initialize EmptyObservability.

        Args:
            remote_first: Ignored by this implementation.

        """
        super().__init__(remote_first)
        self._prompts: Dict[str, PromptTemplateType] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def get_callback_handler(self, **kwargs) -> None:
        """Get the callback handler for the observability platform."""
        return None

    def before_shutdown(self) -> None:
        """Perform any necessary cleanup before shutdown."""
        pass

    def record_feedback(self, run_id: str, key: str, score: float, **kwargs) -> None:
        """Record feedback by ignoring it without a remote backend."""
        logger.debug(f"Feedback ignored (no observability backend): run_id={run_id}, key={key}, score={score}")

    def push_prompt(
        self,
        name: str,
        prompt_template: PromptTemplateType,
        metadata: Optional[Dict[str, Any]] = None,
        force_create_new_version: bool = True,
    ) -> None:
        """Store a prompt in memory.

        Args:
            name: Prompt name.
            prompt_template: Prompt template to store.
            metadata: Optional prompt metadata.
            force_create_new_version: Overwrite an existing prompt when `True`.

        """
        if name in self._prompts and not force_create_new_version:
            logger.debug(f"Prompt '{name}' already exists, skipping (force_create_new_version=False)")
            return

        self._prompts[name] = prompt_template
        if metadata:
            self._metadata[name] = metadata
        logger.debug(f"Stored prompt '{name}' in memory")

    def pull_prompt(
        self,
        name: str,
        template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
        **kwargs,
    ) -> PromptReturnType:
        """Retrieve a prompt from memory.

        Args:
            name: Prompt name.
            template_format: Template format for processing.
            **kwargs: Ignored arguments.

        Returns:
            Stored prompt template as a `ChatPromptTemplate`.

        Raises:
            ValueError: If the prompt is missing from memory.

        """
        if name not in self._prompts:
            raise ValueError(
                f"Prompt '{name}' not found. Use push_prompt() first or configure "
                "a remote observability backend (langfuse/langsmith)."
            )

        prompt_template = self._prompts[name]

        # Process the prompt with the base-class helper.
        return self._process_prompt_object(prompt_template, template_format=template_format)

    def delete_prompt(self, name: str) -> None:
        """Delete a prompt from memory.

        Args:
            name: Prompt name to delete.

        """
        if name in self._prompts:
            del self._prompts[name]
            self._metadata.pop(name, None)
            logger.debug(f"Deleted prompt '{name}' from memory")
