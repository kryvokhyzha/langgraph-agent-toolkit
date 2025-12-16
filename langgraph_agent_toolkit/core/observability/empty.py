from typing import Any, Dict, Literal, Optional

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform
from langgraph_agent_toolkit.core.observability.types import PromptReturnType, PromptTemplateType
from langgraph_agent_toolkit.helper.logging import logger


class EmptyObservability(BaseObservabilityPlatform):
    """Empty implementation of observability platform with in-memory prompt storage.

    This implementation stores prompts in memory, allowing the prompt template
    system to work without a remote observability backend like Langfuse or LangSmith.
    """

    def __init__(self, remote_first: bool = False):
        """Initialize EmptyObservability.

        Args:
            remote_first: Ignored in empty implementation.

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
        """Record feedback - silently ignored without a remote backend."""
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
            name: Name of the prompt
            prompt_template: The prompt template to store
            metadata: Optional metadata for the prompt
            force_create_new_version: If True, overwrite existing prompt

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
            name: Name of the prompt to retrieve
            template_format: Format for the template (used for processing)
            **kwargs: Additional arguments (ignored)

        Returns:
            The stored prompt template, processed into a ChatPromptTemplate

        Raises:
            ValueError: If prompt not found in memory

        """
        if name not in self._prompts:
            raise ValueError(
                f"Prompt '{name}' not found. Use push_prompt() first or configure "
                "a remote observability backend (langfuse/langsmith)."
            )

        prompt_template = self._prompts[name]

        # Process the prompt using the base class helper
        return self._process_prompt_object(prompt_template, template_format=template_format)

    def delete_prompt(self, name: str) -> None:
        """Delete a prompt from memory.

        Args:
            name: Name of the prompt to delete

        """
        if name in self._prompts:
            del self._prompts[name]
            self._metadata.pop(name, None)
            logger.debug(f"Deleted prompt '{name}' from memory")
