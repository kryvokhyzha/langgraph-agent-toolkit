from typing import Any, Dict, Literal, Optional

from langsmith import Client as LangsmithClient
from langsmith.utils import LangSmithConflictError

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform
from langgraph_agent_toolkit.core.observability.types import PromptReturnType, PromptTemplateType
from langgraph_agent_toolkit.helper.logging import logger


class LangsmithObservability(BaseObservabilityPlatform):
    """Langsmith implementation of observability platform."""

    def __init__(self, remote_first: bool = False):
        """Initialize LangsmithObservability.

        Args:
            remote_first: If True, prioritize remote prompts over local ones.

        """
        super().__init__(remote_first)
        self.required_vars = ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"]

    @BaseObservabilityPlatform.requires_env_vars
    def get_callback_handler(self, **kwargs) -> None:
        """Get the callback handler - LangSmith uses automatic tracing."""
        return None

    def before_shutdown(self) -> None:
        """Perform any necessary cleanup before shutdown."""
        pass

    @BaseObservabilityPlatform.requires_env_vars
    def record_feedback(self, run_id: str, key: str, score: float, **kwargs) -> None:
        """Record feedback for a run to LangSmith."""
        client = LangsmithClient()

        if "user_id" in kwargs:
            user_id = kwargs.pop("user_id")
            kwargs.setdefault("extra", {})
            kwargs["extra"]["user_id"] = user_id

        client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            **kwargs,
        )

    @BaseObservabilityPlatform.requires_env_vars
    def push_prompt(
        self,
        name: str,
        prompt_template: PromptTemplateType,
        metadata: Optional[Dict[str, Any]] = None,
        force_create_new_version: bool = True,
    ) -> None:
        """Push a prompt to LangSmith.

        Args:
            name: Name of the prompt
            prompt_template: The prompt template (string or list of message dicts)
            metadata: Optional metadata (can include 'model' to create a chain)
            force_create_new_version: If True, always create a new version

        """
        client = LangsmithClient()
        prompt_obj = self._convert_to_chat_prompt(prompt_template)

        # Check if prompt already exists when remote_first is enabled
        if self.remote_first:
            try:
                existing = client.pull_prompt(name)
                if existing:
                    logger.debug(f"Remote-first: Using existing prompt '{name}'")
                    return
            except Exception:
                logger.debug(f"Remote-first: Prompt '{name}' not found, creating new")

        # Check if we should skip creation (prompt exists and force_create=False)
        if not force_create_new_version:
            try:
                existing = client.pull_prompt(name)
                if existing:
                    logger.debug(f"Prompt '{name}' exists, skipping (force_create_new_version=False)")
                    return
            except Exception:
                pass  # Prompt doesn't exist, will create

        # Push to LangSmith
        try:
            if metadata and metadata.get("model"):
                chain = prompt_obj | metadata["model"]
                client.push_prompt(name, object=chain)
            else:
                client.push_prompt(name, object=prompt_obj)
            logger.debug(f"Created prompt '{name}' in LangSmith")
        except LangSmithConflictError:
            logger.debug(f"Prompt '{name}' already exists with same content")

    @BaseObservabilityPlatform.requires_env_vars
    def pull_prompt(
        self,
        name: str,
        template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
        **kwargs,
    ) -> PromptReturnType:
        """Pull a prompt from LangSmith.

        Args:
            name: Name of the prompt
            template_format: Format for the template
            **kwargs: Additional parameters (e.g., version, label)

        Returns:
            ChatPromptTemplate

        """
        client = LangsmithClient()
        prompt_info = client.pull_prompt(name)
        return self._process_prompt_object(prompt_info, template_format=template_format)

    @BaseObservabilityPlatform.requires_env_vars
    def delete_prompt(self, name: str) -> None:
        """Delete a prompt from LangSmith.

        Args:
            name: Name of the prompt to delete

        """
        client = LangsmithClient()
        client.delete_prompt(name)
        logger.debug(f"Deleted prompt '{name}' from LangSmith")
