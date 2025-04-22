from typing import Any, Dict, Literal, Optional

from langsmith import Client as LangsmithClient
from langsmith.utils import LangSmithConflictError

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform
from langgraph_agent_toolkit.core.observability.types import PromptReturnType, PromptTemplateType
from langgraph_agent_toolkit.helper.logging import logger


class LangsmithObservability(BaseObservabilityPlatform):
    """Langsmith implementation of observability platform."""

    def __init__(self, prompts_dir: Optional[str] = None):
        """Initialize LangsmithObservability.

        Args:
            prompts_dir: Optional directory to store prompts locally. If None, a system temp directory is used.

        """
        super().__init__(prompts_dir)
        # Set required environment variables explicitly
        self.required_vars = ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"]

    @BaseObservabilityPlatform.requires_env_vars
    def get_callback_handler(self, **kwargs) -> None:
        """Get the callback handler for the observability platform."""
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
            kwargs["extra"] = kwargs["extra"] or {}
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
        create_new_version: bool = True,
    ) -> None:
        """Push a prompt to LangSmith."""
        client = LangsmithClient()

        # Convert to proper format
        prompt_obj = self._convert_to_chat_prompt(prompt_template)

        # Handle existing prompt versions
        existing_prompt, existing_url = self._handle_existing_prompt(
            name, create_new_version, client, client_pull_method="pull_prompt", client_delete_method="delete_prompt"
        )

        url = None

        # Push to LangSmith if we don't have an existing prompt
        if existing_prompt is None:
            try:
                if metadata and metadata.get("model"):
                    chain = prompt_obj | metadata["model"]
                    url = client.push_prompt(name, object=chain)
                else:
                    url = client.push_prompt(name, object=prompt_obj)
                logger.debug(f"Created new prompt '{name}' in LangSmith")
            except LangSmithConflictError as e:
                logger.debug(f"Prompt '{name}' unchanged, using existing version: {e}")
                try:
                    # Try to retrieve the existing prompt without modifying it
                    existing_prompt = client.pull_prompt(name)
                    url = getattr(existing_prompt, "url", None)
                    logger.debug(f"Using existing prompt '{name}' due to conflict")
                except Exception as fetch_err:
                    logger.warning(f"Failed to retrieve existing prompt after conflict: {fetch_err}")
        else:
            # Use the existing prompt that was found earlier
            url = existing_url
            logger.debug(f"Reusing existing prompt '{name}' in LangSmith")

        # Update metadata and save locally
        full_metadata = metadata.copy() if metadata else {}
        full_metadata["langsmith_url"] = url
        full_metadata["original_prompt"] = prompt_obj

        # Extract template for local storage and save
        template_str = self._extract_template_string(prompt_template, prompt_obj)
        super().push_prompt(name, template_str, full_metadata)

    @BaseObservabilityPlatform.requires_env_vars
    def pull_prompt(
        self,
        name: str,
        template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
        **kwargs,
    ) -> PromptReturnType:
        """Pull a prompt from LangSmith."""
        try:
            client = LangsmithClient()
            prompt_info = client.pull_prompt(name)

            # Process the prompt into a standard format
            return self._process_prompt_object(prompt_info, template_format=template_format)

        except Exception as e:
            logger.warning(f"Failed to pull prompt from remote platform: {e}")

            # Fall back to local storage
            return self._local_pull_prompt(name, template_format=template_format, **kwargs)

    @BaseObservabilityPlatform.requires_env_vars
    def delete_prompt(self, name: str) -> None:
        """Delete a prompt from LangSmith.

        Args:
            name: Name of the prompt to delete

        """
        client = LangsmithClient()
        client.delete_prompt(name)

        # Also delete the local files
        super().delete_prompt(name)
