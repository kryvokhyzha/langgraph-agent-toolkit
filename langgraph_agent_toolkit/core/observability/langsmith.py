from typing import Any, Dict, Optional

from langsmith import Client as LangsmithClient

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

        # Handle versioning
        if create_new_version:
            try:
                client.pull_prompt(name)
                client.delete_prompt(name)
            except Exception:
                pass

        # Convert to proper format
        prompt_obj = self._convert_to_chat_prompt(prompt_template)

        # Push to LangSmith
        if metadata and metadata.get("model"):
            chain = prompt_obj | metadata["model"]
            url = client.push_prompt(name, object=chain)
        else:
            url = client.push_prompt(name, object=prompt_obj)

        # Update metadata and save locally
        full_metadata = metadata.copy() if metadata else {}
        full_metadata["langsmith_url"] = url
        full_metadata["original_prompt"] = prompt_obj

        # Extract template for local storage
        if isinstance(prompt_template, str):
            template_str = prompt_template
        elif isinstance(prompt_template, list) and all(isinstance(msg, dict) for msg in prompt_template):
            # For list of message dicts, serialize to a readable format
            template_str = ""
            for msg in prompt_template:
                template_str += f"[{msg['role']}]: {msg['content']}\n\n"
        else:
            template_str = str(prompt_obj)

        super().push_prompt(name, template_str, full_metadata)

    @BaseObservabilityPlatform.requires_env_vars
    def pull_prompt(self, name: str) -> PromptReturnType:
        """Pull a prompt from LangSmith.

        Args:
            name: Name of the prompt to retrieve

        Returns:
            The prompt object

        """
        try:
            client = LangsmithClient()
            prompt_info = client.pull_prompt(name)
            return prompt_info
        except Exception as e:
            logger.warning(f"Failed to pull prompt from LangSmith: {e}")
            return super().pull_prompt(name)

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
