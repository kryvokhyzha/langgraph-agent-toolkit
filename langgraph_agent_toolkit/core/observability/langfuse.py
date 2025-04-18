from typing import Any, Dict, Optional, Tuple, Union

from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform
from langgraph_agent_toolkit.core.observability.types import PromptReturnType, PromptTemplateType
from langgraph_agent_toolkit.helper.constants import DEFAULT_CACHE_TTL_SECOND
from langgraph_agent_toolkit.helper.logging import logger


class LangfuseObservability(BaseObservabilityPlatform):
    """Langfuse implementation of observability platform."""

    def __init__(self, prompts_dir: Optional[str] = None):
        """Initialize LangfuseObservability.

        Args:
            prompts_dir: Optional directory to store prompts locally. If None, a system temp directory is used.

        """
        super().__init__(prompts_dir)
        # Set required environment variables explicitly
        self.required_vars = ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_HOST"]

    @BaseObservabilityPlatform.requires_env_vars
    def get_callback_handler(self, **kwargs) -> CallbackHandler:
        """Get Langfuse callback handler.

        Args:
            **kwargs: Any keyword arguments

        Returns:
            A configured Langfuse CallbackHandler

        """
        return CallbackHandler(**kwargs)

    def before_shutdown(self) -> None:
        """Perform any necessary cleanup before shutdown."""
        Langfuse().flush()

    @BaseObservabilityPlatform.requires_env_vars
    def record_feedback(self, run_id: str, key: str, score: float, **kwargs) -> None:
        """Record feedback for a run to Langfuse."""
        client = Langfuse()
        client.score(
            trace_id=run_id,
            name=key,
            value=score,
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
        """Push a prompt to Langfuse."""
        langfuse = Langfuse()

        # Create labels from metadata if available
        labels = metadata.get("labels", []) if metadata else ["production"]

        # Check versioning (Langfuse handles this internally)
        if create_new_version:
            try:
                langfuse.get_prompt(name=name)
            except Exception:
                pass

        # Convert to proper format
        prompt_obj = self._convert_to_chat_prompt(prompt_template)

        # Create prompt in Langfuse
        type_prompt = "text" if isinstance(prompt_template, str) else "chat"
        langfuse_prompt = langfuse.create_prompt(
            name=name,
            prompt=prompt_template,
            labels=labels,
            type=type_prompt,
        )

        # Save metadata
        full_metadata = metadata.copy() if metadata else {}
        full_metadata["langfuse_prompt"] = langfuse_prompt
        full_metadata["original_prompt"] = prompt_obj

        # Save locally
        super().push_prompt(name, prompt_template, full_metadata)

    @BaseObservabilityPlatform.requires_env_vars
    def pull_prompt(
        self,
        name: str,
        return_with_prompt_object: bool = False,
        cache_ttl_seconds: Optional[int] = DEFAULT_CACHE_TTL_SECOND,
    ) -> Union[PromptReturnType, Tuple[PromptReturnType, Any]]:
        """Pull a prompt from Langfuse.

        Args:
            name: Name of the prompt to retrieve
            return_with_prompt_object: If True, returns a tuple of (prompt, langfuse_prompt_object)
            cache_ttl_seconds: Cache TTL in seconds for the prompt retrieval

        Returns:
            The prompt template or a tuple containing the prompt and langfuse object

        """
        try:
            langfuse = Langfuse()
            langfuse_prompt = langfuse.get_prompt(name=name, cache_ttl_seconds=cache_ttl_seconds)

            if not langfuse_prompt:
                raise ValueError(f"Prompt '{name}' not found in Langfuse")

            langchain_prompts = langfuse_prompt.get_langchain_prompt()

            # Use get_langchain_prompt method instead of creating a new ChatPromptTemplate
            if return_with_prompt_object:
                return langchain_prompts, langfuse_prompt
            else:
                return langchain_prompts

        except Exception as e:
            logger.warning(f"Failed to pull prompt from Langfuse: {e}")

            local_prompt = super().pull_prompt(name)

            if return_with_prompt_object:
                return local_prompt, None
            else:
                return local_prompt

    @BaseObservabilityPlatform.requires_env_vars
    def delete_prompt(self, name: str) -> None:
        """Delete a prompt from Langfuse.

        Args:
            name: Name of the prompt to delete

        """
        logger.warning(f"Skipping deletion of prompt '{name}' from Langfuse")

        # Delete the local files
        super().delete_prompt(name)
