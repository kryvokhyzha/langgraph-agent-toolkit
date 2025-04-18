from typing import Any, Dict, Literal, Optional, Tuple, Union

from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform
from langgraph_agent_toolkit.core.observability.types import PromptReturnType, PromptTemplateType
from langgraph_agent_toolkit.helper.constants import DEFAULT_CACHE_TTL_SECOND
from langgraph_agent_toolkit.helper.logging import logger


class LangfuseObservability(BaseObservabilityPlatform):
    """Langfuse implementation of observability platform."""

    def __init__(self, prompts_dir: Optional[str] = None):
        super().__init__(prompts_dir)
        self.required_vars = ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_HOST"]

    @BaseObservabilityPlatform.requires_env_vars
    def get_callback_handler(self, **kwargs) -> CallbackHandler:
        return CallbackHandler(**kwargs)

    def before_shutdown(self) -> None:
        Langfuse().flush()

    @BaseObservabilityPlatform.requires_env_vars
    def record_feedback(self, run_id: str, key: str, score: float, **kwargs) -> None:
        Langfuse().score(
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
        langfuse = Langfuse()
        labels = metadata.get("labels", ["production"]) if metadata else ["production"]

        # Handle existing prompt versions - custom implementation for Langfuse
        existing_prompt = None
        if not create_new_version:
            try:
                existing_prompt = langfuse.get_prompt(name=name)
                logger.debug(f"Using existing prompt '{name}' as create_new_version is False")
            except Exception:
                logger.debug(f"Existing prompt '{name}' not found, will create a new one")

        prompt_obj = self._convert_to_chat_prompt(prompt_template)
        type_prompt = "text" if isinstance(prompt_template, str) else "chat"

        if existing_prompt:
            langfuse_prompt = existing_prompt
        else:
            langfuse_prompt = langfuse.create_prompt(
                name=name,
                prompt=prompt_template,
                labels=labels,
                type=type_prompt,
            )

        full_metadata = metadata.copy() if metadata else {}
        full_metadata["langfuse_prompt"] = langfuse_prompt
        full_metadata["original_prompt"] = prompt_obj

        super().push_prompt(name, prompt_template, full_metadata)

    @BaseObservabilityPlatform.requires_env_vars
    def pull_prompt(
        self,
        name: str,
        return_with_prompt_object: bool = False,
        cache_ttl_seconds: Optional[int] = DEFAULT_CACHE_TTL_SECOND,
        template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
        label: Optional[str] = None,
        version: Optional[int] = None,
        **kwargs,
    ) -> Union[PromptReturnType, Tuple[PromptReturnType, Any]]:
        try:
            langfuse = Langfuse()
            get_prompt_kwargs = {"name": name, "cache_ttl_seconds": cache_ttl_seconds}

            if label:
                get_prompt_kwargs["label"] = label
            elif kwargs.get("prompt_label"):
                get_prompt_kwargs["label"] = kwargs.get("prompt_label")

            if version is not None:
                get_prompt_kwargs["version"] = version
            elif kwargs.get("prompt_version"):
                get_prompt_kwargs["version"] = kwargs.get("prompt_version")

            try:
                langfuse_prompt = langfuse.get_prompt(**get_prompt_kwargs)
            except Exception as e:
                logger.debug(f"Prompt not found with parameters: {e}")
                langfuse_prompt = langfuse.get_prompt(name=name, cache_ttl_seconds=cache_ttl_seconds)

            # Process the prompt object using the base class helper
            prompt = self._process_prompt_object(langfuse_prompt.prompt, template_format=template_format)

            return (prompt, langfuse_prompt) if return_with_prompt_object else prompt

        except Exception as e:
            logger.warning(f"Failed to pull prompt from Langfuse: {e}")
            local_prompt = super().pull_prompt(name, template_format=template_format, **kwargs)
            return (local_prompt, None) if return_with_prompt_object else local_prompt

    @BaseObservabilityPlatform.requires_env_vars
    def delete_prompt(self, name: str) -> None:
        logger.warning(f"Skipping deletion of prompt '{name}' from Langfuse")
        super().delete_prompt(name)
