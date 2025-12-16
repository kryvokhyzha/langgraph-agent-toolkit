import hashlib
import inspect
import json
from contextlib import contextmanager
from typing import Any, Dict, Literal, Optional, Tuple, Union

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform
from langgraph_agent_toolkit.core.observability.types import PromptReturnType, PromptTemplateType
from langgraph_agent_toolkit.helper.constants import DEFAULT_CACHE_TTL_SECOND
from langgraph_agent_toolkit.helper.logging import logger


try:
    from langfuse import get_client, propagate_attributes
    from langfuse.langchain import CallbackHandler

    _IS_NEW_LANGFUSE = True
except (ModuleNotFoundError, ImportError) as e:
    logger.debug(f"Falling back to old Langfuse SDK: {e}")
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler

    _IS_NEW_LANGFUSE = False


def _get_langfuse_client():
    """Get the appropriate Langfuse client based on SDK version."""
    return get_client() if _IS_NEW_LANGFUSE else Langfuse()


class LangfuseObservability(BaseObservabilityPlatform):
    """Langfuse implementation of observability platform."""

    def __init__(self, remote_first: bool = False):
        """Initialize LangfuseObservability.

        Args:
            remote_first: If True, prioritize remote prompts over local ones.

        """
        super().__init__(remote_first)
        self.required_vars = ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_HOST"]

    @BaseObservabilityPlatform.requires_env_vars
    def get_callback_handler(self, **kwargs) -> CallbackHandler:
        """Get the Langfuse callback handler for LangChain."""
        valid_params = set(inspect.signature(CallbackHandler.__init__).parameters.keys())
        valid_params.discard("self")
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        return CallbackHandler(**filtered_kwargs)

    def before_shutdown(self) -> None:
        """Flush any pending data before shutdown."""
        _get_langfuse_client().flush()

    @BaseObservabilityPlatform.requires_env_vars
    def record_feedback(self, run_id: str, key: str, score: float, **kwargs) -> None:
        """Record feedback/score for a trace in Langfuse."""
        client = _get_langfuse_client()

        if _IS_NEW_LANGFUSE:
            valid_params = set(inspect.signature(client.create_score).parameters.keys())
            valid_params.discard("self")
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

            # Convert UUID to Langfuse trace ID format (32 lowercase hex chars without hyphens)
            trace_id = str(run_id).replace("-", "").lower()

            with propagate_attributes(user_id=kwargs.get("user_id")):
                client.create_score(
                    name=key,
                    value=score,
                    trace_id=trace_id,
                    **filtered_kwargs,
                )
        else:
            client.score(
                trace_id=run_id,
                name=key,
                value=score,
                **kwargs,
            )

    def _compute_prompt_hash(self, prompt_template: PromptTemplateType) -> str:
        """Compute a hash of the prompt content to detect changes."""
        if isinstance(prompt_template, str):
            content_to_hash = prompt_template
        elif isinstance(prompt_template, list):
            content_to_hash = json.dumps(prompt_template, sort_keys=True)
        else:
            content_to_hash = str(prompt_template)

        return hashlib.md5(content_to_hash.encode("utf-8")).hexdigest()

    @BaseObservabilityPlatform.requires_env_vars
    def push_prompt(
        self,
        name: str,
        prompt_template: PromptTemplateType,
        metadata: Optional[Dict[str, Any]] = None,
        force_create_new_version: bool = True,
    ) -> None:
        """Push a prompt to Langfuse.

        Args:
            name: Name of the prompt
            prompt_template: The prompt template (string or list of message dicts)
            metadata: Optional metadata including 'labels'
            force_create_new_version: If True, always create a new version

        """
        client = _get_langfuse_client()
        labels = metadata.get("labels", ["production"]) if metadata else ["production"]

        # Check if remote_first is enabled - use existing remote prompt if available
        if self.remote_first:
            try:
                existing = client.get_prompt(name=name)
                if existing:
                    logger.debug(f"Remote-first: Using existing prompt '{name}'")
                    return
            except Exception:
                logger.debug(f"Remote-first: Prompt '{name}' not found, creating new")

        # Generate hash for the current prompt to detect content changes
        prompt_hash = self._compute_prompt_hash(prompt_template)

        # Check if prompt exists and compare content
        existing_prompt = None
        content_changed = True

        try:
            existing_prompt = client.get_prompt(name=name)

            # Check if content has changed by comparing hashes stored in commit_message or tags
            existing_hash = None
            if hasattr(existing_prompt, "commit_message") and existing_prompt.commit_message:
                existing_hash = existing_prompt.commit_message
            elif hasattr(existing_prompt, "tags") and existing_prompt.tags and len(existing_prompt.tags) > 0:
                existing_hash = existing_prompt.tags[0]

            if existing_hash and existing_hash == prompt_hash:
                content_changed = False
                logger.debug(f"Prompt '{name}' content unchanged (hash: {existing_hash})")
            else:
                logger.debug(f"Prompt '{name}' content changed (old: {existing_hash}, new: {prompt_hash})")
        except Exception:
            logger.debug(f"Prompt '{name}' not found, will create new")

        # Decide whether to create a new version
        should_create = force_create_new_version or existing_prompt is None or content_changed

        if not should_create:
            logger.debug(f"Reusing existing prompt '{name}' (unchanged, force_create=False)")
            return

        # Create the prompt
        prompt_type = "text" if isinstance(prompt_template, str) else "chat"
        client.create_prompt(
            name=name,
            prompt=prompt_template,
            labels=labels,
            type=prompt_type,
            tags=[prompt_hash],
            commit_message=prompt_hash,
        )
        logger.debug(f"Created prompt '{name}' in Langfuse")

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
        """Pull a prompt from Langfuse.

        Args:
            name: Name of the prompt
            return_with_prompt_object: If True, return tuple of (prompt, langfuse_prompt)
            cache_ttl_seconds: Cache TTL for the prompt
            template_format: Format for the template
            label: Optional label to fetch specific version
            version: Optional version number to fetch
            **kwargs: Additional kwargs (prompt_label, prompt_version as aliases)

        Returns:
            ChatPromptTemplate or tuple of (ChatPromptTemplate, langfuse_prompt)

        """
        client = _get_langfuse_client()

        # Build kwargs for get_prompt
        get_kwargs: Dict[str, Any] = {"name": name, "cache_ttl_seconds": cache_ttl_seconds}

        if label or kwargs.get("prompt_label"):
            get_kwargs["label"] = label or kwargs.get("prompt_label")

        if version is not None or kwargs.get("prompt_version"):
            get_kwargs["version"] = version if version is not None else kwargs.get("prompt_version")

        try:
            langfuse_prompt = client.get_prompt(**get_kwargs)
        except Exception as e:
            logger.debug(f"Prompt not found with parameters {get_kwargs}: {e}")
            langfuse_prompt = client.get_prompt(name=name, cache_ttl_seconds=cache_ttl_seconds)

        prompt = self._process_prompt_object(langfuse_prompt.prompt, template_format=template_format)

        return (prompt, langfuse_prompt) if return_with_prompt_object else prompt

    @BaseObservabilityPlatform.requires_env_vars
    def delete_prompt(self, name: str) -> None:
        """Delete a prompt - Langfuse doesn't support programmatic deletion."""
        logger.warning(f"Langfuse doesn't support prompt deletion via API. Prompt '{name}' not deleted.")

    @contextmanager
    @BaseObservabilityPlatform.requires_env_vars
    def trace_context(self, run_id: str, **kwargs):
        """Create a Langfuse trace context with predefined trace ID.

        Args:
            run_id: The run ID to use as trace ID
            **kwargs: Additional context parameters (user_id, input, agent_name, etc.)

        Yields:
            The span object (new SDK) or None (old SDK)

        """
        if not _IS_NEW_LANGFUSE:
            yield None
            return

        client = get_client()
        trace_id = str(run_id).replace("-", "").lower()
        agent_name = kwargs.get("agent_name", "agent-execution")

        with client.start_as_current_span(
            name=agent_name,
            trace_context={"trace_id": trace_id},
        ) as span:
            # Update trace with context if available
            update_params = {}
            if kwargs.get("user_id"):
                update_params["user_id"] = kwargs["user_id"]
            if kwargs.get("input"):
                update_params["input"] = kwargs["input"]

            if update_params:
                span.update_trace(**update_params)

            try:
                yield span
            finally:
                if kwargs.get("output"):
                    span.update_trace(output=kwargs["output"])
