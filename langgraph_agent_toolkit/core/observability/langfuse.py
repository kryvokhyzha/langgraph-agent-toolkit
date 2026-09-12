import asyncio
import hashlib
import inspect
import json
import os
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from importlib.metadata import version
from threading import Lock
from typing import Any, Dict, Literal, Optional, Tuple, Union
from uuid import uuid4

import langfuse as _langfuse
from langchain_core.callbacks import BaseCallbackHandler

from langgraph_agent_toolkit.core.observability.base import (
    BaseObservabilityPlatform,
    PromptReturnType,
    PromptTemplateType,
)
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.logging import logger


_SDK_MAJOR = int(version("langfuse").split(".")[0])
if _SDK_MAJOR not in (2, 3, 4):
    raise ImportError(f"Unsupported Langfuse SDK major version: {_SDK_MAJOR}")

_IS_NEW_LANGFUSE = _SDK_MAJOR >= 3
Langfuse = _langfuse.Langfuse
get_client = getattr(_langfuse, "get_client", None)
propagate_attributes = getattr(_langfuse, "propagate_attributes", None)


def _legacy_message(message: Any) -> Dict[str, Any]:
    """Retain message data and add the role used by the Langfuse chat view."""
    payload = message.model_dump()
    roles = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool", "function": "function"}
    payload["role"] = getattr(message, "role", roles.get(message.type, message.type))
    return payload


def _legacy_usage(response: Any) -> Optional[Dict[str, int]]:
    """Keep token categories without counting cached or reasoning tokens twice."""
    message = None
    if response.generations and response.generations[0]:
        message = getattr(response.generations[0][0], "message", None)
    raw = getattr(message, "usage_metadata", None) or (response.llm_output or {}).get("token_usage") or {}
    if not raw:
        return None
    usage = {}
    for field, aliases in {
        "input": ("input_tokens", "prompt_tokens", "input"),
        "output": ("output_tokens", "completion_tokens", "output"),
        "total": ("total_tokens", "total"),
    }.items():
        for alias in aliases:
            if alias in raw:
                usage[field] = raw[alias]
                break
    for side in ("input", "output"):
        for name, count in (raw.get(f"{side}_token_details") or {}).items():
            usage[f"{side}_{name}"] = count
            if side in usage:
                usage[side] = max(0, usage[side] - count)
    return usage or None


class _LegacyCallbackHandler(BaseCallbackHandler):
    """Connect LangChain 1 callbacks to the Langfuse SDK v2 client.

    The SDK v2 callback imports modules that LangChain 1 removed.
    Each handler belongs to one run and uses one fixed trace.
    """

    def __init__(self, stateful_client: Any) -> None:
        self.trace = stateful_client
        self.runs: Dict[Any, Any] = {}
        self._generation_ids: set[Any] = set()
        self._first_tokens: set[Any] = set()
        self._lock = Lock()

    def _start(
        self,
        serialized: Optional[Dict[str, Any]],
        input: Any,
        run_id: Any,
        parent_run_id: Any = None,
        *,
        generation: bool = False,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            parent = self.runs.get(parent_run_id, self.trace)
            name = kwargs.get("name") or (serialized or {}).get("name") or ("generation" if generation else "chain")
            arguments = {"id": str(run_id), "name": name, "input": input, "metadata": kwargs.get("metadata")}
            if generation:
                invocation = kwargs.get("invocation_params") or {}
                arguments["model"] = (
                    invocation.get("model_name")
                    or invocation.get("model")
                    or (kwargs.get("metadata") or {}).get("ls_model_name")
                )
                self.runs[run_id] = parent.generation(**arguments)
                self._generation_ids.add(run_id)
            else:
                self.runs[run_id] = parent.span(**arguments)

    def _end(self, run_id: Any, **kwargs: Any) -> None:
        with self._lock:
            observation = self.runs.pop(run_id, None)
            self._generation_ids.discard(run_id)
            self._first_tokens.discard(run_id)
        if observation is not None:
            observation.end(**kwargs)

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._start(serialized, inputs, run_id, parent_run_id, **kwargs)

    def on_chain_end(self, outputs: Any, *, run_id: Any, **kwargs: Any) -> None:
        self._end(run_id, output=outputs)

    def on_chain_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._end(run_id, level="ERROR", status_message=str(error))

    def on_chat_model_start(
        self,
        serialized: Optional[Dict[str, Any]],
        messages: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        inputs = [_legacy_message(message) for batch in messages for message in batch]
        self._start(serialized, inputs, run_id, parent_run_id, generation=True, **kwargs)

    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]],
        prompts: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._start(serialized, prompts, run_id, parent_run_id, generation=True, **kwargs)

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        outputs = [
            _legacy_message(generation.message) if hasattr(generation, "message") else generation.text
            for batch in response.generations
            for generation in batch
        ]
        output = outputs[0] if len(outputs) == 1 else outputs
        usage = _legacy_usage(response)
        self._end(run_id, output=output, usage=usage, usage_details=usage)

    def on_llm_new_token(self, token: str, *, run_id: Any, **kwargs: Any) -> None:
        with self._lock:
            if run_id not in self._generation_ids or run_id in self._first_tokens:
                return
            self.runs[run_id].update(completion_start_time=datetime.now(timezone.utc))
            self._first_tokens.add(run_id)

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self.on_chain_error(error, run_id=run_id)

    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._start(serialized, kwargs.get("inputs") or input_str, run_id, parent_run_id, **kwargs)

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        self._end(run_id, output=output)

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self.on_chain_error(error, run_id=run_id)

    def on_retriever_start(
        self, serialized: Optional[Dict[str, Any]], query: str, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        self._start(serialized, query, run_id, parent_run_id, **kwargs)

    def on_retriever_end(self, documents: Any, *, run_id: Any, **kwargs: Any) -> None:
        self._end(run_id, output=[document.model_dump() for document in documents])

    def on_retriever_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self.on_chain_error(error, run_id=run_id)


if _IS_NEW_LANGFUSE:
    from langfuse.langchain import CallbackHandler
else:
    CallbackHandler = _LegacyCallbackHandler


def _configuration() -> Dict[str, Any]:
    """Use validated settings and retain direct environment configuration."""

    def credential(name: str) -> Optional[str]:
        value = getattr(settings, name, None)
        return value.get_secret_value() if value is not None else os.environ.get(name)

    host = settings.LANGFUSE_HOST
    if "LANGFUSE_HOST" not in settings.model_fields_set:
        host = os.environ.get("LANGFUSE_HOST") or host
    return {
        "public_key": credential("LANGFUSE_PUBLIC_KEY"),
        "secret_key": credential("LANGFUSE_SECRET_KEY"),
        "host": os.environ.get("LANGFUSE_BASE_URL") or host,
        "environment": settings.LANGFUSE_TRACING_ENVIRONMENT,
        "flush_at": settings.LANGFUSE_FLUSH_AT,
        "flush_interval": settings.LANGFUSE_FLUSH_INTERVAL,
        "timeout": settings.LANGFUSE_TIMEOUT,
        "debug": settings.LANGFUSE_DEBUG,
        "sample_rate": settings.LANGFUSE_SAMPLE_RATE,
    }


def _get_langfuse_client() -> Any:
    """Initialize the selected project with explicit settings."""
    return Langfuse(**_configuration())


def _supported_kwargs(function: Any, values: Dict[str, Any]) -> Dict[str, Any]:
    """Keep named parameters that the installed SDK accepts."""
    parameters = inspect.signature(function).parameters
    return {key: value for key, value in values.items() if key in parameters}


def _set_trace_io(span: Any, **io: Any) -> None:
    """Set root observation I/O and retain legacy trace I/O."""
    io = {key: value for key, value in io.items() if value is not None}
    if not io:
        return
    if hasattr(span, "update"):
        span.update(**io)
    if hasattr(span, "update_trace"):
        span.update_trace(**io)
    elif hasattr(span, "set_trace_io"):
        span.set_trace_io(**io)


class LangfuseObservability(BaseObservabilityPlatform):
    """Langfuse observability platform."""

    def __init__(self, remote_first: bool = False):
        """Initialize LangfuseObservability.

        Args:
            remote_first: Prioritize remote prompts when `True`.

        """
        super().__init__(remote_first)
        self.required_vars = ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY"]
        self._client = None
        self._public_key: Optional[str] = None
        self._client_lock = Lock()

    def validate_environment(self) -> bool:
        """Accept the current URL variable and its legacy alias."""
        if self._client is not None:
            return True
        config = _configuration()
        missing = [
            name
            for name, key in (("LANGFUSE_PUBLIC_KEY", "public_key"), ("LANGFUSE_SECRET_KEY", "secret_key"))
            if not config[key]
        ]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        return True

    def _get_client(self) -> Any:
        """Reuse one client for prompts, feedback, and traces."""
        with self._client_lock:
            if self._client is None:
                self._public_key = _configuration()["public_key"]
                self._client = _get_langfuse_client()
            return self._client

    @BaseObservabilityPlatform.requires_env_vars
    def get_callback_handler(self, **kwargs) -> CallbackHandler:
        """Get the LangChain Langfuse callback handler.

        Create a separate handler for each run.
        SDK v2 handlers keep mutable trace state.
        """
        if not _IS_NEW_LANGFUSE:
            trace = self._get_client().trace(
                id=str(kwargs.get("run_id") or uuid4()),
                user_id=kwargs.get("user_id"),
                session_id=kwargs.get("session_id"),
            )
            kwargs = {**kwargs, "stateful_client": trace, "update_stateful_client": False}
        else:
            self._get_client()
            kwargs["public_key"] = self._public_key
        return CallbackHandler(**_supported_kwargs(CallbackHandler, kwargs))

    def before_shutdown(self) -> None:
        """Flush an initialized client without creating a new client."""
        if self._client is not None:
            if _IS_NEW_LANGFUSE:
                self._client.flush()
            else:
                self._client.shutdown()
                self._client = None

    @BaseObservabilityPlatform.requires_env_vars
    def record_feedback(self, run_id: str, key: str, score: float, **kwargs) -> None:
        """Attach feedback to the same trace ID as the run."""
        client = self._get_client()
        method = client.create_score if _IS_NEW_LANGFUSE else client.score
        filtered_kwargs = _supported_kwargs(method, kwargs)
        for key_to_remove in ("name", "value", "trace_id"):
            filtered_kwargs.pop(key_to_remove, None)
        trace_id = str(run_id).replace("-", "").lower() if _IS_NEW_LANGFUSE else str(run_id)
        method(name=key, value=score, trace_id=trace_id, **filtered_kwargs)

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
        client = self._get_client()
        labels = metadata.get("labels", ["production"]) if metadata else ["production"]

        # Use an existing remote prompt when `remote_first` is enabled.
        if self.remote_first:
            try:
                existing = client.get_prompt(name=name)
                if existing:
                    logger.debug(f"Remote-first: Using existing prompt '{name}'")
                    return
            except Exception as exc:
                if getattr(exc, "status_code", None) != 404:
                    raise
                logger.debug(f"Remote-first: Prompt '{name}' not found, creating new")

        # Generate a hash to detect prompt content changes.
        prompt_hash = self._compute_prompt_hash(prompt_template)

        # Get the existing prompt and compare its content.
        existing_prompt = None
        content_changed = True

        try:
            existing_prompt = client.get_prompt(name=name)

            # Compare hashes in `commit_message` or `tags`.
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
        except Exception as exc:
            if getattr(exc, "status_code", None) != 404:
                raise
            logger.debug(f"Prompt '{name}' not found, will create new")

        # Decide whether to create a version.
        should_create = force_create_new_version or existing_prompt is None or content_changed

        if not should_create:
            logger.debug(f"Reusing existing prompt '{name}' (unchanged, force_create=False)")
            return

        # Create the prompt.
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
        cache_ttl_seconds: Optional[int] = settings.LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS,
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
        client = self._get_client()

        # Build arguments for `get_prompt`.
        get_kwargs: Dict[str, Any] = {"name": name, "cache_ttl_seconds": cache_ttl_seconds}

        if label or kwargs.get("prompt_label"):
            get_kwargs["label"] = label or kwargs.get("prompt_label")

        if version is not None or kwargs.get("prompt_version") is not None:
            get_kwargs["version"] = version if version is not None else kwargs.get("prompt_version")
            get_kwargs.pop("label", None)

        langfuse_prompt = client.get_prompt(**get_kwargs)

        prompt = self._process_prompt_object(langfuse_prompt.prompt, template_format=template_format)

        return (prompt, langfuse_prompt) if return_with_prompt_object else prompt

    async def apull_prompt(
        self,
        name: str,
        return_with_prompt_object: bool = False,
        cache_ttl_seconds: Optional[int] = settings.LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS,
        template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
        label: Optional[str] = None,
        version: Optional[int] = None,
        **kwargs,
    ) -> Union[PromptReturnType, Tuple[PromptReturnType, Any]]:
        """Asynchronously run `pull_prompt` in a thread pool."""
        return await asyncio.to_thread(
            self.pull_prompt,
            name,
            return_with_prompt_object=return_with_prompt_object,
            cache_ttl_seconds=cache_ttl_seconds,
            template_format=template_format,
            label=label,
            version=version,
            **kwargs,
        )

    @BaseObservabilityPlatform.requires_env_vars
    def delete_prompt(self, name: str) -> None:
        """Delete all prompt versions when the installed SDK supports deletion."""
        delete = getattr(self._get_client().api.prompts, "delete", None)
        if delete is None:
            raise NotImplementedError("This Langfuse SDK does not support prompt deletion")
        delete(prompt_name=name)

    @contextmanager
    @BaseObservabilityPlatform.requires_env_vars
    def trace_context(self, run_id: str, **kwargs):
        """Open the root trace and propagate user and session attributes."""
        client = self._get_client()
        agent_name = kwargs.get("agent_name", "agent-execution")
        attributes = {key: kwargs[key] for key in ("user_id", "session_id") if kwargs.get(key) is not None}
        if not _IS_NEW_LANGFUSE:
            trace = client.trace(id=str(run_id), name=agent_name, input=kwargs.get("input"), **attributes)
            try:
                yield trace
            finally:
                if kwargs.get("output") is not None:
                    trace.update(output=kwargs["output"])
            return

        trace_id = str(run_id).replace("-", "").lower()
        start_observation = getattr(client, "start_as_current_observation", None) or client.start_as_current_span
        attrs_cm = propagate_attributes(**attributes) if attributes and propagate_attributes else nullcontext()
        with attrs_cm, start_observation(name=agent_name, trace_context={"trace_id": trace_id}) as span:
            if _SDK_MAJOR == 3 and hasattr(span, "update_trace"):
                span.update_trace(**attributes)
            _set_trace_io(span, input=kwargs.get("input"))
            try:
                yield span
            finally:
                _set_trace_io(span, output=kwargs.get("output"))

    def update_trace(self, trace, **attributes) -> None:
        """Record final output on the root trace before it closes."""
        if trace is None:
            return
        if _IS_NEW_LANGFUSE:
            _set_trace_io(trace, **attributes)
        else:
            trace.update(**attributes)
