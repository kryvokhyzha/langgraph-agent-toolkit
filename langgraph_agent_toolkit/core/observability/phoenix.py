"""Send isolated agent traces and prompts to Phoenix."""

import json
import os
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from math import isfinite
from threading import Lock
from typing import Any, Literal
from uuid import UUID

import httpx
from langchain_core.messages import BaseMessage
from tenacity import Retrying, retry_if_exception, stop_before_delay, wait_exponential


try:
    from openinference.instrumentation import OITracer, TraceConfig, using_attributes

    # The upstream package does not export its callback from the public namespace.
    from openinference.instrumentation.langchain._tracer import OpenInferenceTracer
    from opentelemetry.context import Context
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
    from phoenix.client import Client
    from phoenix.client.types.prompts import PromptVersion
except ImportError as exc:
    raise ImportError("Phoenix requires the optional dependencies. Install langgraph-agent-toolkit[phoenix].") from exc

from langgraph_agent_toolkit.core.observability.base import (
    BaseObservabilityPlatform,
    PromptReturnType,
    PromptTemplateType,
)
from langgraph_agent_toolkit.core.observability.masking import mask_secret_values
from langgraph_agent_toolkit.core.settings import settings


_trace_id: ContextVar[int | None] = ContextVar("phoenix_trace_id", default=None)


class _RunIdGenerator(RandomIdGenerator):
    def generate_trace_id(self) -> int:
        return _trace_id.get() or super().generate_trace_id()


def _callback_data(value: Any) -> Any:
    """Mask typed secrets and retain the LangChain message format."""
    active: set[int] = set()

    def mask(item: Any) -> Any:
        if isinstance(item, BaseMessage):
            return mask_secret_values(data=item.to_json())
        if not isinstance(item, (dict, list, tuple)):
            return mask_secret_values(data=item)
        identity = id(item)
        if identity in active:
            return "<circular>"
        active.add(identity)
        try:
            if isinstance(item, dict):
                return {mask_secret_values(data=key): mask(value) for key, value in item.items()}
            return [mask(value) for value in item]
        finally:
            active.remove(identity)

    return mask(value)


class _PhoenixCallback(OpenInferenceTracer):
    def __init__(self, tracer: OITracer, *, user_id: str | None, session_id: str | None):
        super().__init__(tracer, separate_trace_from_runtime_context=False)
        self._user_id = user_id
        self._session_id = session_id

    def _prepare_run(self, run: Any) -> None:
        run.inputs = _callback_data(run.inputs)
        run.outputs = _callback_data(run.outputs)
        run.extra = mask_secret_values(data=run.extra or {})
        metadata = dict(run.extra.get("metadata") or {})
        # LangChain copies the internal checkpoint key to its run metadata.
        for key in ("thread_id", "conversation_id", "session_id"):
            metadata.pop(key, None)
        if self._session_id is not None:
            metadata.update(session_id=self._session_id, thread_id=self._session_id)
        if self._user_id is not None:
            metadata["user_id"] = self._user_id
        run.extra["metadata"] = metadata

    def _start_trace(self, run: Any) -> None:
        self._prepare_run(run)
        with using_attributes(user_id=self._user_id, session_id=self._session_id):
            super()._start_trace(run)

    def _end_trace(self, run: Any) -> None:
        self._prepare_run(run)
        with using_attributes(user_id=self._user_id, session_id=self._session_id):
            super()._end_trace(run)


def _native_prompt_data(prompt: PromptVersion) -> dict[str, Any]:
    """Read the native format through the tested SDK compatibility boundary."""
    # The native public format() method does not expose the stored template format.
    data = prompt._dumps()
    if data.get("template_format") not in ("F_STRING", "MUSTACHE", "NONE"):
        raise NotImplementedError("Phoenix returned an unsupported prompt format")
    if data["template"].get("type") != "chat":
        raise NotImplementedError("Phoenix returned an unsupported prompt type")
    return data


class PhoenixObservability(BaseObservabilityPlatform):
    """Use Phoenix without global tracing or LangChain patches.

    Each instance owns its default HTTP client and tracer provider.
    The caller owns any injected client or provider.
    """

    def __init__(
        self,
        remote_first: bool = False,
        *,
        endpoint: str | None = None,
        project_name: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        client: Any = None,
        http_client: httpx.Client | None = None,
        tracer_provider: Any = None,
        trace_config: TraceConfig | None = None,
    ) -> None:
        super().__init__(remote_first)
        self.endpoint = (endpoint or settings.PHOENIX_ENDPOINT).rstrip("/")
        self.project_name = project_name or settings.PHOENIX_PROJECT_NAME
        self.timeout = timeout if timeout is not None else settings.PHOENIX_TIMEOUT
        if not isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Phoenix timeout must be positive")
        schedule_delay_millis = int(os.environ.get("OTEL_BSP_SCHEDULE_DELAY", "1000"))
        if schedule_delay_millis <= 0:
            raise ValueError("OTEL_BSP_SCHEDULE_DELAY must be positive")
        self._feedback_retry_window = self.timeout + schedule_delay_millis / 1000
        if api_key is None and settings.PHOENIX_API_KEY is not None:
            api_key = settings.PHOENIX_API_KEY.get_secret_value()
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self._owns_http_client = client is None and http_client is None
        self._http_client = http_client
        self._owns_provider = tracer_provider is None
        with ExitStack() as cleanup:
            if client is None:
                if self._http_client is None:
                    self._http_client = httpx.Client(
                        base_url=self.endpoint + "/", headers=headers, timeout=self.timeout
                    )
                    cleanup.callback(self._http_client.close)
                client = Client(http_client=self._http_client)
            self._client = client
            if tracer_provider is None:
                tracer_provider = TracerProvider(
                    resource=Resource.create({"openinference.project.name": self.project_name}),
                    id_generator=_RunIdGenerator(),
                    shutdown_on_exit=False,
                )
                cleanup.callback(tracer_provider.shutdown)
                with ExitStack() as exporter_cleanup:
                    exporter = OTLPSpanExporter(
                        endpoint=self.endpoint + "/v1/traces",
                        headers={"content-type": "application/x-protobuf", **headers},
                        timeout=self.timeout,
                    )
                    exporter_cleanup.callback(exporter.shutdown)
                    tracer_provider.add_span_processor(
                        BatchSpanProcessor(exporter, schedule_delay_millis=schedule_delay_millis)
                    )
                    exporter_cleanup.pop_all()
            self._provider = tracer_provider
            self._tracer = OITracer(
                tracer_provider.get_tracer("langgraph_agent_toolkit.phoenix"), config=trace_config or TraceConfig()
            )
            cleanup.pop_all()
        self._closed = False
        self._shutdown_lock = Lock()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Phoenix observability is closed")

    def get_callback_handler(self, **kwargs: Any) -> Any:
        """Create one native OpenInference callback for each run."""
        self._check_open()
        return _PhoenixCallback(self._tracer, user_id=kwargs.get("user_id"), session_id=kwargs.get("session_id"))

    @contextmanager
    def trace_context(self, run_id: str, **kwargs: Any):
        """Start a root trace with the request UUID as its trace ID."""
        self._check_open()
        identity = UUID(str(run_id)).int
        if not identity:
            raise ValueError("Phoenix run_id must not be the zero UUID")
        token = _trace_id.set(identity)
        try:
            with using_attributes(user_id=kwargs.get("user_id"), session_id=kwargs.get("session_id")):
                with self._tracer.start_as_current_span(
                    kwargs.get("agent_name") or "agent",
                    context=Context(),
                    openinference_span_kind="agent",
                ) as span:
                    self.update_trace(span, input=kwargs.get("input"))
                    yield span
        finally:
            _trace_id.reset(token)

    def update_trace(self, trace: Any, **attributes: Any) -> None:
        """Record root input and output through OpenInference masking."""
        for key in ("input", "output"):
            if key in attributes and attributes[key] is not None:
                value = mask_secret_values(data=attributes[key])
                serialized = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
                trace.set_attribute(f"{key}.value", serialized)
                trace.set_attribute(f"{key}.mime_type", "text/plain" if isinstance(value, str) else "application/json")

    def record_feedback(self, run_id: str, key: str, score: float, **kwargs: Any) -> None:
        """Save human feedback as a native trace annotation."""
        self._check_open()
        metadata = dict(kwargs.get("metadata") or {})
        if kwargs.get("user_id") is not None:
            metadata["user_id"] = kwargs["user_id"]
        # Phoenix can accept the trace before its background task saves it.
        retry = Retrying(
            retry=retry_if_exception(
                lambda exc: isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404
            ),
            stop=stop_before_delay(self._feedback_retry_window),
            wait=wait_exponential(multiplier=0.05, max=0.5),
            reraise=True,
        )
        retry(
            self._client.traces.add_trace_annotation,
            trace_id=UUID(str(run_id)).hex,
            annotation_name=key,
            annotator_kind="HUMAN",
            score=score,
            label=kwargs.get("label"),
            explanation=kwargs.get("comment", kwargs.get("explanation")),
            metadata=mask_secret_values(data=metadata) or None,
            identifier=kwargs.get("feedback_id"),
            sync=True,
        )

    def push_prompt(
        self,
        name: str,
        prompt_template: PromptTemplateType,
        metadata: dict[str, Any] | None = None,
        force_create_new_version: bool = True,
    ) -> None:
        """Create a native chat prompt version and optional version tags."""
        self._check_open()
        if self.remote_first or not force_create_new_version:
            try:
                self._client.prompts.get(prompt_identifier=name)
            except ValueError as exc:
                cause = exc.__cause__ or exc.__context__
                if not isinstance(cause, httpx.HTTPStatusError) or cause.response.status_code != 404:
                    raise
            else:
                return
        options = dict(metadata or {})
        model_name = options.pop("model_name", None)
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Phoenix prompt metadata must include model_name")
        template_format = options.pop("template_format", "F_STRING")
        if template_format not in ("F_STRING", "MUSTACHE", "NONE"):
            raise NotImplementedError("Phoenix supports F_STRING, MUSTACHE, and NONE prompt formats")
        model_provider = options.pop("model_provider", "OPENAI")
        description = options.pop("description", None)
        labels = options.pop("labels", [])
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise ValueError("Phoenix prompt labels must be a list of strings")
        messages = (
            [{"role": "user", "content": prompt_template}]
            if isinstance(prompt_template, str)
            else [
                {
                    "role": {"human": "user", "ai": "assistant"}.get(item["role"], item["role"]),
                    "content": item["content"],
                }
                for item in prompt_template
            ]
        )
        if any(item["role"] not in ("system", "user", "assistant") for item in messages):
            raise NotImplementedError("Phoenix prompt placeholders are not supported")
        version = self._client.prompts.create(
            name=name,
            version=PromptVersion(
                messages,
                model_name=model_name,
                model_provider=model_provider,
                description=description,
                template_format=template_format,
            ),
            prompt_description=description,
            prompt_metadata=mask_secret_values(data=options) or None,
        )
        for label in labels:
            self._client.prompts.tags.create(prompt_version_id=version.id, name=label)

    def pull_prompt(
        self,
        name: str,
        template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
        **kwargs: Any,
    ) -> PromptReturnType:
        """Get the latest, tagged, or explicitly identified prompt version."""
        self._check_open()
        if template_format == "jinja2":
            raise NotImplementedError("Phoenix does not support jinja2 prompt templates")
        version_id = kwargs.get("prompt_version_id", kwargs.get("version"))
        label = kwargs.get("label", kwargs.get("tag"))
        if version_id is not None and label is not None:
            raise ValueError("Select a Phoenix prompt version or label, not both")
        if version_id is not None and not isinstance(version_id, str):
            raise ValueError("Phoenix prompt version must be its native string ID")
        native = self._client.prompts.get(
            **({"prompt_version_id": version_id} if version_id else {"prompt_identifier": name, "tag": label})
        )
        data = _native_prompt_data(native)
        messages = data["template"]["messages"]
        if any(not isinstance(item.get("content"), str) for item in messages):
            raise NotImplementedError("Phoenix multimodal prompt templates cannot be converted to ChatPromptTemplate")
        stored_format = data["template_format"]
        if stored_format == "NONE":
            escaped = [{**item, "content": item["content"].replace("{", "{{").replace("}", "}}")} for item in messages]
            prompt = self._process_prompt_object(escaped, template_format="f-string")
        else:
            expected_format = {"F_STRING": "f-string", "MUSTACHE": "mustache"}[stored_format]
            if template_format != expected_format:
                raise ValueError(f"Phoenix prompt uses {expected_format}; set template_format={expected_format!r}")
            prompt = self._process_prompt_object(messages, template_format=template_format)
        return (prompt, native) if kwargs.get("return_with_prompt_object") else prompt

    def delete_prompt(self, name: str) -> None:
        """Delete a prompt through the native Phoenix API."""
        self._check_open()
        self._client.prompts.delete(prompt_identifier=name)

    def before_shutdown(self) -> None:
        """Flush and close the resources that this instance owns."""
        with self._shutdown_lock:
            if self._closed:
                return
            self._closed = True
        try:
            if self._owns_provider:
                self._provider.shutdown()
        finally:
            if self._owns_http_client and self._http_client is not None:
                self._http_client.close()
