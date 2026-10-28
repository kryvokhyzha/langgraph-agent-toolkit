"""Test the Phoenix adapter contract and resource ownership."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import SecretStr


pytest.importorskip("phoenix.client")
pytest.importorskip("openinference.instrumentation.langchain")

from langgraph_agent_toolkit.core.observability import phoenix as adapter


@pytest.fixture
def backend():
    provider = MagicMock()
    provider.force_flush.return_value = True
    client = MagicMock()
    platform = adapter.PhoenixObservability(client=client, tracer_provider=provider)
    yield platform
    platform.before_shutdown()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_timeout_must_be_positive_and_finite(timeout):
    with pytest.raises(ValueError, match="timeout"):
        adapter.PhoenixObservability(timeout=timeout, client=MagicMock(), tracer_provider=MagicMock())


@pytest.mark.parametrize("schedule", ["0", "-1", "invalid"])
def test_batch_delay_must_be_positive(monkeypatch, schedule):
    monkeypatch.setenv("OTEL_BSP_SCHEDULE_DELAY", schedule)
    with pytest.raises(ValueError):
        adapter.PhoenixObservability(client=MagicMock(), tracer_provider=MagicMock())


def test_batch_delay_controls_feedback_retry_window(monkeypatch):
    monkeypatch.setenv("OTEL_BSP_SCHEDULE_DELAY", "2000")
    platform = adapter.PhoenixObservability(timeout=3, client=MagicMock(), tracer_provider=MagicMock())
    assert platform._feedback_retry_window == 5
    platform.before_shutdown()


def test_callback_masks_secrets_without_changing_the_agent_input(backend):
    @dataclass
    class ToolResult:
        credential: SecretStr

    secret = SecretStr("private-value")
    inputs = {"messages": [[HumanMessage(content="hello", additional_kwargs={"key": secret})]], "secret": secret}
    outputs = {"output": ToolResult(secret)}
    callback = backend.get_callback_handler(user_id="user-1", session_id="public-thread")
    run = SimpleNamespace(
        inputs=inputs,
        outputs=outputs,
        extra={"metadata": {"thread_id": "internal-owner-key", "conversation_id": "internal-conversation"}},
    )
    callback._prepare_run(run)
    assert inputs["secret"] is secret
    assert isinstance(inputs["messages"][0][0], HumanMessage)
    assert run.inputs["secret"] == "<secret>"
    assert run.outputs["output"]["credential"] == "<secret>"
    assert run.extra["metadata"] == {"user_id": "user-1", "session_id": "public-thread", "thread_id": "public-thread"}
    assert "private-value" not in str(run.inputs)
    assert "private-value" not in str(run.outputs)


def test_callback_masks_circular_containers():
    data = {"secret": SecretStr("private-value")}
    data["self"] = data
    data["list"] = [data]
    masked = adapter._callback_data(data)
    assert masked == {"secret": "<secret>", "self": "<circular>", "list": ["<circular>"]}


def test_callbacks_do_not_reuse_identity(backend):
    first = backend.get_callback_handler(user_id="user-1", session_id="thread-1")
    second = backend.get_callback_handler()
    run = SimpleNamespace(inputs={}, outputs={}, extra={"metadata": {"thread_id": "internal-key"}})
    second._prepare_run(run)
    assert first is not second
    assert second._user_id is None
    assert second._session_id is None
    assert run.extra["metadata"] == {}


def test_feedback_uses_native_trace_annotation_without_blocking_export(backend):
    run_id = uuid4()
    events = []
    backend._provider.force_flush.side_effect = lambda **kwargs: events.append("flush") or True
    backend._client.traces.add_trace_annotation.side_effect = lambda **kwargs: events.append("annotation")
    backend.record_feedback(str(run_id), "helpful", 1.0, user_id="user-1", comment="Useful")
    assert events == ["annotation"]
    backend._provider.force_flush.assert_not_called()
    kwargs = backend._client.traces.add_trace_annotation.call_args.kwargs
    assert kwargs["trace_id"] == run_id.hex
    assert kwargs["annotator_kind"] == "HUMAN"
    assert kwargs["explanation"] == "Useful"
    assert kwargs["metadata"] == {"user_id": "user-1"}
    assert kwargs["sync"] is True


def test_owned_resources_close_once_even_if_provider_shutdown_fails(monkeypatch):
    http = MagicMock()
    provider = MagicMock()
    provider.shutdown.side_effect = RuntimeError("export failed")
    monkeypatch.setattr(adapter.httpx, "Client", lambda **kwargs: http)
    monkeypatch.setattr(adapter, "Client", lambda **kwargs: MagicMock())
    monkeypatch.setattr(adapter, "TracerProvider", lambda **kwargs: provider)
    platform = adapter.PhoenixObservability()
    with pytest.raises(RuntimeError, match="export failed"):
        platform.before_shutdown()
    platform.before_shutdown()
    provider.shutdown.assert_called_once()
    http.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        platform.get_callback_handler()


def test_injected_resources_remain_open(backend):
    backend.before_shutdown()
    backend._provider.shutdown.assert_not_called()
    backend._client.close.assert_not_called()


@pytest.mark.parametrize("force,remote_first", [(False, False), (True, True)])
def test_existing_remote_prompt_is_not_overwritten(backend, force, remote_first):
    backend._remote_first = remote_first
    backend.push_prompt("welcome", "ignored", force_create_new_version=force)
    backend._client.prompts.create.assert_not_called()


def test_prompt_lookup_does_not_hide_authorization_errors(backend):
    request = httpx.Request("GET", "http://phoenix.invalid/v1/prompts/welcome/latest")
    error = httpx.HTTPStatusError("Unauthorized", request=request, response=httpx.Response(401, request=request))
    backend._client.prompts.get.side_effect = error
    with pytest.raises(httpx.HTTPStatusError):
        backend.push_prompt("welcome", "Hello", {"model_name": "example-model"}, force_create_new_version=False)
    backend._client.prompts.create.assert_not_called()


def test_push_creates_native_chat_version_and_tags(backend):
    backend._client.prompts.create.return_value.id = "native-version-id"
    metadata = {"model_name": "example-model", "labels": ["production"], "key": SecretStr("private-value")}
    backend.push_prompt("welcome", "Hello {name}", metadata)
    options = backend._client.prompts.create.call_args.kwargs
    formatted = options["version"].format(variables={"name": "reader"}, sdk="openai")
    assert formatted.messages == [{"role": "user", "content": "Hello reader"}]
    assert options["prompt_metadata"] == {"key": "<secret>"}
    assert metadata["labels"] == ["production"]
    backend._client.prompts.tags.create.assert_called_once_with(
        prompt_version_id="native-version-id", name="production"
    )


@pytest.mark.parametrize(
    "metadata,error",
    [
        ({}, ValueError),
        ({"model_name": "model", "labels": "production"}, ValueError),
        ({"model_name": "model", "template_format": "JINJA2"}, NotImplementedError),
    ],
)
def test_invalid_prompt_contract_is_explicit(backend, metadata, error):
    with pytest.raises(error):
        backend.push_prompt("welcome", "Hello", metadata)
    backend._client.prompts.create.assert_not_called()


def test_placeholder_templates_are_not_silently_changed(backend):
    with pytest.raises(NotImplementedError, match="placeholders"):
        backend.push_prompt("welcome", [{"role": "placeholder", "content": "messages"}], {"model_name": "model"})


def test_pull_native_version_and_return_object(backend):
    native = adapter.PromptVersion(
        [{"role": "assistant", "content": "Hello {name}"}], model_name="model", template_format="F_STRING"
    )
    backend._client.prompts.get.return_value = native
    prompt, returned = backend.pull_prompt("welcome", version="native-version-id", return_with_prompt_object=True)
    assert returned is native
    assert prompt.format_messages(name="reader") == [AIMessage(content="Hello reader")]
    backend._client.prompts.get.assert_called_once_with(prompt_version_id="native-version-id")


@pytest.mark.parametrize("kwargs", [{"version": 1}, {"version": "id", "label": "production"}])
def test_prompt_version_selection_is_unambiguous(backend, kwargs):
    with pytest.raises(ValueError):
        backend.pull_prompt("welcome", **kwargs)


def test_jinja2_pull_is_explicitly_unsupported(backend):
    with pytest.raises(NotImplementedError, match="jinja2"):
        backend.pull_prompt("welcome", template_format="jinja2")


def test_delete_uses_native_api(backend):
    backend.delete_prompt("welcome")
    backend._client.prompts.delete.assert_called_once_with(prompt_identifier="welcome")


def test_none_prompt_retains_literal_braces(backend):
    content = 'Return {"answer": "{{literal}}"}'
    backend._client.prompts.get.return_value = adapter.PromptVersion(
        [{"role": "user", "content": content}], model_name="model", template_format="NONE"
    )
    prompt = backend.pull_prompt("literal")
    assert prompt.input_variables == []
    assert prompt.format_messages()[0].content == content


def test_prompt_format_mismatch_is_not_silent(backend):
    backend._client.prompts.get.return_value = adapter.PromptVersion(
        [{"role": "user", "content": "Hello {{name}}"}], model_name="model", template_format="MUSTACHE"
    )
    with pytest.raises(ValueError, match="mustache"):
        backend.pull_prompt("welcome")


@pytest.mark.parametrize("status,retries", [(404, 2), (401, 1), (422, 1)])
def test_feedback_retries_only_pending_trace_ingestion(backend, status, retries):
    request = httpx.Request("POST", "http://phoenix.invalid/v1/trace_annotations")
    error = httpx.HTTPStatusError("failed", request=request, response=httpx.Response(status, request=request))
    backend._client.traces.add_trace_annotation.side_effect = [error, None]
    if status == 404:
        backend.record_feedback(str(uuid4()), "helpful", 1)
    else:
        with pytest.raises(httpx.HTTPStatusError):
            backend.record_feedback(str(uuid4()), "helpful", 1)
    assert backend._client.traces.add_trace_annotation.call_count == retries


def test_feedback_retry_window_does_not_wait_after_its_deadline(backend):
    backend._feedback_retry_window = 0.001
    request = httpx.Request("POST", "http://phoenix.invalid/v1/trace_annotations")
    error = httpx.HTTPStatusError("pending", request=request, response=httpx.Response(404, request=request))
    backend._client.traces.add_trace_annotation.side_effect = error
    with pytest.raises(httpx.HTTPStatusError):
        backend.record_feedback(str(uuid4()), "helpful", 1)
    backend._client.traces.add_trace_annotation.assert_called_once()


def test_corrupt_prompt_lookup_does_not_create_a_revision(backend):
    requests = []

    def corrupt_response(request):
        requests.append(request)
        return httpx.Response(200, content=b"invalid json")

    with httpx.Client(base_url="http://phoenix.invalid", transport=httpx.MockTransport(corrupt_response)) as client:
        backend._client = adapter.Client(http_client=client)
        with pytest.raises(ValueError):
            backend.push_prompt("welcome", "Hello", {"model_name": "model"}, force_create_new_version=False)
    assert [request.method for request in requests] == ["GET"]


def test_invalid_native_batch_limits_close_partial_owned_resources(monkeypatch):
    monkeypatch.setenv("OTEL_BSP_MAX_QUEUE_SIZE", "1")
    monkeypatch.setenv("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "2")
    http = MagicMock()
    provider = MagicMock()
    exporter = MagicMock()
    monkeypatch.setattr(adapter.httpx, "Client", lambda **kwargs: http)
    monkeypatch.setattr(adapter, "Client", lambda **kwargs: MagicMock())
    monkeypatch.setattr(adapter, "TracerProvider", lambda **kwargs: provider)
    monkeypatch.setattr(adapter, "OTLPSpanExporter", lambda **kwargs: exporter)
    with pytest.raises(ValueError, match="max_export_batch_size"):
        adapter.PhoenixObservability()
    http.close.assert_called_once()
    provider.shutdown.assert_called_once()
    exporter.shutdown.assert_called_once()


def test_sdk_client_failure_closes_only_owned_http_client(monkeypatch):
    http = MagicMock()
    monkeypatch.setattr(adapter.httpx, "Client", lambda **kwargs: http)
    monkeypatch.setattr(adapter, "Client", MagicMock(side_effect=ValueError("client setup failed")))
    with pytest.raises(ValueError, match="client setup failed"):
        adapter.PhoenixObservability()
    http.close.assert_called_once()
    http.reset_mock()
    with pytest.raises(ValueError, match="client setup failed"):
        adapter.PhoenixObservability(http_client=http)
    http.close.assert_not_called()
