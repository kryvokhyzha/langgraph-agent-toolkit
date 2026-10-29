"""Test the installed Phoenix and OpenInference SDKs at their transport boundaries.

The transports record real SDK requests. These tests do not emulate a server.
"""

import asyncio
import gzip
import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import requests
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from pydantic import SecretStr


pytest.importorskip("phoenix.client")
pytest.importorskip("openinference.instrumentation.langchain")

from opentelemetry import trace
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from langgraph_agent_toolkit.core.observability import phoenix as adapter
from langgraph_agent_toolkit.core.prompts.chat_prompt_template import ObservabilityChatPromptTemplate


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _value(value):
    kind = value.WhichOneof("value")
    if kind == "array_value":
        return [_value(item) for item in value.array_value.values]
    if kind == "kvlist_value":
        return {item.key: _value(item.value) for item in value.kvlist_value.values}
    return getattr(value, kind)


def _attributes(span):
    return {item.key: _value(item.value) for item in span.attributes}


@pytest.fixture
def wire_sdk(monkeypatch):
    endpoint = "http://phoenix.invalid/space"
    rest = []
    otlp = []
    versions = []
    tags = {}
    annotation_attempts = []

    def respond(request):
        assert request.url.host == "phoenix.invalid"
        rest.append(request)
        assert request.headers.get("authorization") == "Bearer explicit-test-key"
        path = request.url.path.removeprefix("/space/")
        if path == "arize_phoenix_version":
            return httpx.Response(200, text="20.26.1")
        if path == "v1/prompts" and request.method == "POST":
            body = json.loads(request.content)
            version = {**body["version"], "id": f"native-version-{len(versions) + 1}"}
            versions.append(version)
            return httpx.Response(200, json={"data": version})
        if path.startswith("v1/prompt_versions/") and path.endswith("/tags"):
            name = json.loads(request.content)["name"]
            tags[name] = path.split("/")[2]
            return httpx.Response(200, json={})
        if path.startswith("v1/prompt_versions/"):
            identifier = path.split("/")[2]
            return httpx.Response(200, json={"data": next(item for item in versions if item["id"] == identifier)})
        if request.method == "DELETE":
            versions.clear()
            return httpx.Response(204)
        if path.startswith("v1/prompts/"):
            if not versions:
                return httpx.Response(404)
            identifier = path.rsplit("/", 1)[1]
            version_id = tags.get(identifier, versions[-1]["id"])
            return httpx.Response(200, json={"data": next(item for item in versions if item["id"] == version_id)})
        if path == "v1/trace_annotations":
            annotation_attempts.append(request)
            if len(annotation_attempts) == 1:
                return httpx.Response(404, json={"detail": "Trace not found"})
            return httpx.Response(200, json={"data": [{"id": "annotation-1"}]})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    real_client = httpx.Client
    monkeypatch.setattr(
        adapter.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs)
    )

    def export(session, request, **kwargs):
        assert request.url == endpoint + "/v1/traces"
        assert request.headers.get("authorization") == "Bearer explicit-test-key"
        body = request.body
        if request.headers.get("content-encoding") == "gzip":
            body = gzip.decompress(body)
        payload = ExportTraceServiceRequest.FromString(body)
        otlp.append(payload)
        response = requests.Response()
        response.status_code = 200
        response._content = b""
        response.request = request
        return response

    monkeypatch.setattr(requests.Session, "send", export)
    from langchain_core.callbacks import BaseCallbackManager

    original_manager = BaseCallbackManager.__init__
    original_provider = trace.get_tracer_provider()
    platform = adapter.PhoenixObservability(endpoint=endpoint, api_key="explicit-test-key", project_name="sdk-test")
    fixture = SimpleNamespace(
        platform=platform,
        rest=rest,
        otlp=otlp,
        versions=versions,
        annotations=annotation_attempts,
        original_manager=original_manager,
        original_provider=original_provider,
        spans=lambda: [
            span
            for payload in otlp
            for resource in payload.resource_spans
            for scope in resource.scope_spans
            for span in scope.spans
        ],
    )
    yield fixture
    platform.before_shutdown()


@pytest.mark.asyncio
async def test_native_graph_has_model_tool_io_usage_and_public_identity(wire_sdk):
    @tool
    def multiply(value: int) -> str:
        """Multiply the value by two."""
        return str(value * 2)

    graph = create_agent(
        model=ScriptedModel(
            responses=[
                AIMessage(content="", tool_calls=[{"id": "call-1", "name": "multiply", "args": {"value": 3}}]),
                AIMessage(
                    content="Result: 6", usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}
                ),
            ]
        ),
        tools=[multiply],
    )
    platform = wire_sdk.platform
    run_id = uuid4()
    callback = platform.get_callback_handler(run_id=str(run_id), user_id="user-1", session_id="public-thread")
    with platform.trace_context(
        str(run_id), user_id="user-1", session_id="public-thread", input={"request": "compute"}
    ) as root:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Multiply three", additional_kwargs={"key": SecretStr("private-key")})]},
            config={
                "run_id": run_id,
                "callbacks": [callback],
                "configurable": {"thread_id": "internal-storage-key"},
                "metadata": {"key": SecretStr("private-key")},
            },
        )
        platform.update_trace(root, output=result["messages"][-1].content)
    assert platform._provider.force_flush(timeout_millis=2000)
    spans = wire_sdk.spans()
    assert spans and {span.trace_id.hex() for span in spans} == {run_id.hex}
    root_spans = [span for span in spans if not span.parent_span_id]
    assert len(root_spans) == 1
    assert _attributes(root_spans[0])["output.value"] == "Result: 6"
    span_ids = {span.span_id for span in spans}
    assert all(span.parent_span_id in span_ids for span in spans if span.parent_span_id)
    attrs = [_attributes(span) for span in spans]
    assert all(item["session.id"] == "public-thread" and item["user.id"] == "user-1" for item in attrs)
    assert all("internal-storage-key" not in json.dumps(item) for item in attrs)
    assert any(item.get("openinference.span.kind") == "TOOL" and "6" in item.get("output.value", "") for item in attrs)
    models = [item for item in attrs if item.get("openinference.span.kind") == "LLM"]
    assert len(models) == 2
    assert any(item.get("llm.token_count.total") == 16 for item in models)
    assert any(item.get("llm.input_messages.0.message.content") == "Multiply three" for item in models)
    assert any(item.get("llm.output_messages.0.message.content") == "Result: 6" for item in models)
    assert all(b"private-key" not in payload.SerializeToString() for payload in wire_sdk.otlp)


@pytest.mark.asyncio
async def test_concurrent_runs_keep_independent_native_callback_trees(wire_sdk):
    platform = wire_sdk.platform
    run_ids = [uuid4() for _ in range(6)]

    async def invoke(index):
        run_id = run_ids[index]
        callback = platform.get_callback_handler(user_id=f"user-{index}", session_id=f"thread-{index}")
        graph = RunnableLambda(lambda value: {"answer": value, "key": SecretStr("private-tool-key")})
        with platform.trace_context(str(run_id), user_id=f"user-{index}", session_id=f"thread-{index}") as root:
            result = await graph.ainvoke({"index": index}, config={"callbacks": [callback], "run_id": run_id})
            assert isinstance(result["key"], SecretStr)
            platform.update_trace(root, output=result)

    await asyncio.gather(*(invoke(index) for index in range(6)))
    assert platform._provider.force_flush(timeout_millis=2000)
    spans = wire_sdk.spans()
    assert len(spans) == 12
    for index, run_id in enumerate(run_ids):
        related = [span for span in spans if span.trace_id.hex() == run_id.hex]
        assert len(related) == 2
        assert sum(not span.parent_span_id for span in related) == 1
        for span in related:
            attrs = _attributes(span)
            assert attrs["user.id"] == f"user-{index}"
            assert attrs["session.id"] == f"thread-{index}"
            assert "private-tool-key" not in json.dumps(attrs)
    assert trace.get_current_span().get_span_context().is_valid is False


def test_native_prompt_revisions_tags_format_and_wrapper_ids(wire_sdk):
    platform = wire_sdk.platform
    options = {"model_name": "example-model", "labels": ["production"]}
    platform.push_prompt("welcome", "Hello {name}", options)
    first = wire_sdk.versions[0]["id"]
    platform.push_prompt("welcome", "Updated {name}", {"model_name": "example-model"})
    assert len(wire_sdk.versions) == 2
    assert (
        platform.pull_prompt("welcome", label="production").format_messages(name="reader")[0].content == "Hello reader"
    )
    assert platform.pull_prompt("welcome").format_messages(name="reader")[0].content == "Updated reader"
    wrapper = ObservabilityChatPromptTemplate(
        prompt_name="welcome", prompt_version=first, observability_platform=platform
    )
    assert wrapper.prompt_version == first
    assert wrapper.format_messages(name="reader")[0].content == "Hello reader"
    platform.push_prompt("welcome", "ignored", force_create_new_version=False)
    assert len(wire_sdk.versions) == 2
    platform.delete_prompt("welcome")
    with pytest.raises(ValueError, match="not found"):
        platform.pull_prompt("welcome")


def test_feedback_retries_pending_ingestion_using_native_sdk(wire_sdk):
    platform = wire_sdk.platform
    run_id = uuid4()
    with platform.trace_context(str(run_id)):
        pass
    platform.record_feedback(str(run_id), "helpful", 1, comment="Useful", user_id="user-1")
    assert len(wire_sdk.annotations) == 2
    request = wire_sdk.annotations[-1]
    assert request.url.params["sync"] == "true"
    annotation = json.loads(request.content)["data"][0]
    assert annotation["trace_id"] == run_id.hex
    assert annotation["name"] == "helpful"
    assert annotation["annotator_kind"] == "HUMAN"
    assert annotation["result"] == {"score": 1, "explanation": "Useful"}


def test_shutdown_flushes_native_batch_and_preserves_global_tracing(wire_sdk):
    from langchain_core.callbacks import BaseCallbackManager

    platform = wire_sdk.platform
    with platform.trace_context(str(uuid4())):
        pass
    assert not wire_sdk.otlp
    platform.before_shutdown()
    assert len(wire_sdk.spans()) == 1
    assert platform._http_client.is_closed
    assert BaseCallbackManager.__init__ is wire_sdk.original_manager
    assert trace.get_tracer_provider() is wire_sdk.original_provider


def test_explicit_configuration_does_not_reuse_environment_credentials(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=other-service-key")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "authorization=other-trace-key")
    monkeypatch.setenv("PHOENIX_API_KEY", "other-phoenix-key")
    exported = []

    def export(session, request, **kwargs):
        exported.append(request)
        assert "authorization" not in request.headers
        response = requests.Response()
        response.status_code = 200
        response._content = b""
        return response

    monkeypatch.setattr(requests.Session, "send", export)
    platform = adapter.PhoenixObservability(endpoint="http://phoenix.invalid", api_key="")
    try:
        assert "authorization" not in platform._http_client.headers
        with platform.trace_context(str(uuid4())):
            pass
        assert platform._provider.force_flush(timeout_millis=2000)
        assert len(exported) == 1
    finally:
        platform.before_shutdown()


@pytest.mark.asyncio
async def test_native_privacy_flags_hide_root_and_callback_payloads(wire_sdk):
    platform = adapter.PhoenixObservability(
        endpoint="http://phoenix.invalid/space",
        api_key="explicit-test-key",
        trace_config=adapter.TraceConfig(hide_inputs=True, hide_outputs=True),
    )
    try:
        run_id = uuid4()
        callback = platform.get_callback_handler()
        graph = RunnableLambda(lambda value: {"answer": "private-output-value"})
        with platform.trace_context(str(run_id), input="private-input-value") as root:
            result = await graph.ainvoke({"input": "private-input-value"}, config={"callbacks": [callback]})
            platform.update_trace(root, output=result)
        assert platform._provider.force_flush(timeout_millis=2000)
        spans = [span for span in wire_sdk.spans() if span.trace_id.hex() == run_id.hex]
        assert len(spans) == 2
        for span in spans:
            attributes = _attributes(span)
            assert "private-input-value" not in json.dumps(attributes)
            assert "private-output-value" not in json.dumps(attributes)
            assert attributes["input.value"] == "__REDACTED__"
            assert attributes["output.value"] == "__REDACTED__"
    finally:
        platform.before_shutdown()


@pytest.mark.asyncio
async def test_collector_outage_does_not_fail_agent_execution(monkeypatch):
    attempts = []

    def unavailable(session, request, **kwargs):
        attempts.append(request)
        raise requests.ConnectionError("collector unavailable")

    monkeypatch.setattr(requests.Session, "send", unavailable)
    platform = adapter.PhoenixObservability(endpoint="http://phoenix.invalid", timeout=0.05, api_key="")
    graph = RunnableLambda(lambda value: {"answer": value["question"]})
    try:
        with platform.trace_context(str(uuid4())):
            result = await graph.ainvoke({"question": "hello"}, config={"callbacks": [platform.get_callback_handler()]})
        assert result == {"answer": "hello"}
    finally:
        platform.before_shutdown()
    assert attempts
