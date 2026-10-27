"""Test the installed SDK through its HTTP ingestion boundary.

These tests do not start or emulate a Langfuse server.
The transport records SDK requests and returns protocol responses.
"""

import base64
import gzip
import json
import os
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import requests
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from pydantic import SecretStr

from langgraph_agent_toolkit.core.observability import langfuse as adapter


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def get_num_tokens(self, text):
        return len(text.split())

    def get_num_tokens_from_messages(self, messages, tools=None):
        return sum(self.get_num_tokens(str(message.content)) for message in messages)


@pytest.fixture
def wire_sdk(monkeypatch, request):
    """Keep REST and OTLP requests in memory without replacing SDK exporters."""
    public_key = f"pk-test-{uuid4()}"
    secret_key = "sdk-contract-test"
    options = getattr(request, "param", {})
    configured_host = options.get("host", "http://langfuse.invalid")
    for name in tuple(os.environ):
        if name.startswith(("LANGFUSE_", "OTEL_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", public_key)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", secret_key)
    monkeypatch.setenv("LANGFUSE_HOST", configured_host)
    if "base_url" in options:
        monkeypatch.setenv("LANGFUSE_BASE_URL", options["base_url"])
    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")
    monkeypatch.setattr(
        adapter,
        "settings",
        type(adapter.settings)(
            **{
                **adapter.settings.model_dump(),
                "LANGFUSE_PUBLIC_KEY": SecretStr(public_key),
                "LANGFUSE_SECRET_KEY": SecretStr(secret_key),
                "LANGFUSE_HOST": configured_host,
            }
        ),
    )
    rest_requests = []
    otlp_requests = []
    prompts = {}

    def rest(request):
        assert request.url.host == "langfuse.invalid"
        rest_requests.append(request)
        if request.url.path.endswith("/scores"):
            scores = json.loads(request.content)
            assert isinstance(scores, list)
            return httpx.Response(202, json={"message": "Accepted"})
        if request.url.path.endswith("/ingestion"):
            batch = json.loads(request.content)["batch"]
            return httpx.Response(
                207, json={"successes": [{"id": event["id"], "status": 201} for event in batch], "errors": []}
            )
        if request.method == "GET":
            name = request.url.path.rsplit("/", 1)[1]
            if name not in prompts:
                return httpx.Response(404, json={"message": "Prompt not found"})
            return httpx.Response(200, json=prompts[name])
        assert request.method == "POST" and request.url.path.endswith("/prompts")
        body = json.loads(request.content)
        prompt = {
            **body,
            "type": "text" if isinstance(body["prompt"], str) else "chat",
            "version": prompts.get(body["name"], {}).get("version", 0) + 1,
            "config": {},
            "createdAt": "2026-01-01T00:00:00Z",
        }
        prompts[body["name"]] = prompt
        return httpx.Response(200, json=prompt)

    def otlp_send(session, request, **kwargs):
        expected_base = options.get("base_url", configured_host).rstrip("/")
        assert request.url == expected_base + "/api/public/otel/v1/traces"
        otlp_requests.append(request)
        response = requests.Response()
        response.status_code = 200
        response._content = b""
        response.request = request
        return response

    monkeypatch.setattr(requests.Session, "send", otlp_send)
    transport = httpx.Client(transport=httpx.MockTransport(rest))
    kwargs = {
        **adapter._configuration(),
        "httpx_client": transport,
        "flush_at": options.get("flush_at", 1),
        "flush_interval": options.get("flush_interval", 0.01),
    }
    if adapter._IS_NEW_LANGFUSE:
        from opentelemetry.sdk.trace import TracerProvider

        kwargs["tracer_provider"] = TracerProvider()
    client = adapter.Langfuse(**kwargs)
    monkeypatch.setattr(adapter, "_get_langfuse_client", lambda: client)
    observation = adapter.LangfuseObservability()
    yield SimpleNamespace(
        client=client,
        observation=observation,
        rest=rest_requests,
        otlp=otlp_requests,
        auth="Basic " + base64.b64encode(f"{public_key}:{secret_key}".encode()).decode(),
    )
    client.shutdown()
    transport.close()


def ingestion_events(sdk):
    return [
        event
        for request in sdk.rest
        if request.url.path.endswith("/ingestion")
        for event in json.loads(request.content)["batch"]
    ]


def exported_spans(sdk):
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    spans = []
    for request in sdk.otlp:
        assert request.headers["Authorization"] == sdk.auth
        assert request.headers["Content-Type"] == "application/x-protobuf"
        body = request.body
        if request.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        exported = ExportTraceServiceRequest.FromString(body)
        spans.extend(
            span for resource in exported.resource_spans for scope in resource.scope_spans for span in scope.spans
        )
    return spans


@pytest.mark.parametrize(
    "wire_sdk",
    [
        {"host": "http://langfuse.invalid"},
        {"host": "http://langfuse.invalid/proxy/"},
        {"base_url": "http://langfuse.invalid/"},
        {"base_url": "http://langfuse.invalid/proxy/"},
    ],
    indirect=True,
)
def test_validated_host_keeps_endpoint_paths_and_proxy_prefixes(wire_sdk):
    """Export through normalized settings without a redirect or lost URL prefix."""
    with wire_sdk.observation.trace_context(str(uuid4()), agent_name="endpoint-contract", input="Input"):
        pass
    wire_sdk.client.flush()

    if adapter._IS_NEW_LANGFUSE:
        assert any(span.name == "endpoint-contract" for span in exported_spans(wire_sdk))
        assert wire_sdk.otlp
    else:
        assert any(
            event["type"] == "trace-create" and event["body"].get("name") == "endpoint-contract"
            for event in ingestion_events(wire_sdk)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("builder", ["native", "deepagents"])
async def test_graph_tool_prompt_and_feedback_reach_sdk_http_boundary(wire_sdk, builder):
    """Keep trace links and results through graph execution and SDK serialization."""
    obs = wire_sdk.observation
    obs.push_prompt("arithmetic", "Add {{a}} and {{b}}", force_create_new_version=False)
    prompt, remote = obs.pull_prompt("arithmetic", return_with_prompt_object=True, template_format="jinja2")
    assert remote.version == 1
    obs.push_prompt("arithmetic", "Add {{a}} and {{b}}", force_create_new_version=False)
    writes = [
        request for request in wire_sdk.rest if request.method == "POST" and request.url.path.endswith("/prompts")
    ]
    assert len(writes) == 1

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    model = ScriptedModel(
        responses=[
            AIMessage(content="", tool_calls=[{"id": "sum-1", "name": "add", "args": {"a": 2, "b": 5}}]),
            AIMessage(content="The sum is 7."),
        ]
    )
    if builder == "native":
        graph = create_agent(model, [add])
    else:
        pytest.importorskip("deepagents")
        from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph

        graph = build_graph(model=model)
    run_id = str(uuid4())
    messages = prompt.invoke({"a": 2, "b": 5}).to_messages()
    callback = obs.get_callback_handler(run_id=run_id, user_id="sdk-user", session_id="sdk-thread")
    with obs.trace_context(
        run_id, agent_name="sdk-integration", user_id="sdk-user", session_id="sdk-thread", input=messages[0].content
    ) as trace:
        output = await graph.ainvoke({"messages": messages}, config={"callbacks": [callback]})
        obs.update_trace(trace, output=output["messages"][-1].content)
    assert float(next(message for message in output["messages"] if isinstance(message, ToolMessage)).content) == 7
    obs.record_feedback(run_id, "correctness", 1, comment="Verified sum")
    wire_sdk.client.flush()

    trace_id = run_id.replace("-", "") if adapter._IS_NEW_LANGFUSE else run_id
    events = ingestion_events(wire_sdk)
    if adapter._SDK_MAJOR >= 5:
        score = next(
            score
            for request in wire_sdk.rest
            if request.url.path.endswith("/scores")
            for score in json.loads(request.content)
        )
        assert not any(request.url.path.endswith("/ingestion") for request in wire_sdk.rest)
        assert not hasattr(wire_sdk.client.api, "ingestion")
        assert not hasattr(wire_sdk.client.api, "legacy")
    else:
        score = next(event["body"] for event in events if event["type"] == "score-create")
    assert (score["traceId"], score["name"], score["value"], score["comment"]) == (
        trace_id,
        "correctness",
        1,
        "Verified sum",
    )
    assert all(request.headers["authorization"] == wire_sdk.auth for request in wire_sdk.rest)

    if not adapter._IS_NEW_LANGFUSE:
        observations = {}
        for event in events:
            if event["type"].startswith(("span-", "generation-", "trace-")):
                body = event["body"]
                observations.setdefault(body["id"], {}).update(body)
        root = observations[trace_id]
        assert (root["userId"], root["sessionId"], root["output"]) == ("sdk-user", "sdk-thread", "The sum is 7.")
        tool_span = next(value for value in observations.values() if value.get("name") == "add")
        assert tool_span["traceId"] == trace_id
        assert tool_span["parentObservationId"] in observations
        assert tool_span["endTime"]
        assert "7" in json.dumps(tool_span["output"])
        return

    spans = exported_spans(wire_sdk)
    assert spans
    if adapter._SDK_MAJOR >= 5:
        assert all(request.headers["Content-Encoding"] == "gzip" for request in wire_sdk.otlp)
        assert all(request.headers["x-langfuse-ingestion-version"] == "4" for request in wire_sdk.otlp)
    assert all(span.trace_id.hex() == trace_id for span in spans)
    root = next(span for span in spans if span.name == "sdk-integration")
    attributes = {item.key: item.value.string_value for item in root.attributes}
    assert attributes["langfuse.observation.output"] == "The sum is 7."
    assert attributes["user.id"] == "sdk-user"
    assert attributes["session.id"] == "sdk-thread"
    tool_span = next(span for span in spans if span.name == "add")
    assert tool_span.parent_span_id in {span.span_id for span in spans}
    assert tool_span.end_time_unix_nano >= tool_span.start_time_unix_nano > 0
    tool_attributes = {item.key: item.value.string_value for item in tool_span.attributes}
    assert tool_attributes["langfuse.observation.type"] == "tool"
    assert "7" in tool_attributes["langfuse.observation.output"]


@pytest.mark.skipif(adapter._SDK_MAJOR < 5, reason="SDK v5 score batch flush contract")
@pytest.mark.parametrize("wire_sdk", [{"flush_at": 1000, "flush_interval": 3600}], indirect=True)
def test_shutdown_flushes_buffered_scores_once(wire_sdk):
    """Send the final feedback batch before the process stops."""
    run_id = str(uuid4())
    wire_sdk.observation.record_feedback(run_id, "accuracy", 1, comment="Final feedback")
    assert not wire_sdk.rest

    wire_sdk.observation.before_shutdown()
    wire_sdk.observation.before_shutdown()

    score_requests = [request for request in wire_sdk.rest if request.url.path.endswith("/scores")]
    assert len(score_requests) == 1
    scores = json.loads(score_requests[0].content)
    assert len(scores) == 1
    assert scores[0]["traceId"] == run_id.replace("-", "")
    assert scores[0]["comment"] == "Final feedback"
    assert scores[0]["name"] == "accuracy"


@pytest.mark.skipif(adapter._SDK_MAJOR < 4, reason="Current SDK serializer contract")
def test_root_input_masks_secrets_and_sanitizes_nonfinite_model_values(wire_sdk):
    """Keep sensitive and invalid values out of exported root observations."""
    from pydantic import BaseModel

    class Input(BaseModel):
        secret: SecretStr
        number: float

    secret = "synthetic-sensitive-value"
    value = Input(secret=SecretStr(secret), number=float("nan"))
    with wire_sdk.observation.trace_context(str(uuid4()), agent_name="safe-input", input={"payload": value}):
        pass
    wire_sdk.client.flush()

    root = next(span for span in exported_spans(wire_sdk) if span.name == "safe-input")
    serialized = next(item.value.string_value for item in root.attributes if item.key == "langfuse.observation.input")
    assert secret not in serialized

    def reject_nonfinite(value):
        raise AssertionError(f"The export contains a non-JSON number: {value}")

    payload = json.loads(serialized, parse_constant=reject_nonfinite)["payload"]
    assert isinstance(payload["secret"], str)
    assert payload["number"] == "NaN"


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("container", ["mapping", "command"])
@pytest.mark.asyncio
async def test_typed_secrets_stay_masked_in_root_and_callback_exports(wire_sdk, legacy, binary, container):
    """Mask Pydantic secrets in root and child observations for each SDK."""
    if legacy:
        from pydantic.v1 import BaseModel, SecretBytes, SecretStr
    else:
        from pydantic import BaseModel, SecretBytes, SecretStr
    from langgraph.types import Command

    secret_type = SecretBytes if binary else SecretStr

    class Input(BaseModel):
        value: secret_type

    secret = "synthetic-private-observation-value"
    value = Input(value=secret_type(secret.encode() if binary else secret))
    content = {"payload": [value, {"nested": value}]}
    if container == "command":
        content = Command(resume=content)
    run_id = str(uuid4())
    callback = wire_sdk.observation.get_callback_handler(run_id=run_id)
    with wire_sdk.observation.trace_context(run_id, agent_name="masked-root", input=content) as root:
        output = await RunnableLambda(lambda data: data).ainvoke(content, config={"callbacks": [callback]})
        wire_sdk.observation.update_trace(root, output=output)
    wire_sdk.client.flush()

    if adapter._IS_NEW_LANGFUSE:
        observations = exported_spans(wire_sdk)
        root = next(span for span in observations if span.name == "masked-root")
        inputs = [item.value.string_value for item in root.attributes if item.key == "langfuse.observation.input"]
        serialized = " ".join(item.value.string_value for span in observations for item in span.attributes)
    else:
        observations = ingestion_events(wire_sdk)
        inputs = [
            json.dumps(event["body"]["input"])
            for event in observations
            if event["type"] == "trace-create" and event["body"].get("name") == "masked-root"
        ]
        serialized = json.dumps(observations)

    assert secret not in serialized
    assert len(inputs) == 1
    decoded = json.loads(inputs[0])
    if container == "command":
        decoded = decoded["resume"]
        assert isinstance(output, Command)
        output = output.resume
    assert decoded == {"payload": [{"value": "<secret>"}, {"nested": {"value": "<secret>"}}]}
    assert isinstance(output["payload"][0].value, secret_type)


@pytest.mark.asyncio
async def test_failed_run_exports_an_error_observation(wire_sdk):
    """Retain the failure when a real runnable raises an exception."""
    obs = wire_sdk.observation
    run_id = str(uuid4())

    def fail(value):
        raise ValueError("invalid calculation")

    callback = obs.get_callback_handler(run_id=run_id)
    with pytest.raises(ValueError, match="invalid calculation"):
        with obs.trace_context(run_id, agent_name="failing-execution"):
            await RunnableLambda(fail).ainvoke(
                "input", config={"callbacks": [callback], "run_name": "failing-calculation"}
            )
    wire_sdk.client.flush()
    if not adapter._IS_NEW_LANGFUSE:
        events = ingestion_events(wire_sdk)
        failed_id = next(
            event["body"]["id"]
            for event in events
            if event["type"] == "span-create" and event["body"]["name"] == "failing-calculation"
        )
        error = next(
            event["body"] for event in events if event["type"] == "span-update" and event["body"]["id"] == failed_id
        )
        assert error["level"] == "ERROR"
        assert error["statusMessage"] == "invalid calculation"
        assert error["endTime"]
        return
    failed = next(span for span in exported_spans(wire_sdk) if span.name == "failing-calculation")
    assert failed.trace_id.hex() == run_id.replace("-", "")
    assert failed.status.code == 2
    attributes = {item.key: item.value.string_value for item in failed.attributes}
    assert attributes["langfuse.observation.level"] == "ERROR"
    assert "invalid calculation" in attributes["langfuse.observation.status_message"]
