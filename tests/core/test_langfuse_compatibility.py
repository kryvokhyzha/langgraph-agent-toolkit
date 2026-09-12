"""Run the adapter against the installed Langfuse SDK with local transports."""

import asyncio
import json
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr


pytest.importorskip("langfuse")
from langgraph_agent_toolkit.core.observability import langfuse as adapter


@pytest.fixture
def sdk(monkeypatch):
    """Keep all SDK traffic in local memory."""
    public_key = f"pk-test-{uuid4()}"
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", public_key)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-no-network")
    monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse.invalid")
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")
    monkeypatch.setattr(
        adapter,
        "settings",
        adapter.settings.model_copy(
            update={
                "LANGFUSE_PUBLIC_KEY": SecretStr(public_key),
                "LANGFUSE_SECRET_KEY": SecretStr("sk-test-no-network"),
                "LANGFUSE_HOST": "http://langfuse.invalid",
            }
        ),
    )
    requests = []
    prompt = {
        "name": "welcome",
        "version": 3,
        "type": "text",
        "prompt": "Hello {{name}}",
        "config": {},
        "labels": ["production"],
        "tags": [],
        "createdAt": "2026-01-01T00:00:00Z",
    }

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=prompt)
        body = json.loads(request.content) if request.content else {}
        if request.url.path.endswith("/ingestion"):
            return httpx.Response(
                207,
                json={
                    "successes": [{"id": item["id"], "status": 201} for item in body.get("batch", [])],
                    "errors": [],
                },
            )
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={**prompt, **body, "version": 4})

    transport = httpx.Client(transport=httpx.MockTransport(respond))
    kwargs = dict(
        public_key=public_key,
        secret_key="sk-test-no-network",
        host="http://langfuse.invalid",
        httpx_client=transport,
        flush_at=1,
        flush_interval=0.01,
    )
    exporter = None
    if adapter._IS_NEW_LANGFUSE:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        kwargs["tracer_provider"] = TracerProvider()
        if adapter._SDK_MAJOR >= 4:
            kwargs["span_exporter"] = exporter
        else:
            processor = import_module("langfuse._client.span_processor")
            monkeypatch.setattr(processor, "OTLPSpanExporter", lambda **_: exporter)
    client = adapter.Langfuse(**kwargs)
    monkeypatch.setattr(adapter, "_get_langfuse_client", lambda: client)
    observation = adapter.LangfuseObservability()
    yield SimpleNamespace(client=client, observation=observation, requests=requests, exporter=exporter)
    client.shutdown()
    transport.close()


def events(sdk):
    sdk.observation.before_shutdown()
    return [
        item
        for request in sdk.requests
        if request.url.path.endswith("/ingestion")
        for item in json.loads(request.content).get("batch", [])
    ]


def test_prompt_and_feedback_use_real_sdk_contracts(sdk):
    obs = sdk.observation
    result = obs.pull_prompt("welcome", label="production", version=3, template_format="jinja2")
    assert result.invoke({"name": "Sam"}).to_messages()[0].content == "Hello Sam"
    request = next(request for request in sdk.requests if request.method == "GET")
    assert request.url.params["version"] == "3"
    assert "label" not in request.url.params
    obs.push_prompt("welcome", "Hello {{name}}")
    created = next(request for request in sdk.requests if request.method == "POST")
    assert json.loads(created.content)["prompt"] == "Hello {{name}}"
    run_id = str(uuid4())
    obs.record_feedback(run_id, "accuracy", 0.9, comment="Correct", user_id="owner")
    score = next(item["body"] for item in events(sdk) if item["type"] == "score-create")
    expected = run_id.replace("-", "") if adapter._IS_NEW_LANGFUSE else run_id
    assert score["traceId"] == expected
    assert score["comment"] == "Correct"
    assert score["name"] == "accuracy"


@pytest.mark.asyncio
async def test_concurrent_callbacks_keep_trace_user_session_and_output(sdk):
    obs = sdk.observation
    run_ids = [str(uuid4()), str(uuid4())]
    handlers = []

    async def work(value):
        await asyncio.sleep(0)
        return f"reply-{value}"

    async def run(index):
        run_id = run_ids[index]
        handler = obs.get_callback_handler(
            run_id=run_id, user_id=f"user-{index}", session_id=f"session-{index}", update_trace=True
        )
        handlers.append(handler)
        with obs.trace_context(
            run_id,
            agent_name=f"agent-{index}",
            user_id=f"user-{index}",
            session_id=f"session-{index}",
            input=f"input-{index}",
        ) as trace:
            output = await RunnableLambda(work).ainvoke(index, config={"callbacks": [handler]})
            obs.update_trace(trace, output=output)

    await asyncio.gather(run(0), run(1))
    assert handlers[0] is not handlers[1]
    if not adapter._IS_NEW_LANGFUSE:
        batch = events(sdk)
        for index, run_id in enumerate(run_ids):
            traces = [
                item["body"]
                for item in batch
                if item["type"] in ("trace-create", "trace-update") and item["body"].get("id") == run_id
            ]
            assert any(item.get("userId") == f"user-{index}" for item in traces)
            assert any(item.get("sessionId") == f"session-{index}" for item in traces)
            assert any(item.get("output") == f"reply-{index}" for item in traces)
            children = [
                item["body"]
                for item in batch
                if item["type"] == "span-create" and item["body"].get("traceId") == run_id
            ]
            assert children
        return

    obs.before_shutdown()
    spans = sdk.exporter.get_finished_spans()
    for index, run_id in enumerate(run_ids):
        related = [span for span in spans if span.context.trace_id == int(run_id.replace("-", ""), 16)]
        root = next(span for span in related if span.name == f"agent-{index}")
        assert root.attributes["langfuse.observation.output"] == f"reply-{index}"
        assert root.attributes["user.id"] == f"user-{index}"
        assert root.attributes["session.id"] == f"session-{index}"
        children = [span for span in related if span.parent and span.parent.span_id == root.context.span_id]
        assert children
        if adapter._SDK_MAJOR >= 4:
            assert all(span.attributes["user.id"] == f"user-{index}" for span in children)
            assert all(span.attributes["session.id"] == f"session-{index}" for span in children)


def test_delete_prompt_reports_supported_contract(sdk):
    if adapter._SDK_MAJOR == 2:
        with pytest.raises(NotImplementedError):
            sdk.observation.delete_prompt("welcome")
    else:
        sdk.observation.delete_prompt("welcome")
        assert any(request.method == "DELETE" and request.url.path.endswith("/welcome") for request in sdk.requests)


def test_shutdown_does_not_create_a_client(monkeypatch):
    factory = MagicMock()
    monkeypatch.setattr(adapter, "_get_langfuse_client", factory)
    adapter.LangfuseObservability().before_shutdown()
    factory.assert_not_called()


def test_client_is_reused_for_prompt_and_feedback(sdk):
    sdk.observation.pull_prompt("welcome")
    client = sdk.observation._get_client()
    sdk.observation.record_feedback(str(uuid4()), "accuracy", 1)
    assert sdk.observation._get_client() is client


def test_prompt_read_error_does_not_create_a_version(sdk, monkeypatch):
    error = httpx.ConnectError("offline")
    get_prompt = MagicMock(side_effect=error)
    create_prompt = MagicMock()
    monkeypatch.setattr(sdk.client, "get_prompt", get_prompt)
    monkeypatch.setattr(sdk.client, "create_prompt", create_prompt)
    with pytest.raises(httpx.ConnectError):
        sdk.observation.push_prompt("welcome", "Hi")
    create_prompt.assert_not_called()


def test_base_url_alias_is_accepted(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://langfuse.invalid")
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    assert adapter.LangfuseObservability().validate_environment()


def merged_legacy_observations(sdk):
    observations = {}
    for event in events(sdk):
        if event["type"].startswith(("span-", "generation-")):
            body = event["body"]
            observations.setdefault(body["id"], {}).update(body)
    return observations


@pytest.mark.skipif(adapter._SDK_MAJOR != 2, reason="SDK v2 callback contract")
def test_legacy_callbacks_preserve_parentage_tool_calls_and_usage(sdk):
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    trace_id = str(uuid4())
    handler = sdk.observation.get_callback_handler(run_id=trace_id, user_id="owner", session_id="session")
    chain_id, model_id, tool_id, retriever_id = (uuid4() for _ in range(4))
    handler.on_chain_start({"name": "agent"}, {"input": "Find a document"}, run_id=chain_id)
    handler.on_chat_model_start(
        {"name": "chat"},
        [[HumanMessage(content="Find a document")]],
        run_id=model_id,
        parent_run_id=chain_id,
        invocation_params={"model_name": "local-model"},
    )
    response = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {"query": "example"}, "id": "call-1", "type": "tool_call"}],
        usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
    )
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=response)]]), run_id=model_id)
    handler.on_tool_start(
        {"name": "search"},
        '{"query":"example"}',
        inputs={"query": "example"},
        run_id=tool_id,
        parent_run_id=chain_id,
    )
    handler.on_retriever_start({"name": "documents"}, "example", run_id=retriever_id, parent_run_id=tool_id)
    handler.on_retriever_end([Document(page_content="Found", metadata={"source": "test"})], run_id=retriever_id)
    handler.on_tool_end("Found", run_id=tool_id)
    handler.on_chain_end({"answer": "Found"}, run_id=chain_id)
    assert handler.runs == {}

    observations = merged_legacy_observations(sdk)
    assert set(observations) == {str(chain_id), str(model_id), str(tool_id), str(retriever_id)}
    assert all(item["traceId"] == trace_id for item in observations.values())
    assert observations[str(model_id)]["parentObservationId"] == str(chain_id)
    assert observations[str(tool_id)]["parentObservationId"] == str(chain_id)
    assert observations[str(retriever_id)]["parentObservationId"] == str(tool_id)
    assert observations[str(model_id)]["usage"] == {"input": 10, "output": 4, "total": 14}
    assert observations[str(model_id)]["model"] == "local-model"
    assert "call-1" in json.dumps(observations[str(model_id)]["output"])
    assert observations[str(model_id)]["input"][0]["role"] == "user"
    assert observations[str(model_id)]["output"]["role"] == "assistant"
    assert observations[str(tool_id)]["input"] == {"query": "example"}
    assert observations[str(retriever_id)]["output"][0]["page_content"] == "Found"
    assert observations[str(chain_id)]["output"] == {"answer": "Found"}
    assert all(item.get("endTime") for item in observations.values())


@pytest.mark.skipif(adapter._SDK_MAJOR != 2, reason="SDK v2 callback contract")
@pytest.mark.parametrize("kind", ["chain", "llm", "tool", "retriever"])
def test_legacy_error_callbacks_finish_observations(sdk, kind):
    handler = sdk.observation.get_callback_handler(run_id=str(uuid4()))
    run_id = uuid4()
    inputs = ["input"] if kind == "llm" else "input"
    getattr(handler, f"on_{kind}_start")({"name": kind}, inputs, run_id=run_id)
    getattr(handler, f"on_{kind}_error")(RuntimeError("expected failure"), run_id=run_id)
    assert handler.runs == {}
    observation = merged_legacy_observations(sdk)[str(run_id)]
    assert observation["level"] == "ERROR"
    assert observation["statusMessage"] == "expected failure"
    assert observation["endTime"]


@pytest.mark.skipif(adapter._SDK_MAJOR != 2, reason="SDK v2 callback contract")
def test_legacy_text_model_keeps_usage(sdk):
    from langchain_core.outputs import Generation, LLMResult

    handler = sdk.observation.get_callback_handler(run_id=str(uuid4()))
    run_id = uuid4()
    handler.on_llm_start({"name": "completion"}, ["input"], run_id=run_id)
    handler.on_llm_end(
        LLMResult(
            generations=[[Generation(text="answer")]],
            llm_output={"token_usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}},
        ),
        run_id=run_id,
    )
    assert handler.runs == {}
    observation = merged_legacy_observations(sdk)[str(run_id)]
    assert "answer" in json.dumps(observation["output"])
    assert observation["usage"] == {"input": 12, "output": 3, "total": 15}


@pytest.mark.skipif(adapter._SDK_MAJOR != 2, reason="SDK v2 callback contract")
def test_legacy_stream_records_first_token_time_once_and_token_categories(sdk):
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    handler = sdk.observation.get_callback_handler(run_id=str(uuid4()))
    run_id = uuid4()
    handler.on_chat_model_start({"name": "chat"}, [[HumanMessage(content="input")]], run_id=run_id)
    handler.on_llm_new_token("first", run_id=run_id)
    handler.on_llm_new_token("second", run_id=run_id)
    usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "input_token_details": {"cache_read": 80},
        "output_token_details": {"reasoning": 5},
    }
    response = AIMessage(content="answer", usage_metadata=usage)
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=response)]]), run_id=run_id)
    handler.on_llm_new_token("late", run_id=run_id)
    assert not handler.runs
    assert not handler._first_tokens
    assert not handler._generation_ids
    batch = events(sdk)
    updates = [
        event["body"] for event in batch if event["type"] == "generation-update" and event["body"]["id"] == str(run_id)
    ]
    assert len([body for body in updates if body.get("completionStartTime")]) == 1
    finished = next(body for body in updates if body.get("endTime"))
    assert finished["usageDetails"] == {
        "input": 20,
        "output": 15,
        "total": 120,
        "input_cache_read": 80,
        "output_reasoning": 5,
    }
    assert response.usage_metadata == usage


@pytest.mark.skipif(adapter._SDK_MAJOR != 2, reason="SDK v2 callback contract")
def test_legacy_chat_preserves_tool_roles_multimodal_content_and_model_metadata(sdk):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    handler = sdk.observation.get_callback_handler(run_id=str(uuid4()))
    run_id = uuid4()
    content = [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "https://images.invalid/example.png"}},
    ]
    handler.on_chat_model_start(
        {"name": "chat"},
        [
            [
                SystemMessage(content="system"),
                HumanMessage(content=content),
                ToolMessage(content="result", tool_call_id="call-id"),
            ]
        ],
        run_id=run_id,
        metadata={"ls_model_name": "metadata-model"},
    )
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="answer"))]]), run_id=run_id)
    observation = merged_legacy_observations(sdk)[str(run_id)]
    assert [message["role"] for message in observation["input"]] == ["system", "user", "tool"]
    assert observation["input"][1]["content"] == content
    assert observation["input"][2]["tool_call_id"] == "call-id"
    assert observation["output"]["role"] == "assistant"
    assert observation["output"]["content"] == "answer"
    assert observation["model"] == "metadata-model"
