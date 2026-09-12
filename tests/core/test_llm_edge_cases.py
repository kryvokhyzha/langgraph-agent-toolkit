"""Check provider response boundaries through the model factory and real SDK."""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from langgraph_agent_toolkit.core.models import CompletionModelFactory
from langgraph_agent_toolkit.core.models.transport import _CONNECTION_ENV


@pytest.fixture(autouse=True)
def isolate_provider_settings(monkeypatch):
    for name in _CONNECTION_ENV:
        monkeypatch.delenv(name, raising=False)
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.setenv(name, "false")


@asynccontextmanager
async def provider_model(provider_name, respond, *, streaming=True):
    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as sync_client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            options = {
                "api_key": "local-edge-test-key",
                "http_client": sync_client,
                "http_async_client": async_client,
                "streaming": streaming,
                "stream_usage": True,
                "max_retries": 2,
            }
            if provider_name == "azure_openai":
                options.update(
                    azure_endpoint="https://provider.invalid",
                    azure_deployment="local-model",
                    api_version="2025-01-01-preview",
                )
            else:
                options["base_url"] = "https://provider.invalid/v1"
            model = CompletionModelFactory.create(
                provider_name,
                "local-model",
                model_parameter_values=(),
                configurable_fields=(),
                config_prefix="",
                **options,
            )
            yield model, async_client


def event(*, delta=None, finish_reason=None, usage=None):
    value = {
        "id": "chatcmpl-edge-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "local-model",
        "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        "usage": usage,
    }
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


def stream_response(*events):
    return httpx.Response(
        200,
        content=b"".join(events) + b"data: [DONE]\n\n",
        headers={"Content-Type": "text/event-stream"},
    )


@pytest.mark.parametrize("provider_name", ["openai", "azure_openai"])
async def test_empty_choice_usage_chunks_preserve_text_and_length_limit(provider_name):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return stream_response(
            event(),
            event(delta={"role": "assistant", "content": None}),
            event(delta={"content": "Київ "}),
            event(delta={"content": "☀"}),
            event(delta={}, finish_reason="length"),
            event(usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}),
        )

    async with provider_model(provider_name, respond) as (model, _):
        result = await model.ainvoke("Return a short answer.")

    assert result.content == "Київ ☀"
    assert result.response_metadata["finish_reason"] == "length"
    assert result.usage_metadata["input_tokens"] == 7
    assert result.usage_metadata["output_tokens"] == 3
    assert result.usage_metadata["total_tokens"] == 10
    assert len(requests) == 1
    assert requests[0]["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("provider_name", ["openai", "azure_openai"])
async def test_interleaved_tool_fragments_preserve_call_ids_and_arguments(provider_name):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return stream_response(
            event(delta={"role": "assistant", "content": None}),
            event(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_first",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"city":"'},
                        },
                        {
                            "index": 1,
                            "id": "call_second",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"city":"'},
                        },
                    ]
                }
            ),
            event(delta={"tool_calls": [{"index": 1, "function": {"arguments": 'Paris"}'}}]}),
            event(delta={"tool_calls": [{"index": 0, "function": {"arguments": 'Київ"}'}}]}),
            event(delta={}, finish_reason="tool_calls"),
            event(usage={"prompt_tokens": 7, "completion_tokens": 12, "total_tokens": 19}),
        )

    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Return the city name.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    async with provider_model(provider_name, respond) as (model, _):
        result = await model.bind_tools([tool]).ainvoke("Compare two cities.")

    assert result.content == ""
    assert result.tool_calls == [
        {"name": "lookup", "args": {"city": "Київ"}, "id": "call_first", "type": "tool_call"},
        {"name": "lookup", "args": {"city": "Paris"}, "id": "call_second", "type": "tool_call"},
    ]
    assert result.invalid_tool_calls == []
    assert result.response_metadata["finish_reason"] == "tool_calls"
    assert result.usage_metadata["output_tokens"] == 12
    assert len(requests) == 1
    assert requests[0]["tools"] == [tool]


@pytest.mark.parametrize("provider_name", ["openai", "azure_openai"])
@pytest.mark.parametrize("method", ["invoke", "ainvoke", "stream", "astream"])
async def test_streamed_refusal_fragments_are_preserved_without_retry(provider_name, method):
    requests = []

    def respond(request):
        requests.append(request)
        return stream_response(
            event(delta={"role": "assistant", "content": None}),
            event(delta={"refusal": "I cannot "}),
            event(delta={"refusal": "help with that."}),
            event(delta={}, finish_reason="stop"),
            event(usage={"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}),
        )

    async with provider_model(provider_name, respond) as (model, _):
        if method == "ainvoke":
            result = await model.ainvoke("Return the provider refusal.")
        elif method == "invoke":
            result = model.invoke("Return the provider refusal.")
        else:
            chunks = (
                [chunk async for chunk in model.astream("Return the provider refusal.")]
                if method == "astream"
                else list(model.stream("Return the provider refusal."))
            )
            assert [chunk.additional_kwargs["refusal"] for chunk in chunks if "refusal" in chunk.additional_kwargs] == [
                "I cannot ",
                "help with that.",
            ]
            result = chunks[0]
            for chunk in chunks[1:]:
                result += chunk

    assert result.content == ""
    assert result.additional_kwargs["refusal"] == "I cannot help with that."
    assert result.response_metadata["finish_reason"] == "stop"
    assert result.usage_metadata["total_tokens"] == 11
    assert len(requests) == 1


@pytest.mark.parametrize("provider_name", ["openai", "azure_openai"])
async def test_nonstream_refusal_is_preserved_without_retry(provider_name):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-edge-test",
                "object": "chat.completion",
                "created": 1,
                "model": "local-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": None, "refusal": "I cannot help with that."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async with provider_model(provider_name, respond, streaming=False) as (model, _):
        result = await model.ainvoke("Return the provider refusal.")

    assert result.content == ""
    assert result.additional_kwargs["refusal"] == "I cannot help with that."
    assert result.response_metadata["finish_reason"] == "stop"
    assert len(requests) == 1


@pytest.mark.parametrize("provider_name", ["openai", "azure_openai"])
async def test_cancelled_stream_closes_response_without_retry_or_client_close(provider_name):
    waiting_for_next_chunk = asyncio.Event()
    response_closed = asyncio.Event()
    requests = []
    chunks = []

    class WaitingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield event(delta={"role": "assistant", "content": "partial"})
            waiting_for_next_chunk.set()
            await asyncio.Event().wait()

        async def aclose(self):
            response_closed.set()

    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=WaitingStream())
        return stream_response(event(delta={"role": "assistant", "content": "recovered"}, finish_reason="stop"))

    async with provider_model(provider_name, respond) as (model, client):

        async def consume():
            async for chunk in model.astream("Wait for another token."):
                chunks.append(chunk.content)

        pending = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(waiting_for_next_chunk.wait(), timeout=3)
        finally:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, timeout=3)
        assert response_closed.is_set()
        assert "".join(chunks) == "partial"
        assert len(requests) == 1
        assert not client.is_closed
        assert (await model.ainvoke("Continue with a new request.")).content == "recovered"
        assert len(requests) == 2
