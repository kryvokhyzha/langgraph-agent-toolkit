import json
from contextlib import asynccontextmanager

import httpx
import pytest

from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_openai_gateway_with_injected_httpx_clients(use_async, stream):
    """Keep HTTPX injection and gateway response parsing after dependency updates."""
    requests = []

    def respond(request):
        requests.append(request)
        payload = json.loads(request.content)
        assert payload["model"] == "gateway-model"
        assert payload["messages"][-1]["content"] == "Hello"
        if payload.get("stream"):
            chunks = [
                {"delta": {"role": "assistant", "content": "Answer"}, "finish_reason": None},
                {"delta": {}, "finish_reason": "stop"},
            ]
            body = "".join(
                "data: "
                + json.dumps(
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "gateway-model",
                        "choices": [{"index": 0, **chunk}],
                    }
                )
                + "\n\n"
                for chunk in chunks
            )
            return httpx.Response(200, text=body + "data: [DONE]\n\n", headers={"Content-Type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gateway-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant_gateway", "content": "Answer"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as sync_client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            model = CompletionModelFactory.create(
                "openai",
                "gateway-model",
                configurable_fields=(),
                config_prefix="",
                model_parameter_values=(),
                api_key="test-only",
                base_url="https://gateway.invalid/v1",
                http_client=sync_client,
                http_async_client=async_client,
                max_retries=0,
            )
            if stream:
                chunks = [chunk async for chunk in model.astream("Hello")] if use_async else list(model.stream("Hello"))
                assert "".join(chunk.content for chunk in chunks) == "Answer"
            else:
                result = await model.ainvoke("Hello") if use_async else model.invoke("Hello")
                assert result.content == "Answer"
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/chat/completions"


@asynccontextmanager
async def streamed_model(events, *, responses=False):
    requests = []
    body = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"

    def respond(request):
        requests.append(request)
        return httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})

    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as sync_client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            model = CompletionModelFactory.create(
                "openai",
                "gateway-model",
                configurable_fields=(),
                config_prefix="",
                model_parameter_values=(),
                api_key="local-dependency-test",
                base_url="https://gateway.invalid/v1",
                http_client=sync_client,
                http_async_client=async_client,
                use_responses_api=responses,
                max_retries=0,
            )
            yield model, requests


@pytest.mark.parametrize("use_async", [False, True])
async def test_v3_tool_argument_slices_reconstruct_the_complete_call(use_async):
    """Preserve tool arguments through the model factory and Core v3 streams."""
    deltas = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_city",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"city":"'},
                }
            ],
        },
        {"tool_calls": [{"index": 0, "function": {"arguments": "Ки"}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": 'їв"}'}}]},
        {},
    ]
    events = [
        {
            "id": "chatcmpl-tool-args",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gateway-model",
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": "tool_calls" if index == len(deltas) - 1 else None}
            ],
        }
        for index, delta in enumerate(deltas)
    ]

    async with streamed_model(events) as (model, requests):
        if use_async:
            stream = await model.astream_events("Find this city.", version="v3")
            protocol_events = [event async for event in stream]
            chunks = [chunk async for chunk in stream.tool_calls]
            calls = await stream.tool_calls
            output = await stream.output
        else:
            stream = model.stream_events("Find this city.", version="v3")
            protocol_events = list(stream)
            chunks = list(stream.tool_calls)
            calls = stream.tool_calls.get()
            output = stream.output

    slices = [
        event["delta"]["args"]
        for event in protocol_events
        if event["event"] == "content-block-delta" and event["delta"]["type"] == "tool-call-delta"
    ]
    assert slices == ['{"city":"', "Ки", 'їв"}']
    assert [chunk["args"] for chunk in chunks] == slices
    assert json.loads("".join(slices)) == {"city": "Київ"}
    assert (
        calls
        == output.tool_calls
        == [{"name": "lookup", "args": {"city": "Київ"}, "id": "call_city", "type": "tool_call"}]
    )
    assert output.invalid_tool_calls == []
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/chat/completions"


def response_events():
    return [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "msg_city", "type": "message", "role": "assistant", "status": "in_progress", "content": []},
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_city",
            "output_index": 0,
            "content_index": 0,
            "delta": "Київ",
        },
        {
            "type": "response.output_text.done",
            "item_id": "msg_city",
            "output_index": 2,
            "content_index": 0,
            "text": "Київ",
        },
    ]


@pytest.mark.parametrize("use_async", [False, True])
async def test_responses_index_drift_keeps_one_text_block(use_async):
    """Keep one output block when a provider changes an item's final index."""
    async with streamed_model(response_events(), responses=True) as (model, requests):
        chunks = (
            [chunk async for chunk in model.astream("Name the city.")]
            if use_async
            else list(model.stream("Name the city."))
        )
        result = chunks[0]
        for chunk in chunks[1:]:
            result += chunk

    assert result.content == [{"type": "text", "text": "Київ", "index": 0, "id": "msg_city"}]
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/responses"


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("event_type", ["error", "response.failed"])
async def test_responses_error_event_cannot_return_a_successful_partial_answer(use_async, event_type):
    """Report an in-band provider failure after partial text without a retry."""
    error = (
        {"type": "error", "code": "server_error", "message": "synthetic stream failure", "param": None}
        if event_type == "error"
        else {
            "type": "response.failed",
            "response": {
                "id": "resp_failure",
                "error": {"code": "server_error", "message": "synthetic stream failure"},
            },
        }
    )
    received = []
    async with streamed_model([*response_events()[:2], error], responses=True) as (model, requests):
        with pytest.raises(ValueError, match="synthetic stream failure"):
            if use_async:
                async for chunk in model.astream("Return an answer."):
                    received.extend(chunk.content)
            else:
                for chunk in model.stream("Return an answer."):
                    received.extend(chunk.content)

    assert any(block.get("text") == "Київ" for block in received)
    assert len(requests) == 1
