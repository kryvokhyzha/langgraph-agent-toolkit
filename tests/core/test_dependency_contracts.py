import json

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
