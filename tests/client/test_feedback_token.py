"""Forward run-scoped feedback tokens only when the caller supplies them."""

import json

import httpx
import pytest

from langgraph_agent_toolkit.client import AgentClient


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_returned_feedback_token_is_forwarded_explicitly_without_client_cache(mode):
    token = "synthetic-feedback-token"
    run_id = "11111111-1111-4111-8111-111111111111"
    feedback_requests = []

    def respond(request):
        if request.url.path == "/agent/invoke":
            return httpx.Response(
                200,
                json={"type": "ai", "content": "Answer", "run_id": run_id, "feedback_token": token},
                request=request,
            )
        assert request.url.path == "/agent/feedback"
        feedback_requests.append(json.loads(request.content))
        return httpx.Response(201, json={"run_id": run_id}, request=request)

    sync_http = httpx.Client(transport=httpx.MockTransport(respond))
    async_http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    async with AgentClient(
        base_url="http://local.invalid",
        agent="agent",
        get_info=False,
        http_client=sync_http,
        async_http_client=async_http,
    ) as client:
        try:
            if mode == "async":
                reply = await client.ainvoke({"message": "hello"})
                await client.acreate_feedback(
                    reply.run_id, "quality", 1, {"comment": "Useful"}, "user-1", feedback_token=reply.feedback_token
                )
                await client.acreate_feedback(reply.run_id, "quality", 0)
            else:
                reply = client.invoke({"message": "hello"})
                client.create_feedback(
                    reply.run_id, "quality", 1, {"comment": "Useful"}, "user-1", feedback_token=reply.feedback_token
                )
                client.create_feedback(reply.run_id, "quality", 0)
        finally:
            sync_http.close()
            await async_http.aclose()

    assert feedback_requests[0] == {
        "run_id": run_id,
        "key": "quality",
        "score": 1.0,
        "user_id": "user-1",
        "kwargs": {"comment": "Useful"},
        "feedback_token": token,
    }
    assert feedback_requests[1].get("feedback_token") is None
