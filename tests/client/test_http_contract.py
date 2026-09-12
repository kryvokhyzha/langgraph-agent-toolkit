import asyncio
import json
from unittest.mock import patch

import httpx
import pytest

from langgraph_agent_toolkit.client import AgentClient, AgentClientError
from langgraph_agent_toolkit.schema import FeedbackResponse, MessageInput


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_history_and_feedback_use_real_httpx_contract(mode):
    """Build real HTTPX requests and preserve tool message fields."""
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"messages": [], "next_offset": 20, "total": 35})
        body = json.loads(request.content)
        if request.url.path.endswith("feedback"):
            return httpx.Response(201, json={"run_id": body["run_id"], "message": "saved"})
        return httpx.Response(200, json={"thread_id": body["thread_id"], "message": "saved"})

    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as sync_http:
        async with httpx.AsyncClient(transport=transport) as async_http:
            client = AgentClient(
                "http://test",
                agent="selected",
                get_info=False,
                auth_secret="personal-token",
                http_client=sync_http,
                async_http_client=async_http,
            )
            messages = [
                {"type": "ai", "content": "", "tool_calls": [{"name": "lookup", "args": {}, "id": "call-1"}]},
                {"type": "tool", "content": "found", "tool_call_id": "call-1"},
            ]
            if mode == "sync":
                client.clear_history("thread")
                client.add_messages(messages, "thread")
                feedback = client.create_feedback("run", "score", 1.0)
                history = client.get_history("thread", offset=10, limit=10)
            else:
                await client.aclear_history("thread")
                await client.aadd_messages(messages, "thread")
                feedback = await client.acreate_feedback("run", "score", 1.0)
                history = await client.aget_history("thread", offset=10, limit=10)
            assert requests[0].method == "DELETE"
            assert json.loads(requests[0].content)["thread_id"] == "thread"
            sent = json.loads(requests[1].content)["messages"]
            assert sent[0]["tool_calls"][0]["id"] == "call-1"
            assert sent[1]["tool_call_id"] == "call-1"
            assert requests[2].url.path == "/selected/feedback"
            assert isinstance(feedback, FeedbackResponse)
            assert requests[3].url.params["offset"] == "10"
            assert requests[3].url.params["limit"] == "10"
            assert "user_id" not in requests[3].url.params
            assert history.next_offset == 20
            assert history.total == 35
            assert all(r.headers["authorization"] == "Bearer personal-token" for r in requests)
            await client.aclose()
            assert not sync_http.is_closed
            assert not async_http.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("protocol", ["sse", "jsonl"])
async def test_stream_errors_raise_after_partial_output(mode, protocol):
    """Report a server failure as an error after any received tokens."""
    frames = [
        {"type": "token", "content": "partial"},
        {"type": "error", "content": "backend unavailable"},
    ]
    body = "".join(("data: " if protocol == "sse" else "") + json.dumps(frame) + "\n\n" for frame in frames)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    with httpx.Client(transport=transport) as sync_http:
        async with httpx.AsyncClient(transport=transport) as async_http:
            client = AgentClient(
                "http://test",
                agent="selected",
                get_info=False,
                http_client=sync_http,
                async_http_client=async_http,
            )
            if mode == "sync":
                stream = (
                    client.stream({"message": "hello"})
                    if protocol == "sse"
                    else client.stream_jsonl({"message": "hello"})
                )
                assert next(stream) == "partial"
                with pytest.raises(AgentClientError, match="backend unavailable"):
                    next(stream)
            else:
                stream = (
                    client.astream({"message": "hello"})
                    if protocol == "sse"
                    else client.astream_jsonl({"message": "hello"})
                )
                assert await anext(stream) == "partial"
                with pytest.raises(AgentClientError, match="backend unavailable"):
                    await anext(stream)


@pytest.mark.asyncio
async def test_owned_clients_are_reused_and_closed():
    """Reuse one pool for each request mode and close owned pools."""
    response = lambda request: httpx.Response(200, json={"type": "ai", "content": "ok"})
    sync_http = httpx.Client(transport=httpx.MockTransport(response))
    async_http = httpx.AsyncClient(transport=httpx.MockTransport(response))
    client = AgentClient("http://test", agent="selected", get_info=False)
    with patch("httpx.Client", return_value=sync_http) as sync_factory:
        with patch("httpx.AsyncClient", return_value=async_http) as async_factory:
            async with client:
                client.invoke({"message": "first"})
                client.invoke({"message": "second"})
                await client.ainvoke({"message": "first"})
                await client.ainvoke({"message": "second"})
            assert sync_factory.call_count == 1
            assert async_factory.call_count == 1
    assert sync_http.is_closed
    assert async_http.is_closed


def test_async_clients_reject_reuse_in_another_loop():
    """Require cleanup before an owned pool moves to another event loop."""
    async_http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    client = AgentClient(get_info=False, async_http_client=async_http)
    first = asyncio.new_event_loop()
    second = asyncio.new_event_loop()

    async def get_client():
        return client._get_async_http_client()

    try:
        assert first.run_until_complete(get_client()) is async_http
        with pytest.raises(AgentClientError, match="another event loop"):
            second.run_until_complete(get_client())
    finally:
        first.run_until_complete(async_http.aclose())
        first.close()
        second.close()


def test_tool_message_requires_call_id():
    with pytest.raises(ValueError, match="tool_call_id"):
        MessageInput(type="tool", content="result")


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_sse_comments_do_not_end_the_stream(mode):
    """Ignore keepalive comments and stop only at the completion event."""
    body = (
        ": keepalive\n\nevent: message\nid: first\n"
        'data:{"type":"token","content":"first"}\n\n'
        ': another keepalive\n\ndata: {"type":"token","content":"second"}\n\n'
        'data: [DONE]\n\ndata: {"type":"token","content":"after completion"}\n\n'
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    with httpx.Client(transport=transport) as sync_http:
        async with httpx.AsyncClient(transport=transport) as async_http:
            client = AgentClient(
                "http://test", agent="selected", get_info=False, http_client=sync_http, async_http_client=async_http
            )
            if mode == "sync":
                output = list(client.stream({"message": "hello"}))
            else:
                output = [value async for value in client.astream({"message": "hello"})]
    assert output == ["first", "second"]


@pytest.mark.parametrize(
    "event",
    ["{invalid", "[]", "{}", '{"type":"message","content":"invalid"}', '{"type":"token","content":42}'],
)
def test_malformed_stream_events_raise_client_errors(event):
    """Reject invalid wire data without leaking JSON, key, or validation exceptions."""
    client = AgentClient(get_info=False)
    with pytest.raises(AgentClientError):
        client._parse_jsonl_line(event)
    with pytest.raises(AgentClientError):
        client._parse_stream_line(f"data: {event}")
