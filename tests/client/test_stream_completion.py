"""An HTTP 200 or final message does not prove that the SSE stream completed."""

import httpx
import pytest

from langgraph_agent_toolkit.client import AgentClient, AgentClientError


class TrackingSyncStream(httpx.SyncByteStream):
    """Record response exhaustion and synchronous cleanup."""

    def __init__(self, body):
        self.body = body
        self.exhausted = False
        self.closed = False

    def __iter__(self):
        yield self.body
        self.exhausted = True

    def close(self):
        self.closed = True


class TrackingAsyncStream(httpx.AsyncByteStream):
    """Record response exhaustion and asynchronous cleanup."""

    def __init__(self, body):
        self.body = body
        self.exhausted = False
        self.closed = False

    async def __aiter__(self):
        yield self.body
        self.exhausted = True

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "body,expected_count",
    [
        (b"", 0),
        (b'data: {"type":"token","content":"partial"}\n\n', 1),
        (b'data: {"type":"message","content":{"type":"ai","content":"answer"}}\n\n', 1),
    ],
)
async def test_sse_eof_without_done_raises_and_closes(asynchronous, body, expected_count):
    source = TrackingAsyncStream(body) if asynchronous else TrackingSyncStream(body)
    response = httpx.Response(200, stream=source)
    assert not response.is_closed
    assert not source.closed
    transport = httpx.MockTransport(lambda request: response)
    with httpx.Client(transport=transport) as sync_client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            client = AgentClient(
                base_url="http://test",
                agent="test",
                get_info=False,
                verify=False,
                http_client=sync_client,
                async_http_client=async_client,
            )
            received = []
            with pytest.raises(AgentClientError, match="completion marker"):
                if asynchronous:
                    async for value in client.astream({"message": "test"}):
                        received.append(value)
                else:
                    for value in client.stream({"message": "test"}):
                        received.append(value)
            assert len(received) == expected_count
            assert response.is_closed
            assert source.exhausted
            assert source.closed


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_caller_can_close_sse_before_done(asynchronous):
    body = b'data: {"type":"token","content":"partial"}\n\n'
    source = TrackingAsyncStream(body) if asynchronous else TrackingSyncStream(body)
    response = httpx.Response(200, stream=source)
    assert not response.is_closed
    assert not source.closed
    transport = httpx.MockTransport(lambda request: response)
    with httpx.Client(transport=transport) as sync_client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            client = AgentClient(
                base_url="http://test",
                agent="test",
                get_info=False,
                verify=False,
                http_client=sync_client,
                async_http_client=async_client,
            )
            if asynchronous:
                stream = client.astream({"message": "test"})
                assert await anext(stream) == "partial"
                assert not response.is_closed
                assert not source.closed
                await stream.aclose()
            else:
                stream = client.stream({"message": "test"})
                assert next(stream) == "partial"
                assert not response.is_closed
                assert not source.closed
                stream.close()
            assert response.is_closed
            assert source.closed
            assert not source.exhausted
