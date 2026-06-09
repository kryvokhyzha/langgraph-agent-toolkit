import json
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import HTTPStatusError, Request, Response

from langgraph_agent_toolkit.client import AgentClient, AgentClientError
from langgraph_agent_toolkit.schema import (
    AddMessagesResponse,
    AgentInfo,
    ChatHistory,
    ChatMessage,
    ClearHistoryResponse,
    FeedbackResponse,
    MessageInput,
    ServiceMetadata,
)


def test_init(mock_env):
    """Test client initialization with different parameters."""
    # Test default values
    client = AgentClient(get_info=False)
    assert client.base_url == "http://0.0.0.0"
    assert client.timeout is None

    # Test custom values
    client = AgentClient(
        base_url="http://test",
        timeout=30.0,
        get_info=False,
    )
    assert client.base_url == "http://test"
    assert client.timeout == 30.0
    client.update_agent("test-agent", verify=False)
    assert client.agent == "test-agent"


def test_headers(mock_env):
    """Test header generation with and without auth."""
    # Test without auth
    client = AgentClient(get_info=False)
    assert client._headers == {}

    # Test with auth
    with patch.dict(os.environ, {"AUTH_SECRET": "test-secret"}, clear=True):
        client = AgentClient(get_info=False)
        assert client._headers == {"Authorization": "Bearer test-secret"}


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_invoke(agent_client, mode):
    """invoke()/ainvoke(): parse the response into a ChatMessage, forward every param, raise on 5xx."""
    QUESTION = "What is the weather?"
    ANSWER = "The weather is sunny."
    mock_request = Request("POST", "http://test/invoke")
    mock_response = Response(200, json={"type": "ai", "content": ANSWER}, request=mock_request)
    target = "httpx.post" if mode == "sync" else "httpx.AsyncClient.post"

    async def call(**kw):
        if mode == "sync":
            return agent_client.invoke({"message": QUESTION}, **kw)
        return await agent_client.ainvoke({"message": QUESTION}, **kw)

    # Success: the JSON body is parsed into a ChatMessage.
    with patch(target, return_value=mock_response):
        response = await call()
        assert isinstance(response, ChatMessage)
        assert response.type == "ai"
        assert response.content == ANSWER

    # Every parameter is forwarded in the request body.
    with patch(target, return_value=mock_response) as mock_post:
        await call(
            model_name="gpt-4o",
            model_provider="openai",
            model_config_key="gpt4o",
            thread_id="test-thread",
            user_id="test-user",
            agent_config={"temperature": 0.7},
            recursion_limit=5,
        )
        body = mock_post.call_args.kwargs["json"]
        assert body["input"]["message"] == QUESTION
        assert body["model_name"] == "gpt-4o"
        assert body["model_provider"] == "openai"
        assert body["model_config_key"] == "gpt4o"
        assert body["thread_id"] == "test-thread"
        assert body["user_id"] == "test-user"
        assert body["agent_config"] == {"temperature": 0.7}
        assert body["recursion_limit"] == 5

    # 5xx -> AgentClientError
    error_response = Response(500, text="Internal Server Error", request=mock_request)
    with patch(target, return_value=error_response):
        with pytest.raises(AgentClientError) as exc:
            await call()
        assert "500 Internal Server Error" in str(exc.value)


def test_stream(agent_client):
    """Test synchronous streaming."""
    QUESTION = "What is the weather?"
    TOKENS = ["The", " weather", " is", " sunny", "."]
    FINAL_ANSWER = "The weather is sunny."

    # Create mock response with streaming events
    events = (
        [f"data: {json.dumps({'type': 'token', 'content': token})}" for token in TOKENS]
        + [f"data: {json.dumps({'type': 'message', 'content': {'type': 'ai', 'content': FINAL_ANSWER}})}"]
        + ["data: [DONE]"]
    )

    # Mock the streaming response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = events
    mock_response.request = Request("POST", "http://test/stream")
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=None)

    with patch("httpx.stream", return_value=mock_response):
        # Collect all streamed responses
        responses = list(agent_client.stream({"message": QUESTION}))

        # Verify tokens were streamed
        assert len(responses) == len(TOKENS) + 1  # tokens + final message
        for i, token in enumerate(TOKENS):
            assert responses[i] == token

        # Verify final message
        final_message = responses[-1]
        assert isinstance(final_message, ChatMessage)
        assert final_message.type == "ai"
        assert final_message.content == FINAL_ANSWER

    # Test with all parameters
    with patch("httpx.stream", return_value=mock_response) as mock_stream:
        list(
            agent_client.stream(
                {"message": QUESTION},
                model_name="gpt-4o",
                model_provider="openai",
                model_config_key="gpt4o",
                thread_id="test-thread",
                user_id="test-user",
                agent_config={"temperature": 0.7},
                recursion_limit=5,
                stream_tokens=True,
            )
        )
        # Verify request
        args, kwargs = mock_stream.call_args
        assert kwargs["json"]["input"]["message"] == QUESTION
        assert kwargs["json"]["model_name"] == "gpt-4o"
        assert kwargs["json"]["model_provider"] == "openai"
        assert kwargs["json"]["model_config_key"] == "gpt4o"
        assert kwargs["json"]["thread_id"] == "test-thread"
        assert kwargs["json"]["user_id"] == "test-user"
        assert kwargs["json"]["agent_config"] == {"temperature": 0.7}
        assert kwargs["json"]["recursion_limit"] == 5
        assert kwargs["json"]["stream_tokens"] is True

    # Test error response
    error_response = Response(500, text="Internal Server Error", request=Request("POST", "http://test/stream"))
    error_response_mock = Mock()
    error_response_mock.__enter__ = Mock(return_value=error_response)
    error_response_mock.__exit__ = Mock(return_value=None)
    with patch("httpx.stream", return_value=error_response_mock):
        with pytest.raises(AgentClientError) as exc:
            list(agent_client.stream({"message": QUESTION}))
        assert "500 Internal Server Error" in str(exc.value)


@pytest.mark.asyncio
async def test_astream(agent_client):
    """Test asynchronous streaming."""
    QUESTION = "What is the weather?"
    TOKENS = ["The", " weather", " is", " sunny", "."]
    FINAL_ANSWER = "The weather is sunny."

    # Create mock response with streaming events
    events = (
        [f"data: {json.dumps({'type': 'token', 'content': token})}" for token in TOKENS]
        + [f"data: {json.dumps({'type': 'message', 'content': {'type': 'ai', 'content': FINAL_ANSWER}})}"]
        + ["data: [DONE]"]
    )

    # Create an async iterator for the events
    async def async_events():
        for event in events:
            yield event

    # Mock the streaming response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.request = Request("POST", "http://test/stream")
    mock_response.aiter_lines = Mock(return_value=async_events())
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    # Use Mock() instead of AsyncMock() since raise_for_status is synchronous
    mock_response.raise_for_status = Mock()

    # Create a mock client that returns the mock_response directly (not as a coroutine)
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    # Make stream a regular method that returns the response object directly
    mock_client.stream = Mock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        # Collect all streamed responses
        responses = []
        async for response in agent_client.astream({"message": QUESTION}):
            responses.append(response)

        # Verify tokens were streamed
        assert len(responses) == len(TOKENS) + 1  # tokens + final message
        for i, token in enumerate(TOKENS):
            assert responses[i] == token

        # Verify final message
        final_message = responses[-1]
        assert isinstance(final_message, ChatMessage)
        assert final_message.type == "ai"
        assert final_message.content == FINAL_ANSWER

    # Test with all parameters
    with patch("httpx.AsyncClient", return_value=mock_client) as mock_client_class:
        async for _ in agent_client.astream(
            {"message": QUESTION},
            model_name="gpt-4o",
            model_provider="openai",
            model_config_key="gpt4o",
            thread_id="test-thread",
            user_id="test-user",
            agent_config={"temperature": 0.7},
            recursion_limit=5,
            stream_tokens=True,
        ):
            pass

        # Get the json payload that was passed to stream
        # This is a bit complex due to the multiple layers of mocking
        mock_client_instance = mock_client_class.return_value.__aenter__.return_value
        stream_call = mock_client_instance.stream.call_args
        kwargs = stream_call[1]

        assert kwargs["json"]["input"]["message"] == QUESTION
        assert kwargs["json"]["model_name"] == "gpt-4o"
        assert kwargs["json"]["model_provider"] == "openai"
        assert kwargs["json"]["model_config_key"] == "gpt4o"
        assert kwargs["json"]["thread_id"] == "test-thread"
        assert kwargs["json"]["user_id"] == "test-user"
        assert kwargs["json"]["agent_config"] == {"temperature": 0.7}
        assert kwargs["json"]["recursion_limit"] == 5
        assert kwargs["json"]["stream_tokens"] is True

    # Test error response
    http_error = HTTPStatusError(
        "500 Internal Server Error",
        request=Request("POST", "http://test/stream"),
        response=Response(500, text="Internal Server Error", request=Request("POST", "http://test/stream")),
    )

    # Set up error mock client
    error_mock_client = AsyncMock()
    error_mock_client.__aenter__.return_value = error_mock_client
    # Make stream raise the exception when called directly
    error_mock_client.stream = Mock(side_effect=http_error)

    with patch("httpx.AsyncClient", return_value=error_mock_client):
        with pytest.raises(AgentClientError) as exc:
            async for _ in agent_client.astream({"message": QUESTION}):
                pass
        assert "500 Internal Server Error" in str(exc.value)


def test_stream_jsonl(agent_client):
    """stream_jsonl parses bare NDJSON lines (no `data:` prefix, no `[DONE]`) and hits /stream/jsonl."""
    TOKENS = ["The", " weather", " is", " sunny", "."]
    FINAL_ANSWER = "The weather is sunny."

    events = [json.dumps({"type": "token", "content": t}) for t in TOKENS] + [
        json.dumps({"type": "message", "content": {"type": "ai", "content": FINAL_ANSWER}})
    ]

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = events
    mock_response.request = Request("POST", "http://test/stream/jsonl")
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=None)

    with patch("httpx.stream", return_value=mock_response) as mock_stream:
        responses = list(agent_client.stream_jsonl({"message": "hi"}))

    # Hit the JSON Lines endpoint, not the SSE one.
    assert mock_stream.call_args.args[1].endswith("/stream/jsonl")
    # Same ChatMessage | str contract as the SSE stream.
    assert responses[: len(TOKENS)] == TOKENS
    final = responses[-1]
    assert isinstance(final, ChatMessage)
    assert final.type == "ai"
    assert final.content == FINAL_ANSWER


async def test_astream_jsonl(agent_client):
    """astream_jsonl parses NDJSON asynchronously and hits /stream/jsonl."""
    TOKENS = ["The", " weather", " is", " sunny", "."]
    FINAL_ANSWER = "The weather is sunny."

    events = [json.dumps({"type": "token", "content": t}) for t in TOKENS] + [
        json.dumps({"type": "message", "content": {"type": "ai", "content": FINAL_ANSWER}})
    ]

    async def async_events():
        for event in events:
            yield event

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.request = Request("POST", "http://test/stream/jsonl")
    mock_response.aiter_lines = Mock(return_value=async_events())
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.stream = Mock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        responses = [r async for r in agent_client.astream_jsonl({"message": "hi"})]

    assert mock_client.stream.call_args.args[1].endswith("/stream/jsonl")
    assert responses[: len(TOKENS)] == TOKENS
    final = responses[-1]
    assert isinstance(final, ChatMessage)
    assert final.content == FINAL_ANSWER


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_create_feedback(agent_client, mode):
    """create_feedback()/acreate_feedback(): forward run_id/key/score/kwargs, raise on 5xx.

    Only the sync method returns a FeedbackResponse; the async one returns None.
    """
    RUN_ID, KEY, SCORE, KWARGS = "test-run", "test-key", 0.8, {"comment": "Great response!"}
    success_json = {"status": "success", "run_id": RUN_ID, "message": "Feedback recorded successfully."}
    mock_response = Response(200, json=success_json, request=Request("POST", "http://test/feedback"))
    target = "httpx.post" if mode == "sync" else "httpx.AsyncClient.post"

    async def call(*args):
        if mode == "sync":
            return agent_client.create_feedback(*args)
        return await agent_client.acreate_feedback(*args)

    with patch(target, return_value=mock_response) as mock_post:
        result = await call(RUN_ID, KEY, SCORE, KWARGS)
        if mode == "sync":
            assert isinstance(result, FeedbackResponse)
            assert result.status == "success"
            assert result.run_id == RUN_ID
        body = mock_post.call_args.kwargs["json"]
        assert body["run_id"] == RUN_ID
        assert body["key"] == KEY
        assert body["score"] == SCORE
        assert body["kwargs"] == KWARGS

    error_response = Response(500, text="Internal Server Error", request=Request("POST", "http://test/feedback"))
    with patch(target, return_value=error_response):
        with pytest.raises(AgentClientError) as exc:
            await call(RUN_ID, KEY, SCORE)
        assert "500 Internal Server Error" in str(exc.value)


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_get_history(agent_client, mode):
    """get_history()/aget_history(): parse ChatHistory, hit /history with thread_id+user_id params, raise on 5xx."""
    THREAD_ID = "test-thread"
    HISTORY = {
        "messages": [
            {"type": "human", "content": "What is the weather?"},
            {"type": "ai", "content": "The weather is sunny."},
        ]
    }
    mock_response = Response(200, json=HISTORY, request=Request("GET", "http://test/history"))
    target = "httpx.get" if mode == "sync" else "httpx.AsyncClient.get"

    async def call(**kw):
        if mode == "sync":
            return agent_client.get_history(THREAD_ID, **kw)
        return await agent_client.aget_history(THREAD_ID, **kw)

    with patch(target, return_value=mock_response) as mock_get:
        history = await call(user_id="test-user")
        assert isinstance(history, ChatHistory)
        assert [m.type for m in history.messages] == ["human", "ai"]
        args, kwargs = mock_get.call_args
        assert args[0].endswith("/history")
        assert kwargs["params"]["thread_id"] == THREAD_ID
        assert kwargs["params"]["user_id"] == "test-user"

    error_response = Response(500, text="Internal Server Error", request=Request("GET", "http://test/history"))
    with patch(target, return_value=error_response):
        with pytest.raises(AgentClientError) as exc:
            await call()
        assert "500 Internal Server Error" in str(exc.value)


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_clear_history(agent_client, mode):
    """clear_history()/aclear_history(): parse response, require thread_id|user_id, raise on 5xx."""
    THREAD_ID, USER_ID = "test-thread", "test-user"
    success_json = {
        "status": "success",
        "thread_id": THREAD_ID,
        "user_id": USER_ID,
        "message": "Messages cleared successfully.",
    }
    mock_response = Response(200, json=success_json, request=Request("DELETE", "http://test/history/clear"))
    target = "httpx.delete" if mode == "sync" else "httpx.AsyncClient.delete"

    async def call(*args):
        if mode == "sync":
            return agent_client.clear_history(*args)
        return await agent_client.aclear_history(*args)

    with patch(target, return_value=mock_response) as mock_delete:
        response = await call(THREAD_ID, USER_ID)
        assert isinstance(response, ClearHistoryResponse)
        assert response.status == "success"
        assert response.thread_id == THREAD_ID
        assert response.user_id == USER_ID
        body = mock_delete.call_args.kwargs["json"]
        assert body["thread_id"] == THREAD_ID
        assert body["user_id"] == USER_ID

    # Requires at least one of thread_id / user_id.
    with pytest.raises(AgentClientError) as exc:
        await call()
    assert "At least one of thread_id or user_id must be provided" in str(exc.value)

    error_response = Response(500, text="Internal Server Error", request=Request("DELETE", "http://test/history/clear"))
    with patch(target, return_value=error_response):
        with pytest.raises(AgentClientError) as exc:
            await call(THREAD_ID)
        assert "500 Internal Server Error" in str(exc.value)


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_add_messages(agent_client, mode):
    """add_messages()/aadd_messages(): accept dicts or MessageInput, require thread_id|user_id, raise on 5xx."""
    THREAD_ID = "test-thread"
    DICT_MESSAGES = [{"type": "human", "content": "Hello!"}, {"type": "ai", "content": "Hi there!"}]
    INPUT_MESSAGES = [MessageInput(type="human", content="Hello!"), MessageInput(type="ai", content="Hi there!")]
    success_json = {"status": "success", "thread_id": THREAD_ID, "message": "Added 2 messages to chat history."}
    mock_response = Response(201, json=success_json, request=Request("POST", "http://test/history/add_messages"))
    target = "httpx.post" if mode == "sync" else "httpx.AsyncClient.post"

    async def call(messages, *args):
        if mode == "sync":
            return agent_client.add_messages(messages, *args)
        return await agent_client.aadd_messages(messages, *args)

    # Both a list of dicts and a list of MessageInput serialize to the same body.
    for messages in (DICT_MESSAGES, INPUT_MESSAGES):
        with patch(target, return_value=mock_response) as mock_post:
            response = await call(messages, THREAD_ID)
            assert isinstance(response, AddMessagesResponse)
            assert response.status == "success"
            body = mock_post.call_args.kwargs["json"]
            assert body["thread_id"] == THREAD_ID
            assert len(body["messages"]) == 2
            assert body["messages"][0]["type"] == "human"
            assert body["messages"][0]["content"] == "Hello!"

    # Requires at least one of thread_id / user_id.
    with pytest.raises(AgentClientError) as exc:
        await call(DICT_MESSAGES)
    assert "At least one of thread_id or user_id must be provided" in str(exc.value)

    error_response = Response(
        500, text="Internal Server Error", request=Request("POST", "http://test/history/add_messages")
    )
    with patch(target, return_value=error_response):
        with pytest.raises(AgentClientError) as exc:
            await call(DICT_MESSAGES, THREAD_ID)
        assert "500 Internal Server Error" in str(exc.value)


def test_info(agent_client):
    assert agent_client.info is None
    assert agent_client.agent == "test-agent"

    # Mock info response
    test_info = ServiceMetadata(
        default_agent="custom-agent",
        agents=[AgentInfo(key="custom-agent", description="Custom agent")],
    )
    test_response = Response(200, json=test_info.model_dump(), request=Request("GET", "http://test/info"))

    # Update an existing client with info
    with patch("httpx.get", return_value=test_response):
        agent_client.retrieve_info()

    assert agent_client.info == test_info
    assert agent_client.agent == "custom-agent"

    # Test invalid update_agent
    with pytest.raises(AgentClientError) as exc:
        agent_client.update_agent("unknown-agent")
    assert "Agent unknown-agent not found in available agents: custom-agent" in str(exc.value)

    # Test a fresh client with info
    with patch("httpx.get", return_value=test_response):
        agent_client = AgentClient(base_url="http://test")
    assert agent_client.info == test_info
    assert agent_client.agent == "custom-agent"

    # Test error on invoke if no agent set
    agent_client = AgentClient(base_url="http://test", get_info=False)
    with pytest.raises(AgentClientError) as exc:
        agent_client.invoke("test")
    assert "No agent selected. Use update_agent() to select an agent." in str(exc.value)


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_invoke_accepts_model_provider_enum(agent_client, mode):
    """invoke()/ainvoke() accept a ModelProvider enum and serialize it to a plain string value.

    Regression guard for the ainvoke fix that normalized ModelProvider -> .value (its siblings
    already did this). The serialized body must hold a plain ``str``, not an enum instance.
    """
    from langgraph_agent_toolkit.schema.models import ModelProvider

    mock_response = Response(200, json={"type": "ai", "content": "ok"}, request=Request("POST", "http://test/invoke"))
    target = "httpx.post" if mode == "sync" else "httpx.AsyncClient.post"

    async def call():
        if mode == "sync":
            return agent_client.invoke({"message": "hi"}, model_provider=ModelProvider.OPENAI)
        return await agent_client.ainvoke({"message": "hi"}, model_provider=ModelProvider.OPENAI)

    with patch(target, return_value=mock_response) as mock_post:
        await call()

    serialized = mock_post.call_args.kwargs["json"]["model_provider"]
    assert serialized == "openai"
    assert type(serialized) is str  # a plain string, not a ModelProvider enum instance
