import json
from unittest.mock import AsyncMock, Mock, patch

import langsmith
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.pregel.types import StateSnapshot
from langgraph.types import Interrupt

from langgraph_agent_toolkit.agents.agents import Agent
from langgraph_agent_toolkit.schema import ChatHistory, ChatMessage, ServiceMetadata
from langgraph_agent_toolkit.schema.models import OpenAICompatibleName
from langgraph_agent_toolkit.observability.factory import ObservabilityFactory


def test_invoke(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    mock_agent.graph.ainvoke = AsyncMock(return_value=[("values", {"messages": [AIMessage(content=ANSWER)]})])

    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.graph.ainvoke.assert_awaited_once()
    input_message = mock_agent.graph.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER


def test_invoke_custom_agent(test_client, mock_agent) -> None:
    """Test that /invoke works with a custom agent_id path parameter."""
    CUSTOM_AGENT = "custom_agent"
    QUESTION = "What is the weather in Tokyo?"
    CUSTOM_ANSWER = "The weather in Tokyo is sunny."
    DEFAULT_ANSWER = "This is from the default agent."

    # Create a separate mock for the default agent
    default_mock = Mock()
    default_mock.graph = Mock()
    default_mock.graph.ainvoke = AsyncMock(return_value=[("values", {"messages": [AIMessage(content=DEFAULT_ANSWER)]})])
    default_mock.graph.aget_state = AsyncMock(return_value={"tasks": []})
    default_mock.observability = Mock()
    default_mock.observability.get_callback_handler = Mock(return_value=None)

    # Configure our custom mock agent
    mock_agent.graph.ainvoke = AsyncMock(return_value=[("values", {"messages": [AIMessage(content=CUSTOM_ANSWER)]})])

    # Patch get_agent to return the correct agent based on the provided agent_id
    def agent_lookup(agent_id):
        if agent_id == CUSTOM_AGENT:
            return mock_agent
        return default_mock

    with patch("langgraph_agent_toolkit.service.routes.get_agent", side_effect=agent_lookup):
        response = test_client.post(f"/{CUSTOM_AGENT}/invoke", json={"message": QUESTION})
        assert response.status_code == 200

        # Verify custom agent was called and default wasn't
        mock_agent.graph.ainvoke.assert_awaited_once()
        default_mock.graph.ainvoke.assert_not_awaited()

        input_message = mock_agent.graph.ainvoke.await_args.kwargs["input"]["messages"][0]
        assert input_message.content == QUESTION

        output = ChatMessage.model_validate(response.json())
        assert output.type == "ai"
        assert output.content == CUSTOM_ANSWER  # Verify we got the custom agent's response


def test_invoke_model_param(test_client, mock_agent) -> None:
    """Test that the model parameter is correctly passed to the agent."""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    CUSTOM_MODEL = OpenAICompatibleName.OPENAI_COMPATIBLE
    mock_agent.graph.ainvoke = AsyncMock(return_value=[("values", {"messages": [AIMessage(content=ANSWER)]})])

    response = test_client.post("/invoke", json={"message": QUESTION, "model": CUSTOM_MODEL})
    assert response.status_code == 200

    # Verify the model was passed correctly in the config
    mock_agent.graph.ainvoke.assert_awaited_once()
    config = mock_agent.graph.ainvoke.await_args.kwargs["config"]
    assert config["configurable"]["model"] == CUSTOM_MODEL

    # Verify the response is still correct
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER

    # Verify an invalid model throws a validation error
    INVALID_MODEL = "gpt-7-notreal"
    response = test_client.post("/invoke", json={"message": QUESTION, "model": INVALID_MODEL})
    assert response.status_code == 422


def test_invoke_custom_agent_config(test_client, mock_agent) -> None:
    """Test that the agent_config parameter is correctly passed to the agent."""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    CUSTOM_CONFIG = {"spicy_level": 0.1, "additional_param": "value_foo"}

    mock_agent.graph.ainvoke = AsyncMock(return_value=[("values", {"messages": [AIMessage(content=ANSWER)]})])

    response = test_client.post("/invoke", json={"message": QUESTION, "agent_config": CUSTOM_CONFIG})
    assert response.status_code == 200

    # Verify the agent_config was passed correctly in the config
    mock_agent.graph.ainvoke.assert_awaited_once()
    config = mock_agent.graph.ainvoke.await_args.kwargs["config"]
    assert config["configurable"]["spicy_level"] == 0.1
    assert config["configurable"]["additional_param"] == "value_foo"

    # Verify the response is still correct
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER

    # Verify a reserved key in agent_config throws a validation error
    INVALID_CONFIG = {"model": "gpt-4o"}
    response = test_client.post("/invoke", json={"message": QUESTION, "agent_config": INVALID_CONFIG})
    assert response.status_code == 422


def test_invoke_interrupt(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    INTERRUPT = "Confirm weather check"
    mock_agent.graph.ainvoke = AsyncMock(
        return_value=[
            ("values", {"messages": [AIMessage(content=ANSWER)]}),
            ("updates", {"__interrupt__": [Interrupt(value=INTERRUPT)]}),
        ]
    )

    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.graph.ainvoke.assert_awaited_once()
    input_message = mock_agent.graph.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == INTERRUPT


def test_feedback(test_client, mock_agent) -> None:
    """Test successful feedback submission to the default agent."""
    mock_agent.observability.record_feedback = Mock(return_value=None)

    body = {
        "run_id": "847c6285-8fc9-4560-a83f-4e6285809254",
        "key": "human-feedback-stars",
        "score": 0.8,
        "kwargs": {"comment": "Great response!"},
    }

    response = test_client.post("/feedback", json=body)

    assert response.status_code == 201
    assert response.json() == {
        "status": "success",
        "run_id": "847c6285-8fc9-4560-a83f-4e6285809254",
        "message": f"Feedback 'human-feedback-stars' recorded successfully for run 847c6285-8fc9-4560-a83f-4e6285809254.",
    }

    mock_agent.observability.record_feedback.assert_called_once_with(
        run_id="847c6285-8fc9-4560-a83f-4e6285809254", key="human-feedback-stars", score=0.8, comment="Great response!"
    )


def test_feedback_custom_agent(test_client) -> None:
    """Test feedback submission to a specific agent."""
    CUSTOM_AGENT = "custom_agent"

    # Create a separate mock for the custom agent
    custom_mock = Mock()
    custom_mock.observability = Mock()
    custom_mock.observability.record_feedback = Mock(return_value=None)

    # Create a mock for default agent too
    default_mock = Mock()
    default_mock.observability = Mock()
    default_mock.observability.record_feedback = Mock(return_value=None)

    body = {"run_id": "847c6285-8fc9-4560-a83f-4e6285809254", "key": "human-feedback-stars", "score": 0.8}

    # Patch get_agent to return the correct agent based on the provided agent_id
    def agent_lookup(agent_id):
        if agent_id == CUSTOM_AGENT:
            return custom_mock
        return default_mock

    with patch("langgraph_agent_toolkit.service.routes.get_agent", side_effect=agent_lookup):
        # Use the correct endpoint with agent_id as a query parameter
        response = test_client.post("/feedback", json=body, params={"agent_id": CUSTOM_AGENT})
        assert response.status_code == 201

        # Verify custom agent's observability was used
        custom_mock.observability.record_feedback.assert_called_once()
        default_mock.observability.record_feedback.assert_not_called()


def test_feedback_error_handling(test_client, mock_agent) -> None:
    """Test error handling in feedback endpoint."""
    # Test ValueError from observability platform
    mock_agent.observability.record_feedback = Mock(side_effect=ValueError("Invalid feedback key"))

    body = {"run_id": "847c6285-8fc9-4560-a83f-4e6285809254", "key": "invalid-key", "score": 0.8}

    response = test_client.post("/feedback", json=body)
    assert response.status_code == 400
    assert "Invalid feedback key" in response.json()["detail"]

    # Test unexpected exception
    mock_agent.observability.record_feedback = Mock(side_effect=RuntimeError("Unexpected error"))

    response = test_client.post("/feedback", json=body)
    assert response.status_code == 500
    assert "Unexpected error recording feedback" in response.json()["detail"]


@patch("langgraph_agent_toolkit.observability.langsmith.LangsmithClient")
def test_feedback_langsmith(mock_client: langsmith.Client, test_client, mock_agent) -> None:
    """Test the Langsmith implementation for backward compatibility."""
    # This test can be removed once the transition to the new implementation is complete
    ls_instance = mock_client.return_value
    ls_instance.create_feedback.return_value = None

    # Mock the agent's observability platform's record_feedback method
    mock_agent.observability.record_feedback = Mock(return_value=None)

    body = {
        "run_id": "847c6285-8fc9-4560-a83f-4e6285809254",
        "key": "human-feedback-stars",
        "score": 0.8,
    }
    response = test_client.post("/feedback", json=body)
    # Update expected status code to 201
    assert response.status_code == 201
    # Update expected response format
    assert response.json() == {
        "status": "success",
        "run_id": "847c6285-8fc9-4560-a83f-4e6285809254",
        "message": f"Feedback 'human-feedback-stars' recorded successfully for run 847c6285-8fc9-4560-a83f-4e6285809254.",
    }

    # Verify that the agent's observability platform's record_feedback method was called correctly
    mock_agent.observability.record_feedback.assert_called_once_with(
        run_id="847c6285-8fc9-4560-a83f-4e6285809254",
        key="human-feedback-stars",
        score=0.8,
    )


def test_history(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    user_question = HumanMessage(content=QUESTION)
    agent_response = AIMessage(content=ANSWER)

    # Create a proper StateSnapshot
    state_snapshot = StateSnapshot(
        values={"messages": [user_question, agent_response]},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
    )

    mock_agent.graph.get_state.return_value = state_snapshot

    response = test_client.post("/history", json={"thread_id": "7bcc7cc1-99d7-4b1d-bdb5-e6f90ed44de6"})
    assert response.status_code == 200

    output = ChatHistory.model_validate(response.json())
    assert output.messages[0].type == "human"
    assert output.messages[0].content == QUESTION
    assert output.messages[1].type == "ai"
    assert output.messages[1].content == ANSWER


@pytest.mark.asyncio
async def test_stream(test_client, mock_agent) -> None:
    """Test streaming tokens and messages."""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " in", " Tokyo", " is", " sunny", "."]
    FINAL_ANSWER = "The weather in Tokyo is sunny."

    # Configure mock to use our async iterator function
    events = [
        (
            "messages",
            (
                AIMessageChunk(content=token),
                {"tags": []},
            ),
        )
        for token in TOKENS
    ] + [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=FINAL_ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.graph.astream = mock_astream

    # Add observability mock for callback_handler
    mock_agent.observability.get_callback_handler = Mock(return_value=None)

    # Make request with streaming
    with test_client.stream("POST", "/stream", json={"message": QUESTION, "stream_tokens": True}) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # Skip [DONE] message
                messages.append(json.loads(line.lstrip("data: ")))

        # Verify streamed tokens
        token_messages = [msg for msg in messages if msg["type"] == "token"]
        assert len(token_messages) == len(TOKENS)
        for i, msg in enumerate(token_messages):
            assert msg["content"] == TOKENS[i]

        # Verify final message
        final_messages = [msg for msg in messages if msg["type"] == "message"]
        assert len(final_messages) == 1
        assert final_messages[0]["content"]["content"] == FINAL_ANSWER
        assert final_messages[0]["content"]["type"] == "ai"


@pytest.mark.asyncio
async def test_stream_no_tokens(test_client, mock_agent) -> None:
    """Test streaming without tokens."""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " in", " Tokyo", " is", " sunny", "."]
    FINAL_ANSWER = "The weather in Tokyo is sunny."

    # Configure mock to use our async iterator function
    events = [
        (
            "messages",
            (
                AIMessageChunk(content=token),
                {"tags": []},
            ),
        )
        for token in TOKENS
    ] + [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=FINAL_ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.graph.astream = mock_astream

    # Add observability mock for callback_handler
    mock_agent.observability.get_callback_handler = Mock(return_value=None)

    # Make request with streaming disabled
    with test_client.stream("POST", "/stream", json={"message": QUESTION, "stream_tokens": False}) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # Skip [DONE] message
                messages.append(json.loads(line.lstrip("data: ")))

        # Verify no token messages
        token_messages = [msg for msg in messages if msg["type"] == "token"]
        assert len(token_messages) == 0

        # Verify final message
        assert len(messages) == 1
        assert messages[0]["type"] == "message"
        assert messages[0]["content"]["content"] == FINAL_ANSWER
        assert messages[0]["content"]["type"] == "ai"


def test_stream_interrupt(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    INTERRUPT = "Confirm weather check"
    # Configure mock to use our async iterator function
    events = [
        (
            "updates",
            {"__interrupt__": [Interrupt(value=INTERRUPT)]},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.graph.astream = mock_astream

    # Add observability mock for callback_handler
    mock_agent.observability.get_callback_handler = Mock(return_value=None)

    # Make request with streaming disabled
    with test_client.stream("POST", "/stream", json={"message": QUESTION, "stream_tokens": False}) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # Skip [DONE] message
                messages.append(json.loads(line.lstrip("data: ")))

        # Verify interrupt message
        assert len(messages) == 1
        assert messages[0]["content"]["content"] == INTERRUPT
        assert messages[0]["content"]["type"] == "ai"


def test_info(test_client, mock_settings) -> None:
    """Test that /info returns the correct service metadata."""

    base_agent = Agent(description="A base agent.", graph=None)
    mock_settings.AUTH_SECRET = None
    mock_settings.DEFAULT_MODEL = OpenAICompatibleName.OPENAI_COMPATIBLE
    mock_settings.AVAILABLE_MODELS = {OpenAICompatibleName.OPENAI_COMPATIBLE}
    with patch.dict("langgraph_agent_toolkit.agents.agents.agents", {"base-agent": base_agent}, clear=True):
        response = test_client.get("/info")
        assert response.status_code == 200
        output = ServiceMetadata.model_validate(response.json())

    assert output.default_agent == "langgraph-supervisor-agent"
    assert len(output.agents) == 1
    assert output.agents[0].key == "base-agent"
    assert output.agents[0].description == "A base agent."

    assert output.default_model == OpenAICompatibleName.OPENAI_COMPATIBLE
    assert output.models == [OpenAICompatibleName.OPENAI_COMPATIBLE]
