import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages import (
    ChatMessage as LangchainChatMessage,
)
from langgraph.prebuilt.chat_agent_executor import _validate_chat_history
from pydantic import BaseModel

from langgraph_agent_toolkit.helper.utils import langchain_to_chat_message, sanitize_chat_history
from langgraph_agent_toolkit.schema import ChatMessage


class TestLangchainToChatMessage:
    """Tests for langchain_to_chat_message function."""

    def test_human_message_with_string_content(self):
        """Test converting HumanMessage with string content."""
        message = HumanMessage(content="Hello, world!")
        result = langchain_to_chat_message(message)

        assert isinstance(result, ChatMessage)
        assert result.type == "human"
        assert result.content == "Hello, world!"

    def test_human_message_with_list_content(self):
        """Test converting HumanMessage with list content."""
        content = ["Hello", {"type": "text", "text": " world!"}]
        message = HumanMessage(content=content)
        result = langchain_to_chat_message(message)

        assert isinstance(result, ChatMessage)
        assert result.type == "human"
        assert result.content == content

    def test_ai_message_with_string_content(self):
        """Test converting AIMessage with string content."""
        message = AIMessage(content="How can I help you?")
        result = langchain_to_chat_message(message)

        assert isinstance(result, ChatMessage)
        assert result.type == "ai"
        assert result.content == "How can I help you?"

    def test_ai_message_with_tool_calls(self):
        """Test converting AIMessage with tool calls."""
        tool_calls = [{"name": "search", "args": {"query": "weather"}, "id": "call_1"}]
        response_metadata = {"token_usage": {"total_tokens": 100}}

        message = AIMessage(
            content="Let me search for that.", tool_calls=tool_calls, response_metadata=response_metadata
        )
        result = langchain_to_chat_message(message)

        assert isinstance(result, ChatMessage)
        assert result.type == "ai"
        assert result.content == "Let me search for that."
        # LangChain automatically adds 'type': 'tool_call' to tool calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"
        assert result.tool_calls[0]["args"] == {"query": "weather"}
        assert result.tool_calls[0]["id"] == "call_1"
        assert result.tool_calls[0]["type"] == "tool_call"
        assert result.response_metadata == response_metadata

    def test_ai_message_with_list_content(self):
        """Test converting AIMessage with list content."""
        content = [
            {"type": "text", "text": "Here's the weather:"},
            {"type": "image", "url": "http://example.com/weather.png"},
        ]
        message = AIMessage(content=content)
        result = langchain_to_chat_message(message)

        assert isinstance(result, ChatMessage)
        assert result.type == "ai"
        assert result.content == content

    def test_tool_message(self):
        """Test converting ToolMessage."""
        message = ToolMessage(content="Weather is sunny, 25°C", tool_call_id="call_123")
        result = langchain_to_chat_message(message)

        assert isinstance(result, ChatMessage)
        assert result.type == "tool"
        assert result.content == "Weather is sunny, 25°C"
        assert result.tool_call_id == "call_123"

    def test_tool_message_with_list_content(self):
        """Test converting ToolMessage with list content."""
        content = [
            {"type": "text", "text": "Weather data:"},
            {"type": "data", "value": {"temperature": 25, "condition": "sunny"}},
        ]
        message = ToolMessage(content=content, tool_call_id="call_123")
        result = langchain_to_chat_message(message)

        assert isinstance(result, ChatMessage)
        assert result.type == "tool"
        assert result.content == content
        assert result.tool_call_id == "call_123"

    def test_langchain_chat_message_custom(self):
        """Test converting LangchainChatMessage with custom role."""
        custom_data = {"system": "initialization", "config": {"mode": "debug"}}
        message = LangchainChatMessage(role="custom", content=[custom_data])
        result = langchain_to_chat_message(message)

        assert isinstance(result, ChatMessage)
        assert result.type == "custom"
        assert result.content == ""
        assert result.custom_data == custom_data

    def test_langchain_chat_message_unsupported_role(self):
        """Test converting LangchainChatMessage with unsupported role."""
        message = LangchainChatMessage(role="system", content="System message")

        with pytest.raises(ValueError, match="Unsupported chat message role: system"):
            langchain_to_chat_message(message)

    def test_string_input(self):
        """Test converting string input."""
        result = langchain_to_chat_message("Hello from string")

        assert isinstance(result, ChatMessage)
        assert result.type == "ai"
        assert result.content == "Hello from string"

    def test_dict_input(self):
        """Test converting dict input."""
        input_dict = {"text": "Hello from dict", "metadata": {"source": "test"}}
        result = langchain_to_chat_message(input_dict)

        assert isinstance(result, ChatMessage)
        assert result.type == "ai"
        assert result.content == input_dict

    def test_dict_input_with_raw(self):
        """Test converting dict input with raw field."""
        raw_content = "Content from raw field"
        input_dict = {"raw": AIMessage(content=raw_content), "other": "data"}
        result = langchain_to_chat_message(input_dict)

        assert isinstance(result, ChatMessage)
        assert result.type == "ai"
        assert result.content == raw_content

    def test_list_input(self):
        """Test converting list input."""
        input_list = [{"type": "text", "text": "Hello"}, {"type": "image", "url": "http://example.com/image.png"}]
        result = langchain_to_chat_message(input_list)

        assert isinstance(result, ChatMessage)
        assert result.type == "ai"
        assert result.content == input_list

    def test_pydantic_model_input(self):
        """Test converting Pydantic BaseModel input."""

        class TestModel(BaseModel):
            message: str
            priority: int

        model = TestModel(message="Test message", priority=1)
        result = langchain_to_chat_message(model)

        assert isinstance(result, ChatMessage)
        assert result.type == "ai"
        assert result.content == {"message": "Test message", "priority": 1}

    def test_unsupported_message_type(self):
        """Test converting unsupported message type."""
        from langgraph_agent_toolkit.helper.exceptions import UnsupportedMessageTypeError

        with pytest.raises(UnsupportedMessageTypeError, match="Unsupported message type: int"):
            langchain_to_chat_message(123)

    def test_empty_list_input(self):
        """Test converting empty list input."""
        result = langchain_to_chat_message([])

        assert isinstance(result, ChatMessage)
        assert result.type == "ai"
        assert result.content == []

    def test_complex_nested_content(self):
        """Test converting message with complex nested content."""
        complex_content = [
            "Text part",
            {"type": "multimodal", "data": {"text": "Nested text", "metadata": {"source": "complex"}}},
            {"type": "text", "text": "More text"},
        ]
        message = AIMessage(content=complex_content)
        result = langchain_to_chat_message(message)

        assert isinstance(result, ChatMessage)
        assert result.type == "ai"
        assert result.content == complex_content


class TestSanitizeChatHistory:
    """Tests for sanitize_chat_history function."""

    def test_empty_messages(self):
        """Test sanitizing empty message list."""
        result = sanitize_chat_history([])
        assert result == []

    def test_messages_without_tool_calls(self):
        """Test sanitizing messages without any tool calls."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?"),
        ]
        result = sanitize_chat_history(messages)

        assert len(result) == 3
        assert result[0].content == "Hello"
        assert result[1].content == "Hi there!"
        assert result[2].content == "How are you?"

    def test_complete_tool_call_chain(self):
        """Test sanitizing messages with complete tool call chain (AIMessage + ToolMessage)."""
        tool_call_id = "call_123"
        messages = [
            HumanMessage(content="Search for weather"),
            AIMessage(
                content="Let me search",
                tool_calls=[{"name": "search", "args": {"query": "weather"}, "id": tool_call_id, "type": "tool_call"}],
            ),
            ToolMessage(content="Weather is sunny", tool_call_id=tool_call_id),
            AIMessage(content="The weather is sunny!"),
        ]
        result = sanitize_chat_history(messages)

        assert len(result) == 4
        assert result == messages
        _validate_chat_history(result)

    def test_incomplete_tool_call(self):
        """Test sanitizing messages with incomplete tool call (AIMessage without ToolMessage)."""
        messages = [
            HumanMessage(content="Search for weather"),
            AIMessage(
                content="Let me search",
                tool_calls=[{"name": "search", "args": {"query": "weather"}, "id": "call_123", "type": "tool_call"}],
            ),
            # No ToolMessage for call_123
        ]
        result = sanitize_chat_history(messages)

        assert len(result) == 2
        # Tool calls should be removed from the AIMessage
        assert result[1].tool_calls == []
        assert result[1].content == "Let me search"

    def test_incomplete_tool_call_with_empty_content(self):
        """Test sanitizing incomplete tool call when AIMessage has empty content."""
        messages = [
            HumanMessage(content="Search"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search", "args": {}, "id": "call_456", "type": "tool_call"}],
            ),
        ]
        with pytest.raises(ValueError, match="do not have a corresponding ToolMessage"):
            _validate_chat_history(messages)

        result = sanitize_chat_history(messages)
        _validate_chat_history(result)

        assert len(result) == 2
        # Tool calls should be removed and content should indicate interruption
        assert result[1].tool_calls == []
        assert result[1].content == "[Tool call was interrupted]"

    def test_partial_tool_calls_completed(self):
        """Test sanitizing messages where some tool calls are complete and others are not."""
        messages = [
            HumanMessage(content="Search and calculate"),
            AIMessage(
                content="Let me help",
                tool_calls=[
                    {"name": "search", "args": {"query": "data"}, "id": "call_complete", "type": "tool_call"},
                    {"name": "calculator", "args": {"expr": "2+2"}, "id": "call_incomplete", "type": "tool_call"},
                ],
            ),
            ToolMessage(content="Search result", tool_call_id="call_complete"),
            # No ToolMessage for call_incomplete
        ]
        with pytest.raises(ValueError, match="do not have a corresponding ToolMessage"):
            _validate_chat_history(messages)

        result = sanitize_chat_history(messages)
        _validate_chat_history(result)

        assert len(result) == 3
        # Only the complete tool call should remain
        assert len(result[1].tool_calls) == 1
        assert result[1].tool_calls[0]["id"] == "call_complete"

    def test_multiple_incomplete_tool_calls(self):
        """Test sanitizing messages with multiple consecutive incomplete tool calls."""
        messages = [
            HumanMessage(content="Query 1"),
            AIMessage(
                content="Processing 1",
                tool_calls=[{"name": "tool1", "args": {}, "id": "call_1", "type": "tool_call"}],
            ),
            HumanMessage(content="Query 2"),
            AIMessage(
                content="Processing 2",
                tool_calls=[{"name": "tool2", "args": {}, "id": "call_2", "type": "tool_call"}],
            ),
        ]
        result = sanitize_chat_history(messages)

        assert len(result) == 4
        # Both AIMessages should have their tool_calls removed
        assert result[1].tool_calls == []
        assert result[3].tool_calls == []

    def test_preserves_message_metadata(self):
        """Test that sanitization preserves other message metadata."""
        messages = [
            AIMessage(
                id="msg_id_123",
                content="Original content",
                name="agent_name",
                tool_calls=[{"name": "search", "args": {}, "id": "call_1", "type": "tool_call"}],
                response_metadata={"model": "gpt-4"},
            ),
        ]
        result = sanitize_chat_history(messages)

        assert len(result) == 1
        assert result[0].id == "msg_id_123"
        assert result[0].name == "agent_name"
        assert result[0].response_metadata == {"model": "gpt-4"}
        assert result[0].tool_calls == []

    def test_orphaned_tool_message_dropped(self):
        """A ToolMessage with no requesting AIMessage tool call is dropped."""
        messages = [
            HumanMessage(content="Hello"),
            ToolMessage(content="stale result", tool_call_id="ghost", name="search"),
            AIMessage(content="Hi!"),
        ]
        result = sanitize_chat_history(messages)

        assert [type(m).__name__ for m in result] == ["HumanMessage", "AIMessage"]

    def test_orphaned_tool_message_after_trimming_passes_validation(self):
        """History whose leading AIMessage was trimmed (leaving an orphan ToolMessage) is repaired."""
        messages = [
            ToolMessage(content="result for a trimmed call", tool_call_id="trimmed", name="search"),
            HumanMessage(content="continue"),
        ]
        sanitized = sanitize_chat_history(messages)

        assert all(not isinstance(m, ToolMessage) for m in sanitized)
        _validate_chat_history(sanitized)  # no exception
