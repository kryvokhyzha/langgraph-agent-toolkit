from unittest.mock import Mock

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages import (
    ChatMessage as LangchainChatMessage,
)
from pydantic import BaseModel

from langgraph_agent_toolkit.helper.utils import langchain_to_chat_message
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
        input_dict = {"raw": Mock(content=raw_content), "other": "data"}
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
