import inspect
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages import (
    ChatMessage as LangchainChatMessage,
)
from pydantic import BaseModel

from langgraph_agent_toolkit.helper.exceptions import UnsupportedMessageTypeError
from langgraph_agent_toolkit.schema import ChatMessage


def convert_message_content_to_string(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content

    text: list[str] = []

    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        if content_item["type"] == "text":
            text.append(content_item["text"])
    return "".join(text)


def langchain_to_chat_message(message: BaseMessage | dict | BaseModel | list) -> ChatMessage:
    """Create a ChatMessage from a LangChain message."""
    if not isinstance(message, (BaseMessage, AIMessage, HumanMessage, ToolMessage, LangchainChatMessage)):
        if isinstance(message, BaseModel):
            message = message.model_dump()
        elif isinstance(message, dict):
            if "raw" in message:
                message = message["raw"].content

    match message:
        case HumanMessage():
            human_message = ChatMessage(
                type="human",
                content=message.content,
            )
            return human_message
        case AIMessage():
            content = message.content
            metadata = dict(message.response_metadata)
            refusal = message.additional_kwargs.get("refusal")
            if isinstance(refusal, str) and refusal:
                metadata["refusal"] = refusal
                content = content or refusal
            ai_message = ChatMessage(
                type="ai",
                content=content,
                response_metadata=metadata,
                usage_metadata=message.usage_metadata,
            )
            if message.tool_calls:
                ai_message.tool_calls = message.tool_calls
            return ai_message
        case ToolMessage():
            tool_message = ChatMessage(
                type="tool",
                content=message.content,
                tool_call_id=message.tool_call_id,
            )
            return tool_message
        case LangchainChatMessage():
            if message.role == "custom":
                custom_message = ChatMessage(
                    type="custom",
                    content="",
                    custom_data=message.content[0],
                )
                return custom_message
            else:
                raise ValueError(f"Unsupported chat message role: {message.role}")
        case list():
            return ChatMessage(
                type="ai",
                content=message,
            )
        case str() | dict():
            return ChatMessage(
                type="ai",
                content=message,
            )
        case _:
            raise UnsupportedMessageTypeError(message_type=message.__class__.__name__)


def remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """Remove tool calls from content."""
    if isinstance(content, str):
        return content
    # Anthropic models currently stream tool calls with the `tool_use` content type.
    return [
        content_item for content_item in content if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]


def sanitize_chat_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Keep paired tool calls and tool results in chat history.

    Remove an `AIMessage` tool call without a `ToolMessage` response.
    Remove a `ToolMessage` without a requesting `AIMessage` tool call.

    Args:
        messages: Messages to sanitize.

    Returns:
        Messages with paired `AIMessage` tool calls and `ToolMessage` results.

    """
    if not messages:
        return messages

    # Collect `tool_call_id` values with `ToolMessage` responses.
    tool_message_ids: set[str] = {
        msg.tool_call_id for msg in messages if isinstance(msg, ToolMessage) and msg.tool_call_id
    }
    # Collect `tool_call_id` values requested by an `AIMessage`.
    ai_tool_call_ids: set[str] = {
        call.get("id")
        for msg in messages
        if isinstance(msg, AIMessage) and msg.tool_calls
        for call in msg.tool_calls
        if call.get("id")
    }

    sanitized_messages: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Remove tool calls without `ToolMessage` responses.
            incomplete_calls = [call for call in msg.tool_calls if call.get("id") not in tool_message_ids]
            if incomplete_calls:
                complete_calls = [call for call in msg.tool_calls if call.get("id") in tool_message_ids]
                sanitized_messages.append(
                    AIMessage(
                        content=msg.content or "[Tool call was interrupted]",
                        id=msg.id,
                        name=msg.name,
                        tool_calls=complete_calls,
                        response_metadata=msg.response_metadata,
                        usage_metadata=msg.usage_metadata,
                    )
                )
            else:
                sanitized_messages.append(msg)
        elif isinstance(msg, ToolMessage):
            # Remove results without requesting tool calls.
            if msg.tool_call_id and msg.tool_call_id in ai_tool_call_ids:
                sanitized_messages.append(msg)
        else:
            sanitized_messages.append(msg)

    return sanitized_messages


def create_ai_message(parts: dict) -> AIMessage:
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    filtered.setdefault("content", "")  # `AIMessage` requires content.
    return AIMessage(**filtered)


def read_file(file_path: Path | str, mode: str = "r", encoding: str = "utf-8", **kwargs) -> Any:
    """Read a file and return its content.

    Args:
        file_path (Path | str): File path.
        mode (str): File mode. The default is "r".
        encoding (str): File encoding. The default is "utf-8".
        **kwargs: Arguments for `open`.

    Returns:
        Any: File content.

    """
    with open(file_path, mode=mode, encoding=encoding, **kwargs) as file:
        return file.read()
