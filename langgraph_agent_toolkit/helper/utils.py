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
from pydantic import BaseModel, HttpUrl, TypeAdapter

from langgraph_agent_toolkit.helper.exceptions import UnsupportedMessageTypeError
from langgraph_agent_toolkit.schema import ChatMessage


def check_str_is_http(x: str) -> str:
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))


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
            ai_message = ChatMessage(
                type="ai",
                content=message.content,
            )
            if message.tool_calls:
                ai_message.tool_calls = message.tool_calls
            if message.response_metadata:
                ai_message.response_metadata = message.response_metadata
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
    # Currently only Anthropic models stream tool calls, using content item type tool_use.
    return [
        content_item for content_item in content if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]


def sanitize_chat_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Sanitize chat history by removing AIMessages with tool_calls that don't have corresponding ToolMessages.

    This is useful when:
    - A previous execution was interrupted before tool responses were added
    - Chat history was corrupted
    - Connection was lost during tool execution

    Args:
        messages: List of messages to sanitize

    Returns:
        Sanitized list of messages where all AIMessages with tool_calls have corresponding ToolMessages

    """
    if not messages:
        return messages

    # Collect all tool_call_ids that have corresponding ToolMessages
    tool_message_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id:
            tool_message_ids.add(msg.tool_call_id)

    # Process messages and fix incomplete tool calls
    sanitized_messages: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Check if all tool_calls have corresponding ToolMessages
            incomplete_calls = [call for call in msg.tool_calls if call.get("id") not in tool_message_ids]
            if incomplete_calls:
                # Create a new AIMessage without the incomplete tool_calls
                complete_calls = [call for call in msg.tool_calls if call.get("id") in tool_message_ids]
                # If no complete calls remain, remove tool_calls entirely
                new_msg = AIMessage(
                    content=msg.content or "[Tool call was interrupted]",
                    id=msg.id,
                    name=msg.name,
                    tool_calls=complete_calls if complete_calls else [],
                    response_metadata=msg.response_metadata,
                )
                sanitized_messages.append(new_msg)
            else:
                sanitized_messages.append(msg)
        else:
            sanitized_messages.append(msg)

    return sanitized_messages


def create_ai_message(parts: dict) -> AIMessage:
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)


def read_file(file_path: Path | str, mode: str = "r", encoding: str = "utf-8", **kwargs) -> Any:
    """Read the content of a file and return it as a string.

    Args:
        file_path (Path | str): The path to the file to read.
        mode (str): The mode in which to open the file. Default is "r".
        encoding (str): The encoding to use for reading the file. Default is "utf-8".
        **kwargs: Additional arguments to pass to the open function.

    Returns:
        Any: The content of the file as a string.

    """
    with open(file_path, mode=mode, encoding=encoding, **kwargs) as file:
        return file.read()
