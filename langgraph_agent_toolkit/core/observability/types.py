from enum import StrEnum, auto
from typing import List, Literal, TypedDict, Union

from langchain_core.prompts import ChatPromptTemplate


class ObservabilityBackend(StrEnum):
    LANGFUSE = auto()
    LANGSMITH = auto()
    EMPTY = auto()


class MessageRole(StrEnum):
    """Enum for message roles in chat templates."""

    SYSTEM = "system"
    HUMAN = "human"
    USER = "user"
    AI = "ai"
    ASSISTANT = "assistant"
    PLACEHOLDER = "placeholder"
    MESSAGES_PLACEHOLDER = "messages_placeholder"


class ChatMessageDict(TypedDict):
    role: str  # Using str instead of Literal to allow compatibility with MessageRole
    content: str


# Type for prompt templates that can be provided to push_prompt
PromptTemplateType = Union[str, List[ChatMessageDict]]

# Type for the return value of pull_prompt
PromptReturnType = Union[ChatPromptTemplate, str, dict, None]
