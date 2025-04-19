from enum import StrEnum, auto
from typing import List, TypedDict, Union

from langchain_core.prompts import ChatPromptTemplate


class ObservabilityBackend(StrEnum):
    LANGFUSE = auto()
    LANGSMITH = auto()
    EMPTY = auto()


class MessageRole(StrEnum):
    """Enum for message roles in chat templates."""

    SYSTEM = auto()
    HUMAN = auto()
    USER = auto()
    AI = auto()
    ASSISTANT = auto()
    PLACEHOLDER = auto()
    MESSAGES_PLACEHOLDER = auto()


class ChatMessageDict(TypedDict):
    role: str
    content: str


# Type for prompt templates that can be provided to push_prompt
PromptTemplateType = Union[str, List[ChatMessageDict]]

# Type for the return value of pull_prompt
PromptReturnType = Union[ChatPromptTemplate, str, dict, None]
