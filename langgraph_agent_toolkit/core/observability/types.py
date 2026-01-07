from enum import StrEnum, auto
from typing import TypedDict


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
