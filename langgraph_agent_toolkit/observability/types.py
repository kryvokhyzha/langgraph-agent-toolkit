from enum import StrEnum, auto


class ObservabilityBackend(StrEnum):
    LANGFUSE = auto()
    LANGSMITH = auto()
    EMPTY = auto()
