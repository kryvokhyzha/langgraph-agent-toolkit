from enum import StrEnum, auto


class MemoryBackends(StrEnum):
    POSTGRES = auto()
    SQLITE = auto()
