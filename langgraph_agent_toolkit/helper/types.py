from enum import StrEnum, auto


class EnvironmentMode(StrEnum):
    PRODUCTION = auto()
    STAGING = auto()
    DEVELOPMENT = auto()
