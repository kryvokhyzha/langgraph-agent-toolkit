from enum import StrEnum, auto


class RunnerType(StrEnum):
    UVICORN = auto()
    GUNICORN = auto()
    AWS_LAMBDA = auto()
    AZURE_FUNCTIONS = auto()
