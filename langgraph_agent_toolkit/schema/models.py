from enum import StrEnum, auto
from typing import TypeAlias


class Provider(StrEnum):
    OPENAI_COMPATIBLE = auto()
    FAKE = auto()


class OpenAICompatibleName(StrEnum):
    """OpenAI compatible model names.

    https://platform.openai.com/docs/guides/text-generation
    """

    OPENAI_COMPATIBLE = "openai-compatible"


class FakeModelName(StrEnum):
    """Fake model for testing."""

    FAKE = "fake"


AllModelEnum: TypeAlias = OpenAICompatibleName | FakeModelName
