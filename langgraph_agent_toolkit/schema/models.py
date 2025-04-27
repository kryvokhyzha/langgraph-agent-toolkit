from enum import StrEnum, auto


class ModelProvider(StrEnum):
    OPENAI = auto()
    AZURE_OPENAI = auto()
    ANTHROPIC = auto()
    GOOGLE_VERTEXAI = auto()
    GOOGLE_GENAI = auto()
    BEDROCK = auto()
    DEEPSEEK = auto()
    OLLAMA = auto()
    FAKE = auto()
