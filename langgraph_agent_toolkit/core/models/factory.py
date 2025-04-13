from functools import cache
from typing import TypeAlias

from langchain_core.runnables import ConfigurableField, RunnableSerializable

from langgraph_agent_toolkit.core.models import ChatOpenAIPatched, FakeToolModel
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.constants import DEFAULT_OPENAI_COMPATIBLE_MODEL_PARAMS
from langgraph_agent_toolkit.schema.models import (
    AllModelEnum,
    FakeModelName,
    OpenAICompatibleName,
)


ModelT: TypeAlias = ChatOpenAIPatched | FakeToolModel | RunnableSerializable


class ModelFactory:
    """Factory for creating model instances."""

    # Map model enum names to their respective API model names
    _MODEL_TABLE = {
        OpenAICompatibleName.OPENAI_COMPATIBLE: settings.COMPATIBLE_MODEL,
        FakeModelName.FAKE: "fake",
    }

    @staticmethod
    @cache
    def create(model_name: AllModelEnum) -> ModelT:
        """Create and return a model instance.

        Args:
            model_name: The model to create from AllModelEnum

        Returns:
            An instance of the requested model

        Raises:
            ValueError: If the requested model is not supported

        """
        api_model_name = ModelFactory._MODEL_TABLE.get(model_name)
        if not api_model_name:
            raise ValueError(f"Unsupported model: {model_name}")

        match model_name:
            case name if name in OpenAICompatibleName:
                if not settings.COMPATIBLE_BASE_URL or not settings.COMPATIBLE_MODEL:
                    raise ValueError("OpenAICompatible base url and endpoint must be configured")

                model = ChatOpenAIPatched(
                    model_name=settings.COMPATIBLE_MODEL,
                    openai_api_base=settings.COMPATIBLE_BASE_URL,
                    openai_api_key=settings.COMPATIBLE_API_KEY,
                    **DEFAULT_OPENAI_COMPATIBLE_MODEL_PARAMS,
                ).configurable_fields(
                    temperature=ConfigurableField(
                        id="temperature",
                        name="Agent Temperature",
                        description="The temperature to use. Default value is `0.0`.",
                    ),
                    max_tokens=ConfigurableField(
                        id="max_tokens",
                        name="Agent Max Tokens",
                        description="The maximum number of tokens to generate. Default value is `1500`.",
                    ),
                    top_p=ConfigurableField(
                        id="top_p",
                        name="Agent Top P",
                        description="The nucleus sampling probability. Default value is `0.7`.",
                    ),
                )

                return model
            case name if name in FakeModelName:
                return FakeToolModel(responses=["This is a test response from the fake model."])
            case _:
                raise ValueError(f"Unsupported model: {model_name}")
