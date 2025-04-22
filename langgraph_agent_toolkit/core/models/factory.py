import warnings
from functools import cache
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

from langchain.chat_models.base import _ConfigurableModel, _init_chat_model_helper
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable
from typing_extensions import TypeAlias

from langgraph_agent_toolkit.core.models import ChatOpenAIPatched, FakeToolModel
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.constants import DEFAULT_OPENAI_MODEL_TYPE_PARAMS
from langgraph_agent_toolkit.schema.models import (
    AllModelEnum,
    FakeModelName,
    OpenAICompatibleName,
)


ModelT: TypeAlias = ChatOpenAIPatched | FakeToolModel | RunnableSerializable | _ConfigurableModel


class ModelFactory:
    """Factory for creating model instances."""

    # Map model enum names to their respective API model names
    _MODEL_TABLE = {
        OpenAICompatibleName.OPENAI_COMPATIBLE: settings.MODEL_NAME,
        FakeModelName.FAKE: "fake",
    }

    @staticmethod
    def __init_chat_model_helper(model: str, *, model_provider: Optional[str] = None, **kwargs: Any) -> BaseChatModel:
        if model_provider == "openai":
            return ChatOpenAIPatched(model_name=model, **kwargs)
        else:
            return _init_chat_model_helper(model, model_provider=model_provider, **kwargs)

    @staticmethod
    def init_chat_model(
        model: Optional[str] = None,
        *,
        model_provider: Optional[str] = None,
        configurable_fields: Optional[Union[Literal["any"], List[str], Tuple[str, ...]]] = None,
        config_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[BaseChatModel, _ConfigurableModel]:
        if not model and not configurable_fields:
            configurable_fields = ("model", "model_provider")
        config_prefix = config_prefix or ""

        if config_prefix and not configurable_fields:
            warnings.warn(
                f"{config_prefix=} has been set but no fields are configurable. Set "
                f"`configurable_fields=(...)` to specify the model params that are "
                f"configurable."
            )

        if not configurable_fields:
            return ModelFactory.__init_chat_model_helper(cast(str, model), model_provider=model_provider, **kwargs)
        else:
            if model:
                kwargs["model"] = model
            if model_provider:
                kwargs["model_provider"] = model_provider
            return _ConfigurableModel(
                default_config=kwargs,
                config_prefix=config_prefix,
                configurable_fields=configurable_fields,
            )

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
                if not settings.MODEL_BASE_URL or not settings.MODEL_NAME:
                    raise ValueError("OpenAICompatible base url and endpoint must be configured")

                model = ModelFactory.init_chat_model(
                    model=settings.MODEL_NAME,
                    model_provider="openai",
                    configurable_fields=("temperature", "max_tokens", "top_p", "streaming"),
                    config_prefix="agent",
                    openai_api_base=settings.MODEL_BASE_URL,
                    openai_api_key=settings.MODEL_API_KEY,
                    **DEFAULT_OPENAI_MODEL_TYPE_PARAMS,
                )

                return model
            case name if name in FakeModelName:
                return FakeToolModel(responses=["This is a test response from the fake model."])
            case _:
                raise ValueError(f"Unsupported model: {model_name}")
