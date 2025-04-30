import warnings
from functools import cache
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

from langchain.chat_models.base import _ConfigurableModel, _init_chat_model_helper
from langchain_core.language_models import BaseChatModel
from typing_extensions import TypeAlias

from langgraph_agent_toolkit.core.models import ChatOpenAIPatched, FakeToolModel
from langgraph_agent_toolkit.helper.constants import (
    DEFAULT_CONFIG_PREFIX,
    DEFAULT_CONFIGURABLE_FIELDS,
    DEFAULT_MODEL_PARAMETER_VALUES,
)
from langgraph_agent_toolkit.schema.models import ModelProvider


ModelT: TypeAlias = FakeToolModel | _ConfigurableModel | BaseChatModel


class ModelFactory:
    """Factory for creating model instances."""

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
    def create(
        model_provider: ModelProvider,
        model_name: Optional[str] = None,
        configurable_fields: Optional[Union[Literal["any"], List[str], Tuple[str, ...]]] = None,
        config_prefix: Optional[str] = None,
        model_parameter_values: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ModelT:
        """Create and return a model instance.

        Args:
            model_provider: The model provider to use. This should be one of the supported model providers.
            model_name: The name of the model to use. If not provided, the default model name will be used.
            configurable_fields: The fields that are configurable. If not provided, the default fields will be used.
            config_prefix: The prefix to use for the configuration. If not provided, the default prefix will be used.
            model_parameter_values: The values for the model parameters. If not provided, the default values will be used.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            An instance of the requested model

        Raises:
            ValueError: If the requested model is not supported

        """  # noqa: E501
        _configurable_fields = configurable_fields or DEFAULT_CONFIGURABLE_FIELDS
        _config_prefix = config_prefix or DEFAULT_CONFIG_PREFIX
        _model_parameter_values = model_parameter_values or DEFAULT_MODEL_PARAMETER_VALUES

        match model_provider:
            case ModelProvider.FAKE:
                return FakeToolModel(responses=["This is a test response from the fake model."])
            case _:
                if model_name is None:
                    raise ValueError("Model name must be provided for non-fake models.")

                return ModelFactory.init_chat_model(
                    model=model_name,
                    model_provider=model_provider,
                    configurable_fields=_configurable_fields,
                    config_prefix=_config_prefix,
                    **_model_parameter_values,
                    **kwargs,
                )

    @classmethod
    def get_model_from_config(cls, config: Dict[str, Any], **override_params) -> BaseChatModel:
        """Create a model from a configuration dictionary.

        Args:
            config: Model configuration dictionary
            **override_params: Parameters to override from the configuration

        Returns:
            A BaseChatModel instance

        Example:
            >>> config = {"provider": "openai", "name": "gpt-4", "temperature": 0.7}
            >>> model = ModelFactory.get_model_from_config(config)

        """
        if not config:
            raise ValueError("Model configuration cannot be empty")

        # Extract basic parameters
        provider = config.get("provider", "openai")
        model_name = config.get("name") or config.get("model_name")

        if not model_name:
            raise ValueError("Model name must be specified in the configuration")

        # Copy the config to avoid modifying the original
        params = dict(config)

        # Remove some keys that are handled separately
        params.pop("provider", None)
        params.pop("name", None)
        params.pop("model_name", None)

        # Apply overrides
        params.update(override_params)

        # Create and return the model
        return cls.create(model_provider=provider, model_name=model_name, **params)
