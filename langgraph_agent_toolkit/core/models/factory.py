import warnings
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
from langchain.embeddings.base import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypeAlias

from langgraph_agent_toolkit.core.models.chat_openai import ChatOpenAIPatched
from langgraph_agent_toolkit.core.models.fake import FakeToolModel
from langgraph_agent_toolkit.helper.constants import (
    DEFAULT_CONFIG_PREFIX,
    DEFAULT_CONFIGURABLE_FIELDS,
    DEFAULT_MODEL_PARAMETER_VALUES,
)
from langgraph_agent_toolkit.schema.models import ModelProvider


ModelT: TypeAlias = FakeToolModel | _ConfigurableModel | BaseChatModel


class _ConfigurableModelCustom(_ConfigurableModel):
    def _model(self, config: Optional[RunnableConfig] = None) -> Runnable:
        params = {**self._default_config, **self._model_params(config)}
        model = CompletionModelFactory._init_chat_model_helper(**params)
        for name, args, kwargs in self._queued_declarative_operations:
            model = getattr(model, name)(*args, **kwargs)
        return model


class CompletionModelFactory:
    """Factory for creating model instances."""

    @staticmethod
    def _init_chat_model_helper(model: str, *, model_provider: Optional[str] = None, **kwargs: Any) -> BaseChatModel:
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
            return CompletionModelFactory._init_chat_model_helper(
                cast(str, model),
                model_provider=model_provider,
                **kwargs,
            )
        else:
            if model:
                kwargs["model"] = model
            if model_provider:
                kwargs["model_provider"] = model_provider
            return _ConfigurableModelCustom(
                default_config=kwargs,
                config_prefix=config_prefix,
                configurable_fields=configurable_fields,
            )

    @staticmethod
    def create(
        model_provider: ModelProvider,
        model_name: Optional[str] = None,
        configurable_fields: Optional[Union[Literal["any"], List[str], Tuple[str, ...]]] = None,
        config_prefix: Optional[str] = None,
        model_parameter_values: Optional[Tuple[Tuple[str, Any], ...]] = None,
        **kwargs: Any,
    ) -> ModelT:
        """Create and return a model instance.

        Args:
            model_provider: The model provider to use. This should be one of the supported model providers.
            model_name: The name of the model to use. If not provided, the default model name will be used.
            configurable_fields: The fields that are configurable. If not provided, the default fields will be used.
            config_prefix: The prefix to use for the configuration. If not provided, the default prefix will be used.
            model_parameter_values: The values for the model parameters as a tuple of (key, value) pairs.
                                    If not provided, the default values will be used.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            An instance of the requested model

        Raises:
            ValueError: If the requested model is not supported

        """  # noqa: E501
        _configurable_fields = DEFAULT_CONFIGURABLE_FIELDS if configurable_fields is None else configurable_fields
        _config_prefix = DEFAULT_CONFIG_PREFIX if config_prefix is None else config_prefix
        _model_parameter_values = (
            DEFAULT_MODEL_PARAMETER_VALUES if model_parameter_values is None else dict(model_parameter_values)
        )

        _model_parameter_values.update(kwargs)

        match model_provider:
            case ModelProvider.FAKE:
                return FakeToolModel(responses=["This is a test response from the fake model."])
            case _:
                if model_name is None:
                    raise ValueError("Model name must be provided for non-fake models.")

                return CompletionModelFactory.init_chat_model(
                    model=model_name,
                    model_provider=model_provider,
                    configurable_fields=_configurable_fields,
                    config_prefix=_config_prefix,
                    **_model_parameter_values,
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
            >>> model = CompletionModelFactory.get_model_from_config(config)

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

        if "model_parameter_values" not in params:
            params["model_parameter_values"] = ()

        # Create and return the model
        return cls.create(model_provider=provider, model_name=model_name, **params)


class EmbeddingModelFactory:
    """Factory for creating embedding model instances."""

    @staticmethod
    def create(
        model_provider: ModelProvider,
        model_name: Optional[str] = None,
        model_parameter_values: Optional[Tuple[Tuple[str, Any], ...]] = None,
        **kwargs: Any,
    ) -> Embeddings:
        """Create and return an embedding model instance.

        Args:
            model_provider: The model provider to use. This should be one of the supported model providers.
            model_name: The name of the model to use. If not provided, an error will be raised.
            model_parameter_values: The values for the model parameters as a tuple of (key, value) pairs.
                                    If not provided, empty dict will be used.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            An instance of the requested embedding model

        Raises:
            ValueError: If the requested model is not supported or model_name is not provided

        Examples:
            >>> model = EmbeddingModelFactory.create(
            ...     model_provider=ModelProvider.OPENAI,
            ...     model_name="text-embedding-3-small",
            ...     openai_api_key="sk-..."
            ... )

        """
        if model_name is None:
            raise ValueError("Model name must be provided for embedding models.")

        _model_parameter_values = {} if model_parameter_values is None else dict(model_parameter_values)
        _model_parameter_values.update(kwargs)

        # Get provider string from enum if needed
        provider_str = model_provider.value if isinstance(model_provider, ModelProvider) else str(model_provider)

        return init_embeddings(
            model=model_name,
            provider=provider_str,
            **_model_parameter_values,
        )

    @classmethod
    def get_model_from_config(cls, config: Dict[str, Any], **override_params) -> Embeddings:
        """Create an embedding model from a configuration dictionary.

        Args:
            config: Model configuration dictionary
            **override_params: Parameters to override from the configuration

        Returns:
            An Embeddings instance

        Example:
            >>> config = {"provider": "openai", "name": "text-embedding-3-small", "api_key": "sk-..."}
            >>> model = EmbeddingModelFactory.get_model_from_config(config)

        """
        if not config:
            raise ValueError("Model configuration cannot be empty")

        # Extract basic parameters
        provider = config.get("provider", "openai")
        model_name = config.get("name") or config.get("model_name") or config.get("model")

        if not model_name:
            raise ValueError("Model name must be specified in the configuration")

        # Copy the config to avoid modifying the original
        params = dict(config)

        # Remove some keys that are handled separately
        params.pop("provider", None)
        params.pop("name", None)
        params.pop("model_name", None)
        params.pop("model", None)

        # Apply overrides
        params.update(override_params)

        if "model_parameter_values" not in params:
            params["model_parameter_values"] = ()

        # Create and return the model
        return cls.create(model_provider=provider, model_name=model_name, **params)
