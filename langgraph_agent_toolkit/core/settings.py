import json
import os
from typing import Annotated, Any

from dotenv import find_dotenv
from pydantic import (
    BeforeValidator,
    Field,
    SecretStr,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from langgraph_agent_toolkit.core.memory.types import MemoryBackends
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend
from langgraph_agent_toolkit.helper.logging import logger
from langgraph_agent_toolkit.helper.utils import check_str_is_http
from langgraph_agent_toolkit.schema.models import (
    AllModelEnum,
    FakeModelName,
    OpenAICompatibleName,
    Provider,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    ENV_MODE: str | None = None

    HOST: str = "0.0.0.0"
    PORT: int = 8080

    AUTH_SECRET: SecretStr | None = None
    USE_FAKE_MODEL: bool = False

    # If DEFAULT_MODEL_TYPE is None, it will be set in model_post_init
    DEFAULT_MODEL_TYPE: AllModelEnum | None = None
    AVAILABLE_MODELS: set[AllModelEnum] = Field(
        ...,
        description="Set of available models. If not set, all models will be available.",
        default_factory=set,
    )

    # Set openai compatible api, mainly used for proof of concept
    MODEL_NAME: str | None = None
    MODEL_API_KEY: SecretStr | None = None
    MODEL_BASE_URL: str | None = None

    # Observability platform
    OBSERVABILITY_BACKEND: ObservabilityBackend | None = None

    # Agent configuration
    AGENT_PATHS: list[str] = [
        "langgraph_agent_toolkit.agents.blueprints.react.agent:react_agent",
        "langgraph_agent_toolkit.agents.blueprints.chatbot.agent:chatbot_agent",
    ]

    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "default"
    LANGCHAIN_ENDPOINT: Annotated[str, BeforeValidator(check_str_is_http)] = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: SecretStr | None = None

    LANGFUSE_SECRET_KEY: SecretStr | None = None
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_HOST: Annotated[str, BeforeValidator(check_str_is_http)] = "http://localhost:3000"

    # Database Configuration
    MEMORY_BACKEND: MemoryBackends = MemoryBackends.SQLITE
    SQLITE_DB_PATH: str = "checkpoints.db"

    # postgresql Configuration
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int | None = None
    POSTGRES_DB: str | None = None
    POSTGRES_POOL_SIZE: int = Field(default=10, description="Maximum number of connections in the pool")
    POSTGRES_MIN_SIZE: int = Field(default=3, description="Minimum number of connections in the pool")
    POSTGRES_MAX_IDLE: int = Field(default=5, description="Maximum number of idle connections")

    def model_post_init(self, __context: Any) -> None:
        # Check for LANGGRAPH_ prefixed environment variables that might override settings
        self._apply_langgraph_env_overrides()

        api_keys = {
            Provider.OPENAI_COMPATIBLE: self.MODEL_BASE_URL and self.MODEL_NAME,
            Provider.FAKE: self.USE_FAKE_MODEL,
        }
        active_keys = [k for k, v in api_keys.items() if v]
        if not active_keys:
            raise ValueError("At least one LLM API key must be provided.")

        for provider in active_keys:
            match provider:
                case Provider.OPENAI_COMPATIBLE:
                    if self.DEFAULT_MODEL_TYPE is None:
                        self.DEFAULT_MODEL_TYPE = OpenAICompatibleName.OPENAI_COMPATIBLE
                    self.AVAILABLE_MODELS.update(set(OpenAICompatibleName))
                case Provider.FAKE:
                    if self.DEFAULT_MODEL_TYPE is None:
                        self.DEFAULT_MODEL_TYPE = FakeModelName.FAKE
                    self.AVAILABLE_MODELS.update(set(FakeModelName))
                case _:
                    raise ValueError(f"Unknown provider: {provider}")

    def _apply_langgraph_env_overrides(self) -> None:
        """Apply any LANGGRAPH_ prefixed environment variables to override settings."""
        for env_name, env_value in os.environ.items():
            if env_name.startswith("LANGGRAPH_"):
                setting_name = env_name[10:]  # Remove the "LANGGRAPH_" prefix
                if hasattr(self, setting_name):
                    try:
                        current_value = getattr(self, setting_name)

                        # Handle different types
                        if isinstance(current_value, list):
                            # Parse JSON array
                            try:
                                parsed_value = json.loads(env_value)
                                if isinstance(parsed_value, list):
                                    setattr(self, setting_name, parsed_value)
                                    logger.debug(f"Applied environment override for {setting_name}")
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse JSON for {setting_name}: {env_value}")
                        elif isinstance(current_value, bool):
                            # Convert string to boolean
                            if env_value.lower() in ("true", "1", "yes"):
                                setattr(self, setting_name, True)
                                logger.debug(f"Applied environment override for {setting_name}")
                            elif env_value.lower() in ("false", "0", "no"):
                                setattr(self, setting_name, False)
                                logger.debug(f"Applied environment override for {setting_name}")
                        elif current_value is None or isinstance(current_value, (str, int, float)):
                            # Convert to the appropriate type
                            if isinstance(current_value, int) or current_value is None and env_value.isdigit():
                                setattr(self, setting_name, int(env_value))
                            elif isinstance(current_value, float) or current_value is None and "." in env_value:
                                try:
                                    setattr(self, setting_name, float(env_value))
                                except ValueError:
                                    setattr(self, setting_name, env_value)
                            else:
                                setattr(self, setting_name, env_value)
                            logger.debug(f"Applied environment override for {setting_name}")
                        # Add more type handling as needed
                    except Exception as e:
                        logger.warning(f"Failed to apply environment override for {setting_name}: {e}")

    @computed_field
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"

    def is_dev(self) -> bool:
        return self.ENV_MODE == "development"


settings = Settings()
