import json
import os
from typing import Annotated, Any, Dict, Optional

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

    # OpenAI Settings
    OPENAI_API_KEY: SecretStr | None = None
    OPENAI_API_BASE_URL: str | None = None
    OPENAI_API_VERSION: str | None = None
    OPENAI_MODEL_NAME: str | None = None

    # Azure OpenAI Settings
    AZURE_OPENAI_API_KEY: SecretStr | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_API_VERSION: str | None = None
    AZURE_OPENAI_MODEL_NAME: str | None = None
    AZURE_OPENAI_DEPLOYMENT_NAME: str | None = None

    # Anthropic Settings
    ANTHROPIC_MODEL_NAME: str | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None

    # Google VertexAI Settings
    GOOGLE_VERTEXAI_MODEL_NAME: str | None = None
    GOOGLE_VERTEXAI_API_KEY: SecretStr | None = None

    # Google GenAI Settings
    GOOGLE_GENAI_MODEL_NAME: str | None = None
    GOOGLE_GENAI_API_KEY: SecretStr | None = None

    # Bedrock Settings
    AWS_BEDROCK_MODEL_NAME: str | None = None

    # DeepSeek Settings
    DEEPSEEK_MODEL_NAME: str | None = None
    DEEPSEEK_API_KEY: SecretStr | None = None

    # Ollama Settings
    OLLAMA_MODEL_NAME: str | None = None
    OLLAMA_BASE_URL: str | None = None

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

    # Model configurations dictionary
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

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

    def _initialize_model_configs(self) -> None:
        """Initialize model configurations from environment variables."""
        model_configs_env = os.environ.get("MODEL_CONFIGS")
        if model_configs_env:
            try:
                configs = json.loads(model_configs_env)
                if isinstance(configs, dict):
                    self.MODEL_CONFIGS = configs
                    logger.info(f"Loaded {len(configs)} model configurations from MODEL_CONFIGS")
                else:
                    logger.warning("MODEL_CONFIGS environment variable is not a valid JSON object")
            except json.JSONDecodeError:
                logger.error("Failed to parse MODEL_CONFIGS environment variable as JSON")

    def get_model_config(self, config_key: str) -> Optional[Dict[str, Any]]:
        """Get a model configuration by key.

        Args:
            config_key: The key of the model configuration to get

        Returns:
            The model configuration dict if found, None otherwise

        """
        return self.MODEL_CONFIGS.get(config_key)

    def setup(self) -> None:
        """Initialize all settings."""
        self._apply_langgraph_env_overrides()
        self._initialize_model_configs()

    @computed_field
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"

    def is_dev(self) -> bool:
        return self.ENV_MODE == "development"


settings = Settings()
settings.setup()
