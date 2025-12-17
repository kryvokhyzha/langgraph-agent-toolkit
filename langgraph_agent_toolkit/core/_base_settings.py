import base64
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
from langgraph_agent_toolkit.helper.types import EnvironmentMode
from langgraph_agent_toolkit.helper.utils import check_str_is_http


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    ENV_MODE: EnvironmentMode = EnvironmentMode.PRODUCTION

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

    # OpenRouter Settings
    OPENROUTER_API_KEY: SecretStr | None = None

    # Observability platform
    OBSERVABILITY_BACKEND: ObservabilityBackend | None = None

    # Agent configuration
    AGENT_PATHS: list[str] = [
        "langgraph_agent_toolkit.agents.blueprints.react.agent:react_agent",
        "langgraph_agent_toolkit.agents.blueprints.chatbot.agent:chatbot_agent",
        "langgraph_agent_toolkit.agents.blueprints.react_so.agent:react_agent_so",
    ]

    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "default"
    LANGCHAIN_ENDPOINT: Annotated[str, BeforeValidator(check_str_is_http)] = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: SecretStr | None = None

    LANGFUSE_SECRET_KEY: SecretStr | None = None
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_HOST: Annotated[str, BeforeValidator(check_str_is_http)] = "https://cloud.langfuse.com"
    LANGFUSE_TRACING_ENVIRONMENT: str | None = None

    # Database Configuration
    MEMORY_BACKEND: MemoryBackends | None = None
    SQLITE_DB_PATH: str = "checkpoints.db"

    # postgresql Configuration
    POSTGRES_APPLICATION_NAME: str = "langgraph-agent-toolkit"
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int | None = None
    POSTGRES_DB: str | None = None
    POSTGRES_SCHEMA: str = "public"
    POSTGRES_POOL_SIZE: int = Field(default=100, description="Maximum number of connections in the pool")
    POSTGRES_MIN_SIZE: int = Field(default=5, description="Minimum number of connections in the pool")
    POSTGRES_MAX_IDLE: int = Field(default=120, description="Maximum number of idle connections")
    POSTGRES_POOL_TIMEOUT: float = Field(default=45.0, description="Timeout in seconds to get a connection from pool")
    POSTGRES_RECONNECT_TIMEOUT: float = Field(default=60.0, description="Timeout for reconnecting to database")
    POSTGRES_MAX_LIFETIME: float = Field(
        default=300.0,
        description="Maximum lifetime of a connection in seconds. After this time, the connection will be closed "
        "and replaced with a new one. Set to 0 to disable. Helps prevent stale connections.",
    )
    POSTGRES_NUM_WORKERS: int = Field(
        default=3,
        description="Number of background workers for pool maintenance (creating/closing connections)",
    )
    POSTGRES_STATEMENT_TIMEOUT: int = Field(
        default=120000,
        description="Maximum time in milliseconds a query can run before being cancelled. "
        "Prevents stuck queries from blocking connections forever. Set to 0 to disable.",
    )
    POSTGRES_LOCK_TIMEOUT: int = Field(
        default=45000,
        description="Maximum time in milliseconds to wait for a lock before giving up. "
        "Prevents deadlocks from blocking connections. Set to 0 to disable.",
    )
    POSTGRES_IDLE_IN_TRANSACTION_SESSION_TIMEOUT: int = Field(
        default=120000,
        description="Maximum time in milliseconds a connection can stay idle in transaction. "
        "Terminates connections that started a transaction but didn't finish. Set to 0 to disable.",
    )

    # Model configurations dictionary
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    MODEL_CONFIGS_BASE64: str | None = None
    MODEL_CONFIGS_PATH: str | None = None

    # Database configurations dictionary
    DB_CONFIGS: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    DB_CONFIGS_BASE64: str | None = None
    DB_CONFIGS_PATH: str | None = None

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

    def _initialize_configs(self, config_type: str) -> Dict[str, Dict[str, Any]]:
        """Initialize configurations from environment variables.

        Args:
            config_type: Type of configuration ('MODEL' or 'DB')

        Returns:
            Dictionary of configurations

        """
        configs = {}

        # Try direct JSON environment variable
        configs_env = os.environ.get(f"{config_type}_CONFIGS")
        if configs_env:
            try:
                parsed_configs = json.loads(configs_env)
                if isinstance(parsed_configs, dict):
                    configs = parsed_configs
                    logger.info(
                        f"Loaded {len(configs)} {config_type.lower()} configurations from {config_type}_CONFIGS"
                    )
                else:
                    logger.warning(f"{config_type}_CONFIGS environment variable is not a valid JSON object")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse {config_type}_CONFIGS environment variable as JSON")
            return configs

        # Try base64 encoded environment variable
        configs_base64_env = os.environ.get(f"{config_type}_CONFIGS_BASE64")
        if configs_base64_env:
            try:
                decoded_configs = base64.b64decode(configs_base64_env).decode("utf-8")
                parsed_configs = json.loads(decoded_configs)
                if isinstance(parsed_configs, dict):
                    configs = parsed_configs
                    logger.info(
                        f"Loaded {len(configs)} {config_type.lower()} configurations from {config_type}_CONFIGS_BASE64"
                    )
                else:
                    logger.warning(f"{config_type}_CONFIGS_BASE64 cannot be parsed as a valid JSON object")
            except (ValueError, json.JSONDecodeError) as e:
                logger.error(f"Failed to decode {config_type}_CONFIGS_BASE64: {e}")
            return configs

        # Try file path
        configs_path_env = os.environ.get(f"{config_type}_CONFIGS_PATH")
        if configs_path_env:
            try:
                with open(configs_path_env, "r", encoding="utf-8") as f:
                    parsed_configs = json.load(f)
                    if isinstance(parsed_configs, dict):
                        configs = parsed_configs
                        logger.info(
                            f"Loaded {len(configs)} {config_type.lower()} configurations from {configs_path_env}"
                        )
                    else:
                        logger.warning(f"{config_type}_CONFIGS_PATH cannot be parsed as a valid JSON object")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error(f"Failed to load {config_type.lower()} configurations from {configs_path_env}: {e}")
            return configs

        logger.info(f"No {config_type}_CONFIGS found in environment variables or file")
        return configs

    def _initialize_model_configs(self) -> None:
        """Initialize model configurations from environment variables."""
        self.MODEL_CONFIGS = self._initialize_configs("MODEL")

    def _initialize_db_configs(self) -> None:
        """Initialize database configurations from environment variables."""
        self.DB_CONFIGS = self._initialize_configs("DB")

    def get_model_config(self, config_key: str) -> Optional[Dict[str, Any]]:
        """Get a model configuration by key.

        Args:
            config_key: The key of the model configuration to get

        Returns:
            The model configuration dict if found, None otherwise

        """
        return self.MODEL_CONFIGS.get(config_key)

    def get_db_config(self, config_key: str) -> Optional[Dict[str, Any]]:
        """Get a database configuration by key.

        Args:
            config_key: The key of the database configuration to get

        Returns:
            The database configuration dict if found, None otherwise

        """
        return self.DB_CONFIGS.get(config_key)

    def setup(self) -> None:
        """Initialize all settings."""
        self._apply_langgraph_env_overrides()
        self._initialize_model_configs()
        self._initialize_db_configs()

        # Set LANGFUSE_TRACING_ENVIRONMENT to ENV_MODE if not explicitly set
        if self.LANGFUSE_TRACING_ENVIRONMENT is None:
            self.LANGFUSE_TRACING_ENVIRONMENT = self.ENV_MODE.value
            os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = self.LANGFUSE_TRACING_ENVIRONMENT

    @computed_field
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"

    def is_dev(self) -> bool:
        return self.ENV_MODE == "development"
