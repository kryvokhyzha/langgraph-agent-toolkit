import base64
import json
import os
from typing import Annotated, Any, Dict, Literal, Mapping, Optional

from dotenv import find_dotenv
from pydantic import (
    BeforeValidator,
    Field,
    HttpUrl,
    SecretStr,
    TypeAdapter,
    ValidationError,
    computed_field,
)
from pydantic_settings import BaseSettings, EnvSettingsSource, SettingsConfigDict

from langgraph_agent_toolkit.core.mcp import MCPServerConfig, ServerName
from langgraph_agent_toolkit.core.memory.types import MemoryBackends
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend
from langgraph_agent_toolkit.helper.logging import logger
from langgraph_agent_toolkit.helper.types import EnvironmentMode


def check_str_is_http(x: str) -> str:
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))


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
    AUTH_MODE: Literal["token", "trusted"] = "trusted"
    AUTH_USERS: dict[str, SecretStr] = Field(default_factory=dict)
    AUTH_SERVICE_USER_ID: str = "service"
    FEEDBACK_SIGNING_SECRET: SecretStr | None = Field(default=None, min_length=32)
    USE_FAKE_MODEL: bool = False

    # OpenAI settings.
    OPENAI_API_KEY: SecretStr | None = None
    OPENAI_API_BASE_URL: str | None = None
    OPENAI_API_VERSION: str | None = None
    OPENAI_MODEL_NAME: str | None = None

    # Per-worker HTTP pools for OpenAI and Azure models created by the factory.
    LLM_HTTP_ASYNC_TRANSPORT: Literal["httpx", "aiohttp"] = "httpx"
    # Match OpenAI SDK 3.13.0 defaults without importing this optional dependency.
    LLM_HTTP_MAX_CONNECTIONS: int = Field(default=1000, gt=0)
    LLM_HTTP_MAX_KEEPALIVE_CONNECTIONS: int = Field(default=100, ge=0)
    LLM_HTTP_KEEPALIVE_EXPIRY: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    LLM_HTTP_CONNECT_TIMEOUT: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    LLM_HTTP_READ_TIMEOUT: float = Field(default=600.0, gt=0, allow_inf_nan=False)
    LLM_HTTP_WRITE_TIMEOUT: float = Field(default=600.0, gt=0, allow_inf_nan=False)
    LLM_HTTP_POOL_TIMEOUT: float = Field(default=600.0, gt=0, allow_inf_nan=False)
    LLM_HTTP_MAX_RETRIES: int = Field(default=2, ge=0)
    LLM_HTTP_SHUTDOWN_TIMEOUT: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    LLM_HTTP_MAX_POOLS: int = Field(default=32, gt=0)

    # Azure OpenAI settings.
    AZURE_OPENAI_API_KEY: SecretStr | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_API_VERSION: str | None = None
    AZURE_OPENAI_MODEL_NAME: str | None = None
    AZURE_OPENAI_DEPLOYMENT_NAME: str | None = None

    # Anthropic settings.
    ANTHROPIC_MODEL_NAME: str | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None

    # Google VertexAI settings.
    GOOGLE_VERTEXAI_MODEL_NAME: str | None = None
    GOOGLE_VERTEXAI_API_KEY: SecretStr | None = None

    # Google GenAI settings.
    GOOGLE_GENAI_MODEL_NAME: str | None = None
    GOOGLE_GENAI_API_KEY: SecretStr | None = None

    # Bedrock settings.
    AWS_BEDROCK_MODEL_NAME: str | None = None

    # DeepSeek settings.
    DEEPSEEK_MODEL_NAME: str | None = None
    DEEPSEEK_API_KEY: SecretStr | None = None

    # Ollama settings.
    OLLAMA_MODEL_NAME: str | None = None
    OLLAMA_BASE_URL: str | None = None

    # OpenRouter settings.
    OPENROUTER_API_KEY: SecretStr | None = None

    # Observability platform.
    OBSERVABILITY_BACKEND: ObservabilityBackend | None = None
    OBSERVABILITY_SHUTDOWN_TIMEOUT: float = Field(default=10.0, gt=0)

    # Agent configuration.
    AGENT_PATHS: list[str] = [
        "langgraph_agent_toolkit.agents.blueprints.react.agent:react_agent",
        "langgraph_agent_toolkit.agents.blueprints.chatbot.agent:chatbot_agent",
        "langgraph_agent_toolkit.agents.blueprints.create_agent.agent:react_agent",
        "langgraph_agent_toolkit.agents.blueprints.create_agent_structured.agent:react_agent_so",
        "langgraph_agent_toolkit.agents.blueprints.interrupt_agent.agent:interrupt_agent",
        "langgraph_agent_toolkit.agents.blueprints.hitl_agent.agent:hitl_agent",
    ]

    MCP_SERVERS: dict[ServerName, MCPServerConfig] = Field(default_factory=dict)
    MCP_AGENT_SERVERS: dict[str, list[ServerName]] = Field(default_factory=dict)
    MCP_DISCOVERY_TIMEOUT: float = Field(default=30.0, gt=0, allow_inf_nan=False)

    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "default"
    LANGCHAIN_ENDPOINT: Annotated[str, BeforeValidator(check_str_is_http)] = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: SecretStr | None = None

    LANGFUSE_SECRET_KEY: SecretStr | None = None
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_HOST: Annotated[str, BeforeValidator(check_str_is_http)] = "https://cloud.langfuse.com"
    LANGFUSE_TRACING_ENVIRONMENT: str | None = None
    LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS: int = 60 * 60
    LANGFUSE_FLUSH_AT: int = 512
    LANGFUSE_FLUSH_INTERVAL: float = 5.0
    LANGFUSE_TIMEOUT: int = 5
    LANGFUSE_DEBUG: bool = False
    LANGFUSE_SAMPLE_RATE: float = 1.0

    # Database configuration.
    MEMORY_BACKEND: MemoryBackends | None = None
    SQLITE_DB_PATH: str = "checkpoints.db"

    # PostgreSQL configuration.
    POSTGRES_APPLICATION_NAME: str = "langgraph-agent-toolkit"
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int | None = None
    POSTGRES_DB: str | None = None
    POSTGRES_SCHEMA: str = "public"
    POSTGRES_POOL_SIZE: int = Field(default=20, ge=1, description="Maximum number of connections in the pool")
    POSTGRES_MIN_SIZE: int = Field(default=2, ge=0, description="Minimum number of connections in the pool")
    POSTGRES_MAX_IDLE: int = Field(default=120, gt=0, description="Idle seconds before closing excess connections")
    POSTGRES_POOL_TIMEOUT: float = Field(
        default=45.0, gt=0, allow_inf_nan=False, description="Timeout in seconds to get a connection from pool"
    )
    POSTGRES_RECONNECT_TIMEOUT: float = Field(
        default=60.0, gt=0, allow_inf_nan=False, description="Timeout for reconnecting to database"
    )
    POSTGRES_CONNECT_TIMEOUT: int = Field(default=10, gt=0)
    POSTGRES_KEEPALIVES_IDLE: int = Field(default=30, gt=0)
    POSTGRES_KEEPALIVES_INTERVAL: int = Field(default=10, gt=0)
    POSTGRES_KEEPALIVES_COUNT: int = Field(default=3, gt=0)
    POSTGRES_TCP_USER_TIMEOUT: int = Field(default=60000, gt=0)
    POSTGRES_HEALTH_CHECK_TIMEOUT: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    POSTGRES_MAX_LIFETIME: float = Field(
        default=300.0,
        gt=0,
        allow_inf_nan=False,
        description="Maximum lifetime of a connection in seconds. After this time, the connection will be closed "
        "and replaced with a new one. Helps prevent stale connections.",
    )
    POSTGRES_NUM_WORKERS: int = Field(
        default=3,
        gt=0,
        description="Number of background workers for pool maintenance (creating/closing connections)",
    )
    POSTGRES_STATEMENT_TIMEOUT: int = Field(
        default=120000,
        ge=0,
        description="Maximum time in milliseconds a query can run before being cancelled. "
        "Prevents stuck queries from blocking connections forever. Set to 0 to disable.",
    )
    POSTGRES_LOCK_TIMEOUT: int = Field(
        default=45000,
        ge=0,
        description="Maximum SQL lock wait in milliseconds. Set to 0 to disable. "
        "THREAD_QUEUE_TIMEOUT controls conversation-lock waiting.",
    )
    POSTGRES_IDLE_IN_TRANSACTION_SESSION_TIMEOUT: int = Field(
        default=120000,
        ge=0,
        description="Maximum time in milliseconds a connection can stay idle in transaction. "
        "Terminates connections that started a transaction but didn't finish. Set to 0 to disable.",
    )

    # Model configuration dictionary.
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    MODEL_CONFIGS_BASE64: str | None = None
    MODEL_CONFIGS_PATH: str | None = None

    # Database configuration dictionary.
    DB_CONFIGS: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    DB_CONFIGS_BASE64: str | None = None
    DB_CONFIGS_PATH: str | None = None

    # Agent configuration.
    # This controls checkpoint ordering. Database I/O still uses async methods.
    CHECKPOINT_DURABILITY: Literal["sync", "async", "exit"] = "sync"
    # Bound service work and database lock resources.
    THREAD_QUEUE_TIMEOUT: float = Field(default=60, gt=0, allow_inf_nan=False)
    THREAD_QUEUE_MAX_WAITERS: int = Field(default=32, ge=0)
    REQUEST_TIMEOUT: float = Field(default=300, gt=0, allow_inf_nan=False)
    REQUEST_CLEANUP_TIMEOUT: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    REQUEST_MAX_CONCURRENT: int = Field(default=8, ge=1)
    REQUEST_QUEUE_MAX_WAITERS: int = Field(default=0, ge=0)
    REQUEST_QUEUE_TIMEOUT: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    RESPONSE_SEND_TIMEOUT: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    REQUEST_MAX_BYTES: int = Field(default=20 * 1024 * 1024, gt=0)
    HISTORY_MAX_PAGE_SIZE: int = Field(default=1000, ge=1, le=1000)
    POSTGRES_LOCK_POOL_SIZE: int = Field(default=10, ge=1)
    POSTGRES_POOL_MAX_WAITING: int = Field(default=100, ge=1)
    THREAD_LOCK_HEARTBEAT: float = Field(default=5, gt=0)
    THREAD_LOCK_HEARTBEAT_TIMEOUT: float = Field(default=5, gt=0)

    DEFAULT_AGENT: str = "create-agent"
    DEFAULT_MAX_MESSAGE_HISTORY_LENGTH: int = 18
    # `TokenTrimMiddleware` uses this token budget for the model message view.
    # `None` disables it. Set a model-specific value to limit history tokens.
    DEFAULT_MAX_TOKENS_HISTORY_LENGTH: int | None = None
    DEFAULT_RECURSION_LIMIT: int = 64
    MULTIMODAL_MAX_ATTACHMENTS: int | None = Field(
        default=None,
        description=(
            "Max number of non-text attachments (image/file/audio/video) allowed per message. "
            "None = no toolkit limit (the model provider still enforces its own per-model caps)."
        ),
    )
    # Detect an interrupted run and resume `Command(resume=...)` on the next request.
    # This requires one checkpointer read per request.
    CHECK_INTERRUPTS: bool = True

    # These are optional defaults for `ClearIntermediateToolCallsMiddleware`.
    # Add it to an agent middleware list. Constructor arguments override these values.
    CLEAR_INTERMEDIATE_TOOL_CALLS: bool = Field(
        default=True,
        description=(
            "Global kill switch for ClearIntermediateToolCallsMiddleware. When False the middleware "
            "is a no-op even if added to an agent."
        ),
    )
    CLEAR_INTERMEDIATE_TOOL_CALLS_BY: Literal["name_args", "name"] = Field(
        default="name_args",
        description=(
            "Dedup key for intermediate tool calls. 'name_args' (safe) collapses only identical "
            "repeat calls; 'name' (aggressive) collapses all repeats of a tool regardless of args."
        ),
    )
    CLEAR_INTERMEDIATE_TOOL_CALLS_KEEP_LAST_N: int = Field(
        default=1,
        ge=1,
        description="How many most-recent results to keep per dedup key in previous turns.",
    )

    # Streamlit configuration.
    DEFAULT_STREAMLIT_USER_ID: str = "streamlit-user"

    # CORS configuration.
    CORS_ENABLED: bool = Field(
        default=False,
        description="Enable CORS middleware. Must be explicitly set to True to enable CORS.",
    )
    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: ["*"],
        description="List of allowed CORS origins. Use ['*'] to allow all origins.",
    )
    CORS_CREDENTIALS: bool = Field(
        default=False,
        description="Allow credentials (cookies, authorization headers) in CORS requests. "
        "Note: Cannot be True when CORS_ORIGINS=['*'] - browsers will reject the response.",
    )
    CORS_METHODS: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        description="List of allowed HTTP methods for CORS requests.",
    )
    CORS_HEADERS: list[str] = Field(
        default_factory=lambda: ["*"],
        description="List of allowed headers for CORS requests. Use ['*'] to allow all headers.",
    )
    CORS_MAX_AGE: int = Field(
        default=600,
        description="Maximum age (in seconds) for preflight requests to be cached.",
    )

    def _apply_langgraph_env_overrides(self) -> None:
        """Validate `LANGGRAPH_` overrides before applying them."""
        source = EnvSettingsSource(type(self), env_prefix="LANGGRAPH_", case_sensitive=True, env_parse_none_str="null")
        self.apply_overrides(source())

    def apply_overrides(self, overrides: Mapping[str, Any]) -> None:
        """Validate all overrides before changing any setting.

        JSON strings are accepted for collection fields. Unknown fields fail.
        """
        fields = type(self).model_fields
        unknown = set(overrides) - set(fields)
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
        validated = {}
        for name, value in overrides.items():
            adapter = TypeAdapter(fields[name].rebuild_annotation())
            try:
                validated[name] = adapter.validate_python(value)
            except ValidationError:
                if not isinstance(value, str):
                    raise
                validated[name] = adapter.validate_json(value)
        for name, value in validated.items():
            setattr(self, name, value)
            logger.debug(f"Applied environment override for {name}")

    def _initialize_configs(self, config_type: str) -> Dict[str, Dict[str, Any]]:
        """Load configurations from the validated settings fields.

        Direct configurations take precedence over base64 and file inputs.
        This keeps values from constructor arguments and `.env` files.
        """
        field_name = f"{config_type}_CONFIGS"
        configs = getattr(self, field_name)
        if configs or field_name in self.model_fields_set:
            return configs

        encoded = getattr(self, f"{field_name}_BASE64")
        path = getattr(self, f"{field_name}_PATH")
        try:
            if encoded:
                raw = base64.b64decode(encoded, validate=True).decode("utf-8")
                parsed = json.loads(raw)
            elif path:
                with open(path, "r", encoding="utf-8") as config_file:
                    parsed = json.load(config_file)
            else:
                return configs
            return TypeAdapter(Dict[str, Dict[str, Any]]).validate_python(parsed)
        except (OSError, ValueError) as error:
            raise ValueError(f"Invalid {field_name} configuration source") from error

    def _initialize_model_configs(self) -> None:
        """Initialize model configurations from environment variables."""
        configs = self._initialize_configs("MODEL")
        if configs != self.MODEL_CONFIGS:
            self.MODEL_CONFIGS = configs

    def _initialize_db_configs(self) -> None:
        """Initialize database configurations from environment variables."""
        configs = self._initialize_configs("DB")
        if configs != self.DB_CONFIGS:
            self.DB_CONFIGS = configs

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

        # Set `LANGFUSE_TRACING_ENVIRONMENT` from `ENV_MODE` when it is unset.
        if self.LANGFUSE_TRACING_ENVIRONMENT is None:
            self.LANGFUSE_TRACING_ENVIRONMENT = self.ENV_MODE.value
            os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = self.LANGFUSE_TRACING_ENVIRONMENT

    @computed_field
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"

    def is_dev(self) -> bool:
        return self.ENV_MODE == "development"
