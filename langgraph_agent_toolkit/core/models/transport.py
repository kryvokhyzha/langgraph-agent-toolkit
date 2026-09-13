"""Own HTTP connections for OpenAI and Azure model calls."""

from __future__ import annotations

import asyncio
import importlib
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from langgraph_agent_toolkit.helper.logging import logger


_manager: ContextVar[LLMTransportManager | None] = ContextVar("llm_transport_manager", default=None)
_CLIENT_FIELDS = ("http_client", "http_async_client", "client", "async_client", "root_client", "root_async_client")
_CONNECTION_FIELDS = (
    "api_key",
    "openai_api_key",
    "base_url",
    "openai_api_base",
    "organization",
    "openai_organization",
    "default_headers",
    "default_query",
    "azure_endpoint",
    "azure_deployment",
    "deployment_name",
    "api_version",
    "openai_api_version",
    "azure_ad_token",
    "azure_ad_token_provider",
    "azure_ad_async_token_provider",
    "openai_proxy",
)
_CONNECTION_ENV = (
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT_ID",
    "OPENAI_CUSTOM_HEADERS",
    "OPENAI_ADMIN_KEY",
    "OPENAI_PROXY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "OPENAI_API_VERSION",
    "AZURE_OPENAI_AD_TOKEN",
    "LANGSMITH_GATEWAY",
    "LANGSMITH_GATEWAY_API_KEY",
    "LANGSMITH_API_KEY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)


class LLMTransportConfig(BaseModel):
    """Set connection limits for each endpoint and credential configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    async_transport: Literal["httpx", "aiohttp"] = "httpx"
    # Match OpenAI SDK 3.13.0 defaults without importing this optional dependency.
    max_connections: int = Field(default=1000, gt=0)
    max_keepalive_connections: int = Field(default=100, ge=0)
    keepalive_expiry: float = Field(default=5, gt=0, allow_inf_nan=False)
    connect_timeout: float = Field(default=5, gt=0, allow_inf_nan=False)
    read_timeout: float = Field(default=600, gt=0, allow_inf_nan=False)
    write_timeout: float = Field(default=600, gt=0, allow_inf_nan=False)
    pool_timeout: float = Field(default=600, gt=0, allow_inf_nan=False)
    max_retries: int = Field(default=2, ge=0)
    shutdown_timeout: float = Field(default=10, gt=0, allow_inf_nan=False)
    max_pools: int = Field(default=32, gt=0)

    @model_validator(mode="after")
    def validate_limits(self) -> LLMTransportConfig:
        if self.max_keepalive_connections > self.max_connections:
            raise ValueError("max_keepalive_connections must not exceed max_connections")
        return self


def _http_library() -> Any:
    """Use the public HTTP library that matches the installed OpenAI SDK."""
    import openai

    name = "httpx2" if int(openai.__version__.split(".")[0]) >= 3 else "httpx"
    return importlib.import_module(name)


def _identity(value: Any) -> Any:
    """Keep equal connection settings together without evaluating token providers."""
    if isinstance(value, Mapping):
        return tuple(sorted((key, _identity(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_identity(item) for item in value)
    try:
        hash(value)
    except TypeError:
        raise TypeError("Model connection settings must contain hashable values or mappings") from None
    return value


def _default_stream_usage(provider: str, params: Mapping[str, Any]) -> bool:
    """Keep LangChain's automatic usage reporting when we inject HTTP clients."""
    if params.get("openai_proxy", os.getenv("OPENAI_PROXY")) is not None:
        return False
    base_url = params.get("base_url", params.get("openai_api_base"))
    if provider == "openai" and base_url is None:
        base_url = os.getenv("OPENAI_API_BASE") or None
    if base_url is not None:
        return False
    if provider == "azure_openai":
        return True
    gateway = os.getenv("LANGSMITH_GATEWAY", "").lower()
    return gateway not in ("", "false", "0", "no") or "OPENAI_BASE_URL" not in os.environ


class LLMTransportManager:
    """Reuse HTTP clients inside one worker and one event loop.

    Enter this manager before model construction. Bind it during graph startup
    and each request. Close it after all model calls finish. Explicit clients
    and explicit socket options keep the caller's existing behavior.
    """

    def __init__(self, config: LLMTransportConfig | None = None) -> None:
        self.config = config or LLMTransportConfig()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._lock = threading.RLock()
        self._pools: dict[Any, tuple[Any, Any]] = {}

    @classmethod
    def from_settings(cls, settings: Any) -> LLMTransportManager:
        values = {name: getattr(settings, f"LLM_HTTP_{name.upper()}") for name in LLMTransportConfig.model_fields}
        return cls(LLMTransportConfig(**values))

    async def __aenter__(self) -> LLMTransportManager:
        if self._closed or self._loop is not None:
            raise RuntimeError("LLM transport manager cannot be entered again")
        self._loop = asyncio.get_running_loop()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    def _check_owner(self) -> None:
        if self._closed or self._loop is None:
            raise RuntimeError("LLM transport manager is not open")
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # LangChain copies context into threads for synchronous model calls.
            return
        if current_loop is not self._loop:
            raise RuntimeError("LLM transport manager belongs to another event loop")

    @contextmanager
    def bind(self) -> Iterator[LLMTransportManager]:
        self._check_owner()
        token = _manager.set(self)
        try:
            yield self
        finally:
            _manager.reset(token)

    def configure(self, provider: str | None, params: dict[str, Any], *, chat: bool = True) -> dict[str, Any]:
        """Add owned clients without changing credentials or explicit clients."""
        if provider not in ("openai", "azure_openai"):
            return params
        if any(params.get(field) is not None for field in _CLIENT_FIELDS):
            return params
        if params.get("http_socket_options") is not None:
            return params
        self._check_owner()

        # The SDK transport types are optional dependencies. Import them on use.
        httpx = _http_library()

        configured = dict(params)
        if "timeout" not in configured and "request_timeout" not in configured:
            configured["timeout"] = httpx.Timeout(
                connect=self.config.connect_timeout,
                read=self.config.read_timeout,
                write=self.config.write_timeout,
                pool=self.config.pool_timeout,
            )
        configured.setdefault("max_retries", self.config.max_retries)
        if chat and configured.get("stream_usage") is None and _default_stream_usage(provider, params):
            configured["stream_usage"] = True

        key = (
            provider,
            tuple((name, _identity(params.get(name))) for name in _CONNECTION_FIELDS),
            tuple((name, os.getenv(name)) for name in _CONNECTION_ENV),
        )
        with self._lock:
            self._check_owner()
            pair = self._pools.get(key)
            if pair is None:
                if len(self._pools) >= self.config.max_pools:
                    raise RuntimeError("LLM HTTP pool limit reached; increase LLM_HTTP_MAX_POOLS")
                pair = self._create_clients(params.get("openai_proxy", os.getenv("OPENAI_PROXY")))
                self._pools[key] = pair
        configured["http_client"], configured["http_async_client"] = pair
        configured["openai_proxy"] = None
        if chat:
            # Our transports already contain the socket options.
            configured["http_socket_options"] = ()
        return configured

    def _create_clients(self, proxy: str | None) -> tuple[Any, Any]:
        from openai import DefaultAioHttpClient, DefaultAsyncHttpxClient, DefaultHttpxClient

        httpx = _http_library()
        limits = httpx.Limits(
            max_connections=self.config.max_connections,
            max_keepalive_connections=self.config.max_keepalive_connections,
            keepalive_expiry=self.config.keepalive_expiry,
        )
        client_kwargs: dict[str, Any] = {"limits": limits}
        socket_options: tuple = ()
        try:
            from langchain_openai.chat_models._client_utils import (
                _resolve_socket_options,
                _should_bypass_socket_options_for_proxy_env,
            )

            if not _should_bypass_socket_options_for_proxy_env(
                http_socket_options=None, http_client=None, http_async_client=None, openai_proxy=proxy
            ):
                socket_options = _resolve_socket_options(None)
        except ImportError:
            # Network phase timeouts still apply if LangChain moves this helper.
            logger.warning("LangChain socket defaults are unavailable; use HTTP transport defaults")

        def kwargs_for(async_client: bool) -> dict[str, Any]:
            result = dict(client_kwargs)
            if socket_options:
                transport_type = httpx.AsyncHTTPTransport if async_client else httpx.HTTPTransport
                options: dict[str, Any] = {"limits": limits, "socket_options": list(socket_options)}
                if proxy:
                    options["proxy"] = httpx.Proxy(proxy)
                    result["mounts"] = {"all://": transport_type(**options)}
                else:
                    result["transport"] = transport_type(**options)
            elif proxy:
                result["proxy"] = proxy
            return result

        sync_client = DefaultHttpxClient(**kwargs_for(False))
        try:
            if self.config.async_transport == "aiohttp":
                async_kwargs = dict(client_kwargs)
                if proxy:
                    async_kwargs["proxy"] = proxy
                async_client = DefaultAioHttpClient(**async_kwargs)
            else:
                async_client = DefaultAsyncHttpxClient(**kwargs_for(True))
        except BaseException:
            sync_client.close()
            raise
        return sync_client, async_client

    async def aclose(self) -> None:
        """Close all owned clients within the configured shutdown deadline."""
        with self._lock:
            if self._closed:
                return
            self._check_owner()
            self._closed = True
            pairs = list(self._pools.values())
            self._pools.clear()
        operations = [
            operation for sync, asynchronous in pairs for operation in (self._close_sync(sync), asynchronous.aclose())
        ]
        if operations:
            results = await asyncio.wait_for(
                asyncio.gather(*operations, return_exceptions=True), timeout=self.config.shutdown_timeout
            )
            if any(isinstance(result, BaseException) for result in results):
                raise RuntimeError("An LLM HTTP client did not close")

    @staticmethod
    async def _close_sync(client: Any) -> None:
        """Keep a stalled close operation out of the default executor."""
        loop = asyncio.get_running_loop()
        completed = loop.create_future()

        def finish(error: BaseException | None) -> None:
            if not completed.done():
                # Store the error as a value if shutdown no longer awaits it.
                completed.set_result(error)

        def close() -> None:
            error = None
            try:
                client.close()
            except BaseException as exc:
                error = exc
            try:
                loop.call_soon_threadsafe(finish, error)
            except RuntimeError:
                pass  # The worker event loop has already stopped.

        threading.Thread(target=close, name="llm-http-close", daemon=True).start()
        error = await completed
        if error is not None:
            raise error


def current_llm_transport_manager() -> LLMTransportManager | None:
    """Return the manager bound to the current request."""
    return _manager.get()


def configure_model_transport(provider: str | None, params: dict[str, Any], *, chat: bool = True) -> dict[str, Any]:
    manager = current_llm_transport_manager()
    return manager.configure(provider, params, chat=chat) if manager is not None else params
