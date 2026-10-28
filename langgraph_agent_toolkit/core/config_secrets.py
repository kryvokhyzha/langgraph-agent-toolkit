"""Protect known configuration credentials until an SDK needs their values."""

from typing import Any
from urllib.parse import urlsplit

from pydantic import SecretStr


_CREDENTIAL_FIELDS = frozenset(
    {
        "api_key",
        "openai_api_key",
        "anthropic_api_key",
        "google_api_key",
        "azure_ad_token",
        "aws_secret_access_key",
        "aws_session_token",
        "password",
        "passwd",
        "secret",
        "secret_key",
        "client_secret",
        "private_key",
        "token",
        "access_token",
        "refresh_token",
        "auth_token",
        "api_token",
        "bearer_token",
        "connection_string",
        "conninfo",
        "dsn",
    }
)


def _protect(value: Any, *, field: str = "", header: bool = False) -> Any:
    if isinstance(value, str):
        if header or field in _CREDENTIAL_FIELDS:
            return SecretStr(value)
        if field in {"url", "uri", "base_url", "openai_api_base", "azure_endpoint", "endpoint", "openai_proxy"}:
            try:
                parsed = urlsplit(value)
                if parsed.username is not None or parsed.password is not None:
                    return SecretStr(value)
            except ValueError:
                pass
        return value
    if isinstance(value, dict):
        return {
            key: _protect(
                item,
                field=key.lower().replace("-", "_") if isinstance(key, str) else "",
                header=field in {"headers", "default_headers"},
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        if field == "model_parameter_values":
            return [
                type(item)((item[0], _protect(item[1], field=item[0].lower())))
                if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str)
                else _protect(item)
                for item in value
            ]
        return [_protect(item) for item in value]
    if isinstance(value, tuple):
        if field == "model_parameter_values":
            return tuple(_protect(list(value), field=field))
        return tuple(_protect(item) for item in value)
    return value


def protect_config_secrets(configs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Copy named configurations and protect known credential fields."""
    return {name: _protect(config) for name, config in configs.items()}


def reveal_config_secrets(value: Any) -> Any:
    """Copy SDK arguments and unwrap typed credentials at the SDK boundary."""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, dict):
        return {key: reveal_config_secrets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [reveal_config_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(reveal_config_secrets(item) for item in value)
    return value
