"""Keep configuration credentials out of representations and validation errors."""

import base64
import json

import pytest
from pydantic import SecretStr, ValidationError

from langgraph_agent_toolkit.core._base_settings import Settings
from langgraph_agent_toolkit.core.config_secrets import reveal_config_secrets
from langgraph_agent_toolkit.core.observability.masking import mask_secret_values
from langgraph_agent_toolkit.service.factory import _setting_environment_value


@pytest.mark.parametrize("kind", ["MODEL", "DB"])
@pytest.mark.parametrize("source", ["constructor", "environment", "base64", "path", "override"])
def test_configuration_credentials_stay_typed_across_sources(kind, source, tmp_path, monkeypatch, mock_env):
    """Protect stored values and preserve the complete SDK configuration."""
    config = {
        "example": {
            "provider": "openai",
            "name": "test-model",
            "api_key": "synthetic-api-secret",
            "password": "synthetic-db-secret",
            "default_headers": {"X-Service-Key": "synthetic-header-secret"},
            "credentials": {"private_key": "synthetic-private-key", "client_email": "service@example.com"},
            "base_url": "https://api.example.com/v1",
            "dsn": "postgresql://user:synthetic-uri-secret@db.example.com/database",
            "url": "https://user:synthetic-url-secret@api.example.com/v1",
            "model_parameter_values": [["api_key", "synthetic-pair-secret"], ["temperature", 0.5]],
            "temperature": 0.3,
        }
    }
    field = f"{kind}_CONFIGS"
    raw = json.dumps(config)
    encoded = base64.b64encode(raw.encode()).decode()
    kwargs = {"_env_file": None}
    if source == "constructor":
        kwargs[field] = config
    elif source == "environment":
        monkeypatch.setenv(field, raw)
    elif source == "base64":
        kwargs[f"{field}_BASE64"] = encoded
    elif source == "path":
        path = tmp_path / "configuration.json"
        path.write_text(raw)
        kwargs[f"{field}_PATH"] = str(path)
    candidate = Settings(**kwargs)
    if source == "override":
        monkeypatch.setenv(f"LANGGRAPH_{field}", raw)
    candidate.setup()
    stored = getattr(candidate, field)
    assert isinstance(stored["example"]["api_key"], SecretStr)
    assert isinstance(stored["example"]["default_headers"]["X-Service-Key"], SecretStr)
    assert isinstance(stored["example"]["url"], SecretStr)
    assert stored["example"]["base_url"] == "https://api.example.com/v1"
    assert stored["example"]["credentials"]["client_email"] == "service@example.com"
    assert stored["example"]["temperature"] == 0.3
    assert json.dumps(config) == raw
    assert reveal_config_secrets(stored) == config
    assert json.loads(_setting_environment_value(stored)) == config
    assert isinstance(candidate.model_dump()[field]["example"]["api_key"], SecretStr)
    for rendered in (repr(candidate), candidate.model_dump_json(), repr(mask_secret_values(data=candidate))):
        assert "synthetic-" not in rendered
        assert encoded not in rendered
    if source == "base64":
        value = getattr(candidate, f"{field}_BASE64")
        assert isinstance(value, SecretStr)
        assert _setting_environment_value(value) == encoded
    serialized = json.loads(candidate.model_dump_json())[field]["example"]
    assert serialized["name"] == "test-model"
    assert serialized["api_key"] == "**********"


@pytest.mark.parametrize("source", ["constructor", "override"])
@pytest.mark.parametrize(
    "invalid",
    [
        {"FEEDBACK_SIGNING_SECRET": "synthetic-short-secret"},
        {"MODEL_CONFIGS": {"example": "synthetic-invalid-secret"}},
        {"AUTH_USERS": {"user": ["synthetic-invalid-secret"]}},
    ],
)
def test_settings_validation_does_not_print_secret_inputs(source, invalid, mock_env):
    """Retain useful field errors without showing input values."""
    with pytest.raises(ValidationError) as captured:
        if source == "constructor":
            Settings(_env_file=None, **invalid)
        else:
            Settings(_env_file=None).apply_overrides(invalid)
    assert "synthetic-" not in str(captured.value)
    assert "input_value" not in str(captured.value)


def test_unknown_configuration_secrets_can_be_typed_explicitly(mock_env):
    """Keep custom credential fields typed without changing other SDK values."""
    credential = SecretStr("synthetic-custom-secret")
    callback = lambda: "synthetic-dynamic-secret"  # noqa: E731
    candidate = Settings(
        _env_file=None,
        MODEL_CONFIGS={"custom": {"custom_credential": credential, "azure_ad_token_provider": callback}},
    )
    configured = candidate.get_model_config("custom")
    assert configured["custom_credential"] is credential
    assert configured["azure_ad_token_provider"] is callback
    revealed = reveal_config_secrets(configured)
    assert revealed["custom_credential"] == "synthetic-custom-secret"
    assert revealed["azure_ad_token_provider"] is callback
    assert configured["custom_credential"] is credential
