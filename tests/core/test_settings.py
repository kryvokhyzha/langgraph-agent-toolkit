import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from langgraph_agent_toolkit.core.settings import Settings, check_str_is_http


def test_check_str_is_http():
    # Test valid HTTP URLs
    assert check_str_is_http("http://example.com/") == "http://example.com/"
    assert check_str_is_http("https://api.test.com/") == "https://api.test.com/"

    # Test invalid URLs
    with pytest.raises(ValidationError):
        check_str_is_http("not_a_url")
    with pytest.raises(ValidationError):
        check_str_is_http("ftp://invalid.com")


def test_settings_default_values():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE_URL": "http://api.example.com",
            "OPENAI_API_KEY": "test_key",
            "OPENAI_MODEL_NAME": "gpt-4",
        },
    ):
        settings = Settings(_env_file=None)
        assert settings.HOST == "0.0.0.0"
        assert settings.PORT == 8080
        assert settings.USE_FAKE_MODEL is False


def test_settings_no_api_keys():
    # Test that settings can be created without API keys when USE_FAKE_MODEL is True
    with patch.dict(os.environ, {"USE_FAKE_MODEL": "true"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.USE_FAKE_MODEL is True


def test_settings_with_compatible_key():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE_URL": "http://api.example.com",
            "OPENAI_API_KEY": "test_key",
            "OPENAI_MODEL_NAME": "gpt-4",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.OPENAI_API_KEY == SecretStr("test_key")
        assert settings.OPENAI_API_BASE_URL == "http://api.example.com"
        assert settings.OPENAI_MODEL_NAME == "gpt-4"


def test_settings_with_fake_model():
    with patch.dict(
        os.environ,
        {
            "USE_FAKE_MODEL": "true",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.USE_FAKE_MODEL is True


def test_settings_with_multiple_providers():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE_URL": "http://api.example.com",
            "OPENAI_API_KEY": "test_key",
            "OPENAI_MODEL_NAME": "gpt-4",
            "USE_FAKE_MODEL": "true",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.OPENAI_API_KEY == SecretStr("test_key")
        assert settings.USE_FAKE_MODEL is True


def test_settings_base_url():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE_URL": "http://api.example.com",
            "OPENAI_API_KEY": "test_key",
            "OPENAI_MODEL_NAME": "gpt-4",
        },
    ):
        settings = Settings(HOST="0.0.0.0", PORT=8000, _env_file=None)
        assert settings.BASE_URL == "http://0.0.0.0:8000"


def test_settings_is_dev():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE_URL": "http://api.example.com",
            "OPENAI_API_KEY": "test_key",
            "OPENAI_MODEL_NAME": "gpt-4",
        },
    ):
        settings = Settings(ENV_MODE="development", _env_file=None)
        assert settings.is_dev() is True

        settings = Settings(ENV_MODE="production", _env_file=None)
        assert settings.is_dev() is False


def test_settings_with_langgraph_string_override():
    # Initialize the settings object first to ensure _apply_langgraph_env_overrides is called
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE_URL": "http://api.example.com",
            "OPENAI_API_KEY": "test_key",
            "OPENAI_MODEL_NAME": "gpt-4",
            "LANGGRAPH_HOST": "127.0.0.1",
        },
        clear=True,
    ):
        # Manually apply the overrides since we're not calling __init__
        settings = Settings(_env_file=None)
        settings._apply_langgraph_env_overrides()
        assert settings.HOST == "127.0.0.1"


def test_settings_with_langgraph_int_override():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE_URL": "http://api.example.com",
            "OPENAI_API_KEY": "test_key",
            "OPENAI_MODEL_NAME": "gpt-4",
            "LANGGRAPH_PORT": "9000",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        settings._apply_langgraph_env_overrides()
        assert settings.PORT == 9000


def test_settings_with_langgraph_boolean_override():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE_URL": "http://api.example.com",
            "OPENAI_API_KEY": "test_key",
            "OPENAI_MODEL_NAME": "gpt-4",
            "LANGGRAPH_USE_FAKE_MODEL": "true",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        settings._apply_langgraph_env_overrides()
        assert settings.USE_FAKE_MODEL is True

    # Test alternative boolean values
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE_URL": "http://api.example.com",
            "OPENAI_API_KEY": "test_key",
            "OPENAI_MODEL_NAME": "gpt-4",
            "LANGGRAPH_USE_FAKE_MODEL": "1",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        settings._apply_langgraph_env_overrides()
        assert settings.USE_FAKE_MODEL is True

    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE_URL": "http://api.example.com",
            "OPENAI_API_KEY": "test_key",
            "OPENAI_MODEL_NAME": "gpt-4",
            "LANGGRAPH_USE_FAKE_MODEL": "false",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        settings._apply_langgraph_env_overrides()
        assert settings.USE_FAKE_MODEL is False


def test_settings_with_langgraph_list_override():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE_URL": "http://api.example.com",
            "OPENAI_API_KEY": "test_key",
            "OPENAI_MODEL_NAME": "gpt-4",
            "LANGGRAPH_AGENT_PATHS": '["custom.path.agent:agent", "another.agent:agent"]',
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        settings._apply_langgraph_env_overrides()
        assert settings.AGENT_PATHS == ["custom.path.agent:agent", "another.agent:agent"]


def test_settings_with_langgraph_multiple_overrides():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE_URL": "http://api.example.com",
            "OPENAI_API_KEY": "test_key",
            "OPENAI_MODEL_NAME": "gpt-4",
            "LANGGRAPH_HOST": "127.0.0.1",
            "LANGGRAPH_PORT": "9000",
            "LANGGRAPH_USE_FAKE_MODEL": "true",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        settings._apply_langgraph_env_overrides()
        assert settings.HOST == "127.0.0.1"
        assert settings.PORT == 9000
        assert settings.USE_FAKE_MODEL is True


def test_model_configs_initialization():
    """Test that model configurations are correctly initialized from environment."""
    test_config = '{"gpt4o": {"provider": "azure_openai", "name": "gpt-4o"}}'

    with patch.dict(
        os.environ,
        {
            "MODEL_CONFIGS": test_config,
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        settings._initialize_model_configs()

        assert "gpt4o" in settings.MODEL_CONFIGS
        assert settings.MODEL_CONFIGS["gpt4o"]["provider"] == "azure_openai"
        assert settings.MODEL_CONFIGS["gpt4o"]["name"] == "gpt-4o"


def test_get_model_config():
    """Test getting a model configuration by key."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=None)
        settings.MODEL_CONFIGS = {
            "gpt4o": {"provider": "azure_openai", "name": "gpt-4o"},
            "gemini": {"provider": "google_genai", "name": "gemini-pro"},
        }

        # Test getting existing config
        assert settings.get_model_config("gpt4o") == {"provider": "azure_openai", "name": "gpt-4o"}

        # Test getting non-existent config
        assert settings.get_model_config("non-existent") is None
