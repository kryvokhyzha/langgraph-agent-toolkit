import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from langgraph_agent_toolkit.core.settings import Settings, check_str_is_http
from langgraph_agent_toolkit.schema.models import (
    FakeModelName,
    OpenAICompatibleName,
)


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
            "MODEL_BASE_URL": "http://api.example.com",
            "MODEL_API_KEY": "test_key",
            "MODEL_NAME": "gpt-4",
        },
    ):
        settings = Settings(_env_file=None)
        assert settings.HOST == "0.0.0.0"
        assert settings.PORT == 8080
        assert settings.USE_FAKE_MODEL is False


def test_settings_no_api_keys():
    # Test that settings raises error when no API keys are provided
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="At least one LLM API key must be provided"):
            _ = Settings(_env_file=None)


def test_settings_with_compatible_key():
    with patch.dict(
        os.environ,
        {
            "MODEL_BASE_URL": "http://api.example.com",
            "MODEL_API_KEY": "test_key",
            "MODEL_NAME": "gpt-4",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.MODEL_API_KEY == SecretStr("test_key")
        assert settings.MODEL_BASE_URL == "http://api.example.com"
        assert settings.MODEL_NAME == "gpt-4"
        assert settings.DEFAULT_MODEL_TYPE == OpenAICompatibleName.OPENAI_COMPATIBLE
        assert settings.AVAILABLE_MODELS == set(OpenAICompatibleName)


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
        assert settings.DEFAULT_MODEL_TYPE == FakeModelName.FAKE
        assert settings.AVAILABLE_MODELS == set(FakeModelName)


def test_settings_with_multiple_providers():
    with patch.dict(
        os.environ,
        {
            "MODEL_BASE_URL": "http://api.example.com",
            "MODEL_API_KEY": "test_key",
            "MODEL_NAME": "gpt-4",
            "USE_FAKE_MODEL": "true",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.MODEL_API_KEY == SecretStr("test_key")
        assert settings.USE_FAKE_MODEL is True
        # When multiple providers are available, OpenAI-compatible should be the default
        # (based on order in the code)
        assert settings.DEFAULT_MODEL_TYPE == OpenAICompatibleName.OPENAI_COMPATIBLE
        # Available models should include both OpenAI-compatible and Fake models
        expected_models = set(OpenAICompatibleName)
        expected_models.update(set(FakeModelName))
        assert settings.AVAILABLE_MODELS == expected_models


def test_settings_base_url():
    with patch.dict(
        os.environ,
        {
            "MODEL_BASE_URL": "http://api.example.com",
            "MODEL_API_KEY": "test_key",
            "MODEL_NAME": "gpt-4",
        },
    ):
        settings = Settings(HOST="0.0.0.0", PORT=8000, _env_file=None)
        assert settings.BASE_URL == "http://0.0.0.0:8000"


def test_settings_is_dev():
    with patch.dict(
        os.environ,
        {
            "MODEL_BASE_URL": "http://api.example.com",
            "MODEL_API_KEY": "test_key",
            "MODEL_NAME": "gpt-4",
        },
    ):
        settings = Settings(ENV_MODE="development", _env_file=None)
        assert settings.is_dev() is True

        settings = Settings(ENV_MODE="production", _env_file=None)
        assert settings.is_dev() is False


def test_settings_with_langgraph_string_override():
    with patch.dict(
        os.environ,
        {
            "MODEL_BASE_URL": "http://api.example.com",
            "MODEL_API_KEY": "test_key",
            "MODEL_NAME": "gpt-4",
            "LANGGRAPH_HOST": "127.0.0.1",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.HOST == "127.0.0.1"


def test_settings_with_langgraph_int_override():
    with patch.dict(
        os.environ,
        {
            "MODEL_BASE_URL": "http://api.example.com",
            "MODEL_API_KEY": "test_key",
            "MODEL_NAME": "gpt-4",
            "LANGGRAPH_PORT": "9000",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.PORT == 9000


def test_settings_with_langgraph_boolean_override():
    with patch.dict(
        os.environ,
        {
            "MODEL_BASE_URL": "http://api.example.com",
            "MODEL_API_KEY": "test_key",
            "MODEL_NAME": "gpt-4",
            "LANGGRAPH_USE_FAKE_MODEL": "true",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.USE_FAKE_MODEL is True

    # Test alternative boolean values
    with patch.dict(
        os.environ,
        {
            "MODEL_BASE_URL": "http://api.example.com",
            "MODEL_API_KEY": "test_key",
            "MODEL_NAME": "gpt-4",
            "LANGGRAPH_USE_FAKE_MODEL": "1",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.USE_FAKE_MODEL is True

    with patch.dict(
        os.environ,
        {
            "MODEL_BASE_URL": "http://api.example.com",
            "MODEL_API_KEY": "test_key",
            "MODEL_NAME": "gpt-4",
            "LANGGRAPH_USE_FAKE_MODEL": "false",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.USE_FAKE_MODEL is False


def test_settings_with_langgraph_list_override():
    with patch.dict(
        os.environ,
        {
            "MODEL_BASE_URL": "http://api.example.com",
            "MODEL_API_KEY": "test_key",
            "MODEL_NAME": "gpt-4",
            "LANGGRAPH_AGENT_PATHS": '["custom.path.agent:agent", "another.agent:agent"]',
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.AGENT_PATHS == ["custom.path.agent:agent", "another.agent:agent"]


def test_settings_with_langgraph_multiple_overrides():
    with patch.dict(
        os.environ,
        {
            "MODEL_BASE_URL": "http://api.example.com",
            "MODEL_API_KEY": "test_key",
            "MODEL_NAME": "gpt-4",
            "LANGGRAPH_HOST": "127.0.0.1",
            "LANGGRAPH_PORT": "9000",
            "LANGGRAPH_USE_FAKE_MODEL": "true",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.HOST == "127.0.0.1"
        assert settings.PORT == 9000
        assert settings.USE_FAKE_MODEL is True
