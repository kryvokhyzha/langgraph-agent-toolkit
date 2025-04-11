from unittest.mock import patch

import pytest
from langchain_community.chat_models import FakeListChatModel
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSerializable

from langgraph_agent_toolkit.core.models.factory import ModelFactory
from langgraph_agent_toolkit.schema.models import (
    FakeModelName,
    OpenAICompatibleName,
)


def test_get_model_openai_compatible():
    # Clear the cache to ensure a fresh test
    ModelFactory.create.cache_clear()

    with patch("langgraph_agent_toolkit.core.models.factory.settings") as mock_settings:
        # Mock the settings attributes directly
        mock_settings.COMPATIBLE_MODEL = "gpt-4"
        mock_settings.COMPATIBLE_BASE_URL = "http://api.example.com"
        mock_settings.COMPATIBLE_API_KEY = "test_key"

        model = ModelFactory.create(OpenAICompatibleName.OPENAI_COMPATIBLE)
        assert isinstance(model, (ChatOpenAI, RunnableSerializable))
        assert model.model_name == "gpt-4"
        assert model.streaming is True
        assert model.openai_api_base == "http://api.example.com"
        assert model.openai_api_key.get_secret_value() == "test_key"


def test_get_model_openai_compatible_missing_config():
    # Clear the cache to ensure a fresh test
    ModelFactory.create.cache_clear()

    with patch("langgraph_agent_toolkit.core.models.factory.settings") as mock_settings:
        # Set the required attributes to None to simulate missing configuration
        mock_settings.COMPATIBLE_BASE_URL = None
        mock_settings.COMPATIBLE_MODEL = None

        with pytest.raises(ValueError, match="OpenAICompatible base url and endpoint must be configured"):
            ModelFactory.create(OpenAICompatibleName.OPENAI_COMPATIBLE)


def test_get_model_fake():
    model = ModelFactory.create(FakeModelName.FAKE)
    assert isinstance(model, FakeListChatModel)
    assert model.responses == ["This is a test response from the fake model."]


def test_get_model_invalid():
    with pytest.raises(ValueError, match="Unsupported model:"):
        # Using type: ignore since we're intentionally testing invalid input
        ModelFactory.create("invalid_model")  # type: ignore
