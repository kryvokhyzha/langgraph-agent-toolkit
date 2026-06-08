from unittest.mock import MagicMock, patch

import pytest
from langchain.chat_models.base import _ConfigurableModel
from langchain_community.chat_models import FakeListChatModel
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI

from langgraph_agent_toolkit.core.models.chat_openai import ChatOpenAIPatched
from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory, EmbeddingModelFactory
from langgraph_agent_toolkit.schema.models import ModelProvider


def test_get_model_openai_compatible():
    # Fix: Pass the base URL directly in kwargs instead of relying on settings
    model = CompletionModelFactory.create(
        ModelProvider.OPENAI, model_name="gpt-4", openai_api_key="test_key", openai_api_base="http://api.example.com"
    )
    assert isinstance(model, (ChatOpenAI, RunnableSerializable, _ConfigurableModel))
    assert model.model_name == "gpt-4"
    assert model.streaming is True
    assert model.openai_api_base == "http://api.example.com"
    assert model.openai_api_key.get_secret_value() == "test_key"


def test_get_model_openai_compatible_missing_config():
    # Fix: No need to patch settings here since we're testing a specific condition
    with pytest.raises(ValueError, match="Model name must be provided for non-fake models"):
        CompletionModelFactory.create(ModelProvider.OPENAI)


def test_get_model_fake():
    model = CompletionModelFactory.create(ModelProvider.FAKE)
    assert isinstance(model, FakeListChatModel)
    assert model.responses == ["This is a test response from the fake model."]


def test_get_model_invalid():
    # Invalid provider string raises ValueError when converting to ModelProvider enum
    with pytest.raises(ValueError, match="is not a valid ModelProvider"):
        CompletionModelFactory.create("invalid_model", model_name=None)  # type: ignore


@pytest.mark.parametrize("input_role", ["assistant", "assistant_custom", "assistantXYZ"])
def test_chat_openai_patched_normalizes_assistant_role(input_role):
    """ChatOpenAIPatched rewrites any choice role starting with 'assistant' back to plain 'assistant'.

    This papers over OpenAI-compatible gateways (e.g. LiteLLM) that return roles like 'assistant_xyz'.
    """
    msg = MagicMock()
    msg.role = input_role
    response = MagicMock(choices=[MagicMock(message=msg)])

    model = ChatOpenAIPatched(model_name="gpt-4o", openai_api_key="test_key")
    with patch.object(ChatOpenAI, "_create_chat_result", return_value="parsed") as mock_super:
        result = model._create_chat_result(response)

    assert msg.role == "assistant"  # normalized in place before delegating
    assert result == "parsed"  # delegated to the real ChatOpenAI parser
    mock_super.assert_called_once()


def test_init_chat_model_helper_openai_uses_patched():
    """The 'openai' provider routes through ChatOpenAIPatched."""
    with patch("langgraph_agent_toolkit.core.models.factory.ChatOpenAIPatched", return_value="patched") as mock_patched:
        result = CompletionModelFactory._init_chat_model_helper("gpt-4o", model_provider="openai", openai_api_key="k")
    assert result == "patched"
    mock_patched.assert_called_once_with(model_name="gpt-4o", openai_api_key="k")


def test_init_chat_model_helper_non_openai_uses_langchain():
    """A non-openai provider defers to LangChain's init helper (not ChatOpenAIPatched)."""
    with patch("langgraph_agent_toolkit.core.models.factory._init_chat_model_helper", return_value="lc") as mock_lc:
        result = CompletionModelFactory._init_chat_model_helper("claude-3", model_provider="anthropic", api_key="k")
    assert result == "lc"
    assert mock_lc.call_args.args[0] == "claude-3"
    assert mock_lc.call_args.kwargs["model_provider"] == "anthropic"


def test_get_model_from_config_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        CompletionModelFactory.get_model_from_config({})


def test_get_model_from_config_missing_name_raises():
    with pytest.raises(ValueError, match="Model name must be specified"):
        CompletionModelFactory.get_model_from_config({"provider": "openai"})


def test_get_model_from_config_forwards_to_create():
    with patch.object(CompletionModelFactory, "create", return_value="model") as mock_create:
        result = CompletionModelFactory.get_model_from_config(
            {"provider": "openai", "name": "gpt-4", "temperature": 0.7}
        )
    assert result == "model"
    kwargs = mock_create.call_args.kwargs
    assert kwargs["model_provider"] == "openai"
    assert kwargs["model_name"] == "gpt-4"
    assert kwargs["temperature"] == 0.7


def test_embedding_factory_requires_model_name():
    with pytest.raises(ValueError, match="Model name must be provided for embedding models"):
        EmbeddingModelFactory.create(ModelProvider.OPENAI)


def test_embedding_factory_delegates_to_init_embeddings():
    with patch("langgraph_agent_toolkit.core.models.factory.init_embeddings", return_value="emb") as mock_init:
        result = EmbeddingModelFactory.create(ModelProvider.OPENAI, "text-embedding-3-small", openai_api_key="k")
    assert result == "emb"
    kwargs = mock_init.call_args.kwargs
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["provider"] == "openai"
    assert kwargs["openai_api_key"] == "k"


def test_embedding_get_model_from_config_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        EmbeddingModelFactory.get_model_from_config({})
