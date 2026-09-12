import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.prompts import ChatPromptTemplate

from langgraph_agent_toolkit.core.observability import langfuse as langfuse_adapter
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.core.observability.factory import ObservabilityFactory
from langgraph_agent_toolkit.core.observability.langfuse import LangfuseObservability
from langgraph_agent_toolkit.core.observability.langsmith import LangsmithObservability
from langgraph_agent_toolkit.core.observability.types import ChatMessageDict, ObservabilityBackend


class TestBaseObservability:
    """Tests for the BaseObservabilityPlatform class."""

    def test_validate_environment_missing(self):
        """Test environment validation with missing variables."""
        obs = EmptyObservability()
        obs.required_vars = ["MISSING_VAR1", "MISSING_VAR2"]

        with pytest.raises(ValueError, match="Missing required environment variables"):
            obs.validate_environment()

    def test_validate_environment_present(self):
        """Test environment validation with present variables."""
        obs = EmptyObservability()

        with patch.dict(os.environ, {"TEST_VAR": "value"}, clear=False):
            obs.required_vars = ["TEST_VAR"]
            assert obs.validate_environment() is True

    def test_push_pull_string_prompt(self):
        """Test pushing and pulling a string prompt."""
        obs = EmptyObservability()

        template = "Hello, {{ name }}! Welcome to {{ place }}."
        obs.push_prompt("greeting", template)

        result = obs.pull_prompt("greeting", template_format="jinja2")
        assert result.invoke({"name": "Sam", "place": "London"}).to_messages()[0].content == (
            "Hello, Sam! Welcome to London."
        )

    def test_push_pull_chat_messages(self):
        """Test pushing and pulling chat message prompts."""
        obs = EmptyObservability()

        messages: list[ChatMessageDict] = [
            {"role": "system", "content": "You are a helpful assistant for {{ domain }}."},
            {"role": "human", "content": "Help me with {{ topic }}."},
        ]

        obs.push_prompt("chat-prompt", messages)
        result = obs.pull_prompt("chat-prompt", template_format="jinja2")
        rendered = result.invoke({"domain": "math", "topic": "sums"}).to_messages()
        assert [(message.type, message.content) for message in rendered] == [
            ("system", "You are a helpful assistant for math."),
            ("human", "Help me with sums."),
        ]

    def test_delete_prompt(self):
        """Test deleting a prompt."""
        obs = EmptyObservability()

        template = "Test template"
        obs.push_prompt("to-delete", template)

        # Verify prompt exists
        result = obs.pull_prompt("to-delete")
        assert isinstance(result, ChatPromptTemplate)

        obs.delete_prompt("to-delete")

        # Verify prompt is deleted
        with pytest.raises(ValueError, match="Prompt 'to-delete' not found"):
            obs.pull_prompt("to-delete")


class TestEmptyObservability:
    """Tests for the EmptyObservability class."""

    def test_feedback_does_not_require_credentials(self, mock_env):
        """Accept feedback when the observability backend is disabled."""
        obs = EmptyObservability()
        assert obs.record_feedback("run-123", "stars", 1.0) is None

    def test_push_prompt_skips_when_not_forcing(self):
        """force_create_new_version=False keeps the ORIGINAL stored prompt (skip branch)."""
        obs = EmptyObservability()

        obs.push_prompt("test-prompt", "original content")
        obs.push_prompt("test-prompt", "new content", force_create_new_version=False)

        assert obs.pull_prompt("test-prompt").invoke({}).to_messages()[0].content == "original content"

    def test_push_prompt_overwrites_when_forcing(self):
        """force_create_new_version=True replaces the stored prompt (overwrite branch)."""
        obs = EmptyObservability()

        obs.push_prompt("test-prompt", "original content")
        obs.push_prompt("test-prompt", "new content", force_create_new_version=True)

        assert obs.pull_prompt("test-prompt").invoke({}).to_messages()[0].content == "new content"

    def test_pull_prompt_not_found(self):
        """Test that pull_prompt raises when prompt not found."""
        obs = EmptyObservability()

        with pytest.raises(ValueError, match="Prompt 'nonexistent' not found"):
            obs.pull_prompt("nonexistent")


class TestLangsmithObservability:
    """Tests for the LangsmithObservability class."""

    def test_requires_env_vars(self):
        """Test environment validation is required."""
        obs = LangsmithObservability()

        with patch.dict(os.environ, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variables"):
                obs.get_callback_handler()

    @patch("langgraph_agent_toolkit.core.observability.langsmith.LangsmithClient")
    def test_push_converts_to_chat_prompt_pull_processes_delete_proxies(self, mock_client_cls):
        """Push converts the raw template to a ChatPromptTemplate; pull processes it back; delete proxies the name."""
        from langchain_core.prompts import ChatPromptTemplate

        mock_client = MagicMock()
        mock_client.pull_prompt.return_value = ChatPromptTemplate.from_template(
            "Test template for {{ topic }}", template_format="jinja2"
        )
        mock_client_cls.return_value = mock_client

        with patch.dict(
            os.environ,
            {
                "LANGSMITH_TRACING": "true",
                "LANGSMITH_API_KEY": "test-key",
                "LANGSMITH_PROJECT": "test-project",
                "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
            },
        ):
            obs = LangsmithObservability()

            # Push: the raw string template must be converted to a ChatPromptTemplate before sending.
            obs.push_prompt("test-prompt", "Test template for {{ topic }}")
            pushed_object = mock_client.push_prompt.call_args.kwargs["object"]
            assert isinstance(pushed_object, ChatPromptTemplate)

            # Pull: the client's prompt object is processed back into a ChatPromptTemplate.
            result = obs.pull_prompt("test-prompt")
            assert isinstance(result, ChatPromptTemplate)

            # Delete proxies the name straight through to the client.
            obs.delete_prompt("test-prompt")
            mock_client.delete_prompt.assert_called_once_with("test-prompt")


class TestLangfuseObservability:
    """Tests for the LangfuseObservability class."""

    def test_requires_env_vars(self, monkeypatch):
        """Reject missing credentials in both supported configuration sources."""
        monkeypatch.setattr(
            langfuse_adapter,
            "settings",
            langfuse_adapter.settings.model_copy(update={"LANGFUSE_PUBLIC_KEY": None, "LANGFUSE_SECRET_KEY": None}),
        )
        client_factory = MagicMock()
        monkeypatch.setattr(langfuse_adapter, "_get_langfuse_client", client_factory)
        obs = LangfuseObservability()

        with patch.dict(os.environ, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variables"):
                obs.get_callback_handler()
        client_factory.assert_not_called()

    @patch("langgraph_agent_toolkit.core.observability.langfuse._get_langfuse_client")
    def test_push_prompt_version_control(self, mock_get_client):
        """Test prompt version control based on content hash."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "human", "content": "Help with {{ topic }}"},
        ]

        with patch.dict(
            os.environ,
            {
                "LANGFUSE_SECRET_KEY": "secret",
                "LANGFUSE_PUBLIC_KEY": "public",
                "LANGFUSE_HOST": "https://cloud.langfuse.com",
            },
        ):
            obs = LangfuseObservability()
            expected_hash = obs._compute_prompt_hash(messages)

            # Test 1: No existing prompt - should create new
            not_found = Exception("Not found")
            not_found.status_code = 404
            mock_client.get_prompt.side_effect = not_found
            mock_new_prompt = MagicMock()
            mock_client.create_prompt.return_value = mock_new_prompt

            obs.push_prompt("test-prompt", messages)
            mock_client.create_prompt.assert_called_once()
            create_kwargs = mock_client.create_prompt.call_args[1]
            assert create_kwargs["commit_message"] == expected_hash

            # Test 2: Existing prompt with same hash - should not create new (force=False)
            mock_client.reset_mock()
            mock_existing = MagicMock()
            mock_existing.commit_message = expected_hash
            mock_client.get_prompt.side_effect = None
            mock_client.get_prompt.return_value = mock_existing

            obs.push_prompt("test-prompt", messages, force_create_new_version=False)
            mock_client.create_prompt.assert_not_called()

            # Test 3: Force new version even with same content
            mock_client.reset_mock()
            obs.push_prompt("test-prompt", messages, force_create_new_version=True)
            mock_client.create_prompt.assert_called_once()


class TestObservabilityFactory:
    """Tests for ObservabilityFactory.create dispatch."""

    def test_create_empty_passes_remote_first(self):
        obs = ObservabilityFactory.create(ObservabilityBackend.EMPTY, remote_first=True)
        assert isinstance(obs, EmptyObservability)
        assert obs.remote_first is True

    @pytest.mark.parametrize(
        "name, cls",
        [("langsmith", LangsmithObservability), ("langfuse", LangfuseObservability)],
    )
    def test_create_dispatches_to_backend(self, name, cls):
        assert isinstance(ObservabilityFactory.create(name), cls)

    def test_create_unsupported_raises(self):
        with pytest.raises(ValueError, match="not a valid ObservabilityBackend"):
            ObservabilityFactory.create("bogus")
