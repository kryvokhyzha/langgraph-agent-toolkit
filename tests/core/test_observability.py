import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.prompts import ChatPromptTemplate

from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.core.observability.langfuse import LangfuseObservability
from langgraph_agent_toolkit.core.observability.langsmith import LangsmithObservability
from langgraph_agent_toolkit.core.observability.types import ChatMessageDict


class TestBaseObservability:
    """Tests for the BaseObservabilityPlatform class."""

    def test_init_with_remote_first(self):
        """Test initialization with remote_first flag."""
        obs = EmptyObservability(remote_first=True)
        assert obs.remote_first is True

        obs_default = EmptyObservability()
        assert obs_default.remote_first is False

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

        result = obs.pull_prompt("greeting")
        assert isinstance(result, ChatPromptTemplate)

    def test_push_pull_chat_messages(self):
        """Test pushing and pulling chat message prompts."""
        obs = EmptyObservability()

        messages: list[ChatMessageDict] = [
            {"role": "system", "content": "You are a helpful assistant for {{ domain }}."},
            {"role": "human", "content": "Help me with {{ topic }}."},
        ]

        obs.push_prompt("chat-prompt", messages)
        result = obs.pull_prompt("chat-prompt")
        assert isinstance(result, ChatPromptTemplate)

    def test_delete_prompt(self):
        """Test deleting a prompt."""
        obs = EmptyObservability()

        template = "Test template"
        obs.push_prompt("to-delete", template)

        # Verify prompt exists
        result = obs.pull_prompt("to-delete")
        assert result is not None

        obs.delete_prompt("to-delete")

        # Verify prompt is deleted
        with pytest.raises(ValueError, match="Prompt 'to-delete' not found"):
            obs.pull_prompt("to-delete")


class TestEmptyObservability:
    """Tests for the EmptyObservability class."""

    def test_record_feedback_logs_debug(self):
        """Test that record_feedback logs debug message instead of raising."""
        obs = EmptyObservability()
        # Should not raise, just log
        obs.record_feedback("run_id", "key", 1.0)

    def test_push_prompt_skip_existing(self):
        """Test that push_prompt skips existing prompt when force_create_new_version=False."""
        obs = EmptyObservability()

        obs.push_prompt("test-prompt", "original content")
        obs.push_prompt("test-prompt", "new content", force_create_new_version=False)

        # Should still have original content
        result = obs.pull_prompt("test-prompt")
        assert result is not None

    def test_push_prompt_overwrite_existing(self):
        """Test that push_prompt overwrites existing prompt when force_create_new_version=True."""
        obs = EmptyObservability()

        obs.push_prompt("test-prompt", "original content")
        obs.push_prompt("test-prompt", "new content", force_create_new_version=True)

        result = obs.pull_prompt("test-prompt")
        assert result is not None

    def test_pull_prompt_not_found(self):
        """Test that pull_prompt raises when prompt not found."""
        obs = EmptyObservability()

        with pytest.raises(ValueError, match="Prompt 'nonexistent' not found"):
            obs.pull_prompt("nonexistent")

    def test_in_memory_storage(self):
        """Test that prompts are stored in memory."""
        obs = EmptyObservability()

        messages: list[ChatMessageDict] = [
            {"role": "system", "content": "System message"},
            {"role": "human", "content": "Human message"},
        ]

        obs.push_prompt("memory-test", messages)

        # Check internal storage
        assert "memory-test" in obs._prompts
        assert obs._prompts["memory-test"] == messages


class TestLangsmithObservability:
    """Tests for the LangsmithObservability class."""

    def test_requires_env_vars(self):
        """Test environment validation is required."""
        obs = LangsmithObservability()

        with patch.dict(os.environ, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variables"):
                obs.get_callback_handler()

    @patch("langgraph_agent_toolkit.core.observability.langsmith.LangsmithClient")
    def test_push_pull_delete_cycle(self, mock_client_cls):
        """Test the full push-pull-delete cycle with LangSmith."""
        from langchain_core.prompts import ChatPromptTemplate

        mock_client = MagicMock()
        mock_client.push_prompt.return_value = "https://api.smith.langchain.com/prompts/123"

        # Mock pull_prompt to return a ChatPromptTemplate
        mock_client.pull_prompt.return_value = ChatPromptTemplate.from_template(
            "Test template for {{ topic }}", template_format="jinja2"
        )

        # Mock the delete_prompt to not be checked for truthiness
        mock_client.delete_prompt = MagicMock(return_value=None)

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
            template = "Test template for {{ topic }}"

            # Push
            obs.push_prompt("test-prompt", template)
            mock_client.push_prompt.assert_called_once()

            # Pull
            result = obs.pull_prompt("test-prompt")
            assert result is not None
            assert isinstance(result, ChatPromptTemplate)

            # Delete - reset mock before delete to get clean call count
            mock_client.delete_prompt.reset_mock()
            obs.delete_prompt("test-prompt")
            mock_client.delete_prompt.assert_called_once_with("test-prompt")


class TestLangfuseObservability:
    """Tests for the LangfuseObservability class."""

    def test_requires_env_vars(self):
        """Test environment validation is required."""
        obs = LangfuseObservability()

        with patch.dict(os.environ, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variables"):
                obs.get_callback_handler()

    @patch("langgraph_agent_toolkit.core.observability.langfuse.get_client")
    def test_record_feedback_with_trace_id_conversion(self, mock_get_client):
        """Test that record_feedback converts UUID to Langfuse trace ID format."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        with patch.dict(
            os.environ,
            {
                "LANGFUSE_SECRET_KEY": "secret",
                "LANGFUSE_PUBLIC_KEY": "public",
                "LANGFUSE_HOST": "https://cloud.langfuse.com",
            },
        ):
            obs = LangfuseObservability()

            # Record feedback with UUID format
            uuid_run_id = "ab9bae0a-c6ec-41d2-8e81-c3a56e357d9d"
            obs.record_feedback(uuid_run_id, "accuracy", 0.95, user_id="user123")

            # Verify create_score was called with converted trace_id
            mock_client.create_score.assert_called_once()
            call_kwargs = mock_client.create_score.call_args[1]

            # UUID should be converted to 32 hex chars (no hyphens)
            expected_trace_id = "ab9bae0ac6ec41d28e81c3a56e357d9d"
            assert call_kwargs["trace_id"] == expected_trace_id
            assert call_kwargs["name"] == "accuracy"
            assert call_kwargs["value"] == 0.95

    @patch("langgraph_agent_toolkit.core.observability.langfuse.get_client")
    def test_trace_context_converts_uuid(self, mock_get_client):
        """Test that trace_context converts UUID to valid Langfuse format."""
        mock_client = MagicMock()
        mock_span = MagicMock()
        mock_client.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_client.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_client.return_value = mock_client

        with patch.dict(
            os.environ,
            {
                "LANGFUSE_SECRET_KEY": "secret",
                "LANGFUSE_PUBLIC_KEY": "public",
                "LANGFUSE_HOST": "https://cloud.langfuse.com",
            },
        ):
            obs = LangfuseObservability()

            uuid_run_id = "ab9bae0a-c6ec-41d2-8e81-c3a56e357d9d"

            with obs.trace_context(
                run_id=uuid_run_id, user_id="user123", input={"message": "test"}, agent_name="test-agent"
            ):
                pass

            # Verify start_as_current_span was called with converted trace_id
            mock_client.start_as_current_span.assert_called_once()
            call_kwargs = mock_client.start_as_current_span.call_args[1]

            expected_trace_id = "ab9bae0ac6ec41d28e81c3a56e357d9d"
            assert call_kwargs["trace_context"]["trace_id"] == expected_trace_id

    @patch("langgraph_agent_toolkit.core.observability.langfuse.get_client")
    def test_compute_prompt_hash_consistency(self, mock_get_client):
        """Test that hash computation is consistent for same content."""
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_SECRET_KEY": "secret",
                "LANGFUSE_PUBLIC_KEY": "public",
                "LANGFUSE_HOST": "https://cloud.langfuse.com",
            },
        ):
            obs = LangfuseObservability()

            # Same string should produce same hash
            hash1 = obs._compute_prompt_hash("Test prompt")
            hash2 = obs._compute_prompt_hash("Test prompt")
            assert hash1 == hash2

            # Different strings should produce different hashes
            hash3 = obs._compute_prompt_hash("Different prompt")
            assert hash1 != hash3

    @patch("langgraph_agent_toolkit.core.observability.langfuse.get_client")
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
            mock_client.get_prompt.side_effect = Exception("Not found")
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
