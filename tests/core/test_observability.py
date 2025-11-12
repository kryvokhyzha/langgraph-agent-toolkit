import os
import tempfile
from pathlib import Path
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

        with tempfile.TemporaryDirectory() as temp_dir:
            obs_with_dir = EmptyObservability(prompts_dir=temp_dir, remote_first=True)
            assert obs_with_dir.prompts_dir == Path(temp_dir)
            assert obs_with_dir.remote_first is True

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
        with tempfile.TemporaryDirectory() as temp_dir:
            obs = EmptyObservability(prompts_dir=temp_dir)

            template = "Hello, {{ name }}! Welcome to {{ place }}."
            obs.push_prompt("greeting", template)

            result = obs.pull_prompt("greeting")
            assert isinstance(result, ChatPromptTemplate)

            raw = obs.get_template("greeting")
            assert raw == template

    def test_push_pull_chat_messages(self):
        """Test pushing and pulling chat message prompts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            obs = EmptyObservability(prompts_dir=temp_dir)

            messages: list[ChatMessageDict] = [
                {"role": "system", "content": "You are a helpful assistant for {{ domain }}."},
                {"role": "human", "content": "Help me with {{ topic }}."},
            ]

            obs.push_prompt("chat-prompt", messages)
            result = obs.pull_prompt("chat-prompt")
            assert isinstance(result, ChatPromptTemplate)

    def test_render_prompt(self):
        """Test rendering a prompt with variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            obs = EmptyObservability(prompts_dir=temp_dir)

            template = "Hello, {{ name }}! Welcome to {{ place }}."
            obs.push_prompt("render-test", template)

            rendered = obs.render_prompt("render-test", name="Alice", place="Wonderland")
            assert rendered == "Hello, Alice! Welcome to Wonderland."

    def test_delete_prompt(self):
        """Test deleting a prompt."""
        with tempfile.TemporaryDirectory() as temp_dir:
            obs = EmptyObservability(prompts_dir=temp_dir)

            template = "Test template"
            obs.push_prompt("to-delete", template)

            template_path = Path(temp_dir) / "to-delete.jinja2"
            assert template_path.exists()

            obs.delete_prompt("to-delete")
            assert not template_path.exists()


class TestEmptyObservability:
    """Tests for the EmptyObservability class."""

    def test_record_feedback_raises(self):
        """Test that record_feedback raises an error."""
        obs = EmptyObservability()
        with pytest.raises(ValueError, match="Cannot record feedback"):
            obs.record_feedback("run_id", "key", 1.0)


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
        mock_client = MagicMock()
        mock_client.push_prompt.return_value = "https://api.smith.langchain.com/prompts/123"

        mock_prompt = MagicMock()
        mock_prompt.template = "Test template string"
        mock_client.pull_prompt.return_value = mock_prompt

        # Mock the delete_prompt to not be checked for truthiness
        mock_client.delete_prompt = MagicMock(return_value=None)

        mock_client_cls.return_value = mock_client

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                os.environ,
                {
                    "LANGSMITH_TRACING": "true",
                    "LANGSMITH_API_KEY": "test-key",
                    "LANGSMITH_PROJECT": "test-project",
                    "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
                },
            ):
                obs = LangsmithObservability(prompts_dir=temp_dir)
                template = "Test template for {{ topic }}"

                # Push
                obs.push_prompt("test-prompt", template)
                mock_client.push_prompt.assert_called_once()

                # Pull
                result = obs.pull_prompt("test-prompt")
                assert result is not None

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
    @patch("langgraph_agent_toolkit.core.observability.base.joblib.dump")
    def test_push_prompt_version_control(self, mock_dump, mock_get_client):
        """Test prompt version control based on content hash."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "human", "content": "Help with {{ topic }}"},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                os.environ,
                {
                    "LANGFUSE_SECRET_KEY": "secret",
                    "LANGFUSE_PUBLIC_KEY": "public",
                    "LANGFUSE_HOST": "https://cloud.langfuse.com",
                },
            ):
                obs = LangfuseObservability(prompts_dir=temp_dir)
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
