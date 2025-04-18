import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.prompts import ChatPromptTemplate

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.core.observability.langfuse import LangfuseObservability
from langgraph_agent_toolkit.core.observability.langsmith import LangsmithObservability
from langgraph_agent_toolkit.core.observability.types import ChatMessageDict, ObservabilityBackend, PromptTemplateType


class TestBaseObservability:
    """Tests for the BaseObservabilityPlatform class."""

    def test_init_defaults(self):
        """Test initialization with default settings."""
        obs = EmptyObservability()  # Use a concrete implementation for testing
        assert isinstance(obs.prompts_dir, Path)
        assert obs.prompts_dir.exists()

    def test_init_custom_dir(self):
        """Test initialization with custom directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            obs = EmptyObservability(prompts_dir=temp_dir)
            assert obs.prompts_dir == Path(temp_dir)
            assert obs.prompts_dir.exists()

    def test_required_vars(self):
        """Test getting/setting required environment variables."""
        obs = EmptyObservability()
        assert obs.required_vars == []

        obs.required_vars = ["TEST_VAR1", "TEST_VAR2"]
        assert obs.required_vars == ["TEST_VAR1", "TEST_VAR2"]

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

            # Push a simple string template
            template = "Hello, {{ name }}! Welcome to {{ place }}."
            obs.push_prompt("greeting", template)

            # Pull the template back
            result = obs.pull_prompt("greeting")
            assert isinstance(result, ChatPromptTemplate)

            # Get the raw template
            raw = obs.get_template("greeting")
            assert raw == template

    def test_push_pull_chat_messages(self):
        """Test pushing and pulling chat message prompts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            obs = EmptyObservability(prompts_dir=temp_dir)

            # Create a list of chat messages
            messages: list[ChatMessageDict] = [
                {"role": "system", "content": "You are a helpful assistant for {{ domain }}."},
                {"role": "human", "content": "Help me with {{ topic }}."},
            ]

            # Push the template
            obs.push_prompt("chat-prompt", messages)

            # Pull the template back
            result = obs.pull_prompt("chat-prompt")
            assert isinstance(result, ChatPromptTemplate)

    def test_render_prompt(self):
        """Test rendering a prompt with variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            obs = EmptyObservability(prompts_dir=temp_dir)

            # Push a template
            template = "Hello, {{ name }}! Welcome to {{ place }}."
            obs.push_prompt("render-test", template)

            # Render the template
            rendered = obs.render_prompt("render-test", name="Alice", place="Wonderland")
            assert rendered == "Hello, Alice! Welcome to Wonderland."

    def test_delete_prompt(self):
        """Test deleting a prompt."""
        with tempfile.TemporaryDirectory() as temp_dir:
            obs = EmptyObservability(prompts_dir=temp_dir)

            # Push a template
            template = "Test template"
            obs.push_prompt("to-delete", template)

            # Verify it exists
            template_path = Path(temp_dir) / "to-delete.jinja2"
            assert template_path.exists()

            # Delete it
            obs.delete_prompt("to-delete")

            # Verify it's gone
            assert not template_path.exists()

    def test_push_with_metadata(self):
        """Test pushing a prompt with metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            obs = EmptyObservability(prompts_dir=temp_dir)

            # Push a template with metadata
            template = "Test with metadata"
            metadata = {"version": "1.0", "author": "Test Author"}
            obs.push_prompt("with-metadata", template, metadata=metadata)

            # Verify metadata file exists
            metadata_path = Path(temp_dir) / "with-metadata.metadata.joblib"
            assert metadata_path.exists()


class TestEmptyObservability:
    """Tests for the EmptyObservability class."""

    def test_callback_handler(self):
        """Test getting callback handler."""
        obs = EmptyObservability()
        assert obs.get_callback_handler() is None

    def test_before_shutdown(self):
        """Test before_shutdown method."""
        obs = EmptyObservability()
        # Should not raise an exception
        obs.before_shutdown()

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

        # Clear environment variables for test if they exist
        with patch.dict(os.environ, clear=True):
            # Now validation should fail
            with pytest.raises(ValueError, match="Missing required environment variables"):
                obs.get_callback_handler()

    @patch("langgraph_agent_toolkit.core.observability.langsmith.LangsmithClient")
    def test_push_prompt(self, mock_client_cls):
        """Test pushing a prompt to LangSmith."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.push_prompt.return_value = "https://api.smith.langchain.com/prompts/123"
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
                template = "Test LangSmith template for {{ topic }}"

                # Push the template
                obs.push_prompt("langsmith-test", template)

                # Assert client was called properly
                mock_client.push_prompt.assert_called_once()
                assert mock_client.push_prompt.call_args[0][0] == "langsmith-test"

    @patch("langgraph_agent_toolkit.core.observability.langsmith.LangsmithClient")
    def test_pull_prompt(self, mock_client_cls):
        """Test pulling a prompt from LangSmith."""
        # Setup mock
        mock_client = MagicMock()
        mock_prompt = MagicMock()
        mock_client.pull_prompt.return_value = mock_prompt
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

            # Pull the prompt
            result = obs.pull_prompt("langsmith-test")

            # Assert client was called
            mock_client.pull_prompt.assert_called_once_with("langsmith-test")
            assert result == mock_prompt

    @patch("langgraph_agent_toolkit.core.observability.langsmith.LangsmithClient")
    def test_delete_prompt(self, mock_client_cls):
        """Test deleting a prompt from LangSmith."""
        # Setup mock
        mock_client = MagicMock()
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

                # Create a local file to delete
                template_path = Path(temp_dir) / "to-delete.jinja2"
                template_path.write_text("Test template")

                # Delete the prompt
                obs.delete_prompt("to-delete")

                # Assert client was called and local file was deleted
                mock_client.delete_prompt.assert_called_once_with("to-delete")
                assert not template_path.exists()


class TestLangfuseObservability:
    """Tests for the LangfuseObservability class."""

    def test_requires_env_vars(self):
        """Test environment validation is required."""
        obs = LangfuseObservability()

        # Clear environment variables for test if they exist
        with patch.dict(os.environ, clear=True):
            # Validation should fail
            with pytest.raises(ValueError, match="Missing required environment variables"):
                obs.get_callback_handler()

    @patch("langgraph_agent_toolkit.core.observability.langfuse.Langfuse")
    def test_get_callback_handler(self, mock_langfuse_cls):
        """Test getting callback handler."""
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_SECRET_KEY": "secret",
                "LANGFUSE_PUBLIC_KEY": "public",
                "LANGFUSE_HOST": "https://cloud.langfuse.com",
            },
        ):
            obs = LangfuseObservability()
            handler = obs.get_callback_handler()
            assert handler is not None

    @patch("langgraph_agent_toolkit.core.observability.langfuse.Langfuse")
    def test_before_shutdown(self, mock_langfuse_cls):
        """Test before_shutdown method."""
        mock_langfuse = MagicMock()
        mock_langfuse_cls.return_value = mock_langfuse

        obs = LangfuseObservability()
        obs.before_shutdown()

        mock_langfuse.flush.assert_called_once()

    @patch("langgraph_agent_toolkit.core.observability.langfuse.Langfuse")
    @patch("langgraph_agent_toolkit.core.observability.base.joblib")
    def test_push_prompt(self, mock_joblib, mock_langfuse_cls):
        """Test pushing a prompt to Langfuse."""
        # Setup mock
        mock_langfuse = MagicMock()

        # Use a special mock object that can be serialized
        class SerializableMock:
            def __init__(self):
                self.id = "langfuse_prompt_id_123"

            def __getstate__(self):
                return {"id": self.id}

            def __setstate__(self, state):
                self.id = state["id"]

        mock_langfuse_prompt = SerializableMock()
        mock_langfuse.create_prompt.return_value = mock_langfuse_prompt
        mock_langfuse_cls.return_value = mock_langfuse

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

                # Create a chat prompt
                messages: list[ChatMessageDict] = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "human", "content": "Help with {{ topic }}"},
                ]

                # Push the prompt
                obs.push_prompt("langfuse-test", messages)

                # Assert Langfuse create_prompt was called with correct type
                mock_langfuse.create_prompt.assert_called_once()
                create_args = mock_langfuse.create_prompt.call_args[1]
                assert create_args["name"] == "langfuse-test"
                assert create_args["type"] == "chat"

                # Verify metadata with langfuse_prompt was saved
                mock_joblib.dump.assert_called_once()
                saved_metadata = mock_joblib.dump.call_args[0][0]
                assert "langfuse_prompt" in saved_metadata
                assert saved_metadata["langfuse_prompt"].id == "langfuse_prompt_id_123"

    @patch("langgraph_agent_toolkit.core.observability.langfuse.Langfuse")
    def test_pull_prompt(self, mock_langfuse_cls):
        """Test pulling a prompt from Langfuse."""
        # Setup mock
        mock_langfuse = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.get_langchain_prompt.return_value = "Returned langchain prompt"
        mock_langfuse.get_prompt.return_value = mock_prompt
        mock_langfuse_cls.return_value = mock_langfuse

        with patch.dict(
            os.environ,
            {
                "LANGFUSE_SECRET_KEY": "secret",
                "LANGFUSE_PUBLIC_KEY": "public",
                "LANGFUSE_HOST": "https://cloud.langfuse.com",
            },
        ):
            obs = LangfuseObservability()

            # Pull the prompt
            result = obs.pull_prompt("langfuse-test")

            # Assert Langfuse get_prompt was called
            mock_langfuse.get_prompt.assert_called_once()
            assert result == "Returned langchain prompt"

            # Test with return_with_prompt_object=True
            result, obj = obs.pull_prompt("langfuse-test", return_with_prompt_object=True)
            assert result == "Returned langchain prompt"
            assert obj == mock_prompt

    @patch("langgraph_agent_toolkit.core.observability.langfuse.Langfuse")
    def test_pull_prompt_fallback_to_local(self, mock_langfuse_cls):
        """Test pulling a prompt with fallback to local storage."""
        # Setup mock to raise an exception
        mock_langfuse = MagicMock()
        mock_langfuse.get_prompt.side_effect = Exception("API error")
        mock_langfuse_cls.return_value = mock_langfuse

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

                # Create a local prompt file
                template = "Local template for {{ topic }}"
                prompt_path = Path(temp_dir) / "local-fallback.jinja2"
                prompt_path.write_text(template)

                # Pull the prompt
                result = obs.pull_prompt("local-fallback")

                # Should fallback to local file
                assert isinstance(result, ChatPromptTemplate)

    @patch("langgraph_agent_toolkit.core.observability.langfuse.Langfuse")
    def test_record_feedback(self, mock_langfuse_cls):
        """Test recording feedback."""
        # Setup mock
        mock_langfuse = MagicMock()
        mock_langfuse_cls.return_value = mock_langfuse

        with patch.dict(
            os.environ,
            {
                "LANGFUSE_SECRET_KEY": "secret",
                "LANGFUSE_PUBLIC_KEY": "public",
                "LANGFUSE_HOST": "https://cloud.langfuse.com",
            },
        ):
            obs = LangfuseObservability()

            # Record feedback
            obs.record_feedback("trace-123", "accuracy", 0.95)

            # Assert score was called
            mock_langfuse.score.assert_called_once()
            score_args = mock_langfuse.score.call_args[1]
            assert score_args["trace_id"] == "trace-123"
            assert score_args["name"] == "accuracy"
            assert score_args["value"] == 0.95


def test_observability_factory():
    """Test creating observability instances based on backend type."""
    from langgraph_agent_toolkit.core.observability.factory import ObservabilityFactory

    # Test with EMPTY backend
    empty_obs = ObservabilityFactory.create(ObservabilityBackend.EMPTY)
    assert isinstance(empty_obs, EmptyObservability)
