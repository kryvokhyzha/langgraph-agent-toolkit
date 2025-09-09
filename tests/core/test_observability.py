import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.prompts import ChatPromptTemplate

from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.core.observability.langfuse import LangfuseObservability
from langgraph_agent_toolkit.core.observability.langsmith import LangsmithObservability
from langgraph_agent_toolkit.core.observability.types import ChatMessageDict, ObservabilityBackend


class TestBaseObservability:
    """Tests for the BaseObservabilityPlatform class."""

    def test_init_defaults(self):
        """Test initialization with default settings."""
        obs = EmptyObservability()  # Use a concrete implementation for testing
        assert isinstance(obs.prompts_dir, Path)
        assert obs.prompts_dir.exists()
        assert obs.remote_first is False

    def test_init_custom_dir(self):
        """Test initialization with custom directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            obs = EmptyObservability(prompts_dir=temp_dir)
            assert obs.prompts_dir == Path(temp_dir)
            assert obs.prompts_dir.exists()
            assert obs.remote_first is False

    def test_init_with_remote_first(self):
        """Test initialization with remote_first flag."""
        obs = EmptyObservability(remote_first=True)
        assert obs.remote_first is True

        with tempfile.TemporaryDirectory() as temp_dir:
            obs_with_dir = EmptyObservability(prompts_dir=temp_dir, remote_first=True)
            assert obs_with_dir.prompts_dir == Path(temp_dir)
            assert obs_with_dir.remote_first is True

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

    def test_handle_existing_prompt(self):
        """Test the _handle_existing_prompt helper method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            obs = EmptyObservability(prompts_dir=temp_dir)

            # Create a mock client for testing
            mock_client = MagicMock()
            mock_prompt = MagicMock()
            mock_prompt.url = "https://test-url.com/prompts/123"
            mock_client.pull_prompt = MagicMock(return_value=mock_prompt)
            mock_client.delete_prompt = MagicMock()

            # Test with force_create_new_version=False (should return existing prompt)
            existing_prompt, url = obs._handle_existing_prompt(
                "test-prompt",
                force_create_new_version=False,
                client=mock_client,
                client_pull_method="pull_prompt",
                client_delete_method="delete_prompt",
            )

            assert existing_prompt == mock_prompt
            assert url == "https://test-url.com/prompts/123"
            mock_client.pull_prompt.assert_called_once_with(name="test-prompt")
            mock_client.delete_prompt.assert_not_called()

            # Reset mocks
            mock_client.pull_prompt.reset_mock()

            # Test with force_create_new_version=True (should delete and not return existing prompt)
            existing_prompt, url = obs._handle_existing_prompt(
                "test-prompt",
                force_create_new_version=True,
                client=mock_client,
                client_pull_method="pull_prompt",
                client_delete_method="delete_prompt",
            )

            assert existing_prompt is None
            assert url is None
            mock_client.pull_prompt.assert_called_once_with(name="test-prompt")
            mock_client.delete_prompt.assert_called_once_with(name="test-prompt")


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
        # Create a mock prompt that can be properly processed
        mock_prompt = MagicMock()
        mock_prompt.__str__ = lambda self: "Test mock prompt template"
        # Add a template attribute to the mock
        mock_prompt.template = "Test template string from mock"
        mock_client.pull_prompt.return_value = mock_prompt
        mock_client_cls.return_value = mock_client

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a local file as fallback
            template_path = Path(temp_dir) / "langsmith-test.jinja2"
            template_path.write_text("Local test template")

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

                # Pull the prompt
                result = obs.pull_prompt("langsmith-test")

                # Assert client was called
                mock_client.pull_prompt.assert_called_once_with("langsmith-test")
                assert result is not None

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

    @patch("langgraph_agent_toolkit.core.observability.langsmith.LangsmithClient")
    def test_push_prompt_remote_first_existing_remote(self, mock_client_cls):
        """Test push_prompt with remote_first=True when remote prompt exists."""
        # Setup mock
        mock_client = MagicMock()
        mock_existing_prompt = MagicMock()
        mock_existing_prompt.url = "https://api.smith.langchain.com/prompts/456"
        mock_client.pull_prompt.return_value = mock_existing_prompt
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
                obs = LangsmithObservability(prompts_dir=temp_dir, remote_first=True)
                template = "Test remote-first template for {{ topic }}"

                # Push the template - should use existing remote prompt
                obs.push_prompt("remote-first-test", template)

                # Should check for existing remote prompt first
                mock_client.pull_prompt.assert_called_once_with("remote-first-test")

                # Should NOT create new prompt since remote exists and remote_first=True
                mock_client.push_prompt.assert_not_called()

    @patch("langgraph_agent_toolkit.core.observability.langsmith.LangsmithClient")
    def test_push_prompt_remote_first_no_remote(self, mock_client_cls):
        """Test push_prompt with remote_first=True when remote prompt doesn't exist."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.pull_prompt.side_effect = Exception("Prompt not found")
        mock_client.push_prompt.return_value = "https://api.smith.langchain.com/prompts/789"
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
                obs = LangsmithObservability(prompts_dir=temp_dir, remote_first=True)
                template = "Test new remote-first template for {{ topic }}"

                # Push the template - should create new since remote doesn't exist
                obs.push_prompt("remote-first-new-test", template)

                # Should check for existing remote prompt first - called twice (remote_first + regular logic)
                assert mock_client.pull_prompt.call_count == 2
                mock_client.pull_prompt.assert_any_call("remote-first-new-test")

                # Should create new prompt since remote doesn't exist
                mock_client.push_prompt.assert_called_once()
                assert mock_client.push_prompt.call_args[0][0] == "remote-first-new-test"


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

        # Configure the mock prompt to have required attributes and methods
        mock_prompt.prompt = "Test prompt content"

        # Create a ChatPromptTemplate for the return value, matching what the implementation expects
        chat_prompt = ChatPromptTemplate.from_template("Test prompt content")
        mock_prompt.get_langchain_prompt = MagicMock(return_value=chat_prompt)

        mock_langfuse.get_prompt.return_value = mock_prompt
        mock_langfuse_cls.return_value = mock_langfuse

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a local file as fallback
            template_path = Path(temp_dir) / "langfuse-test.jinja2"
            template_path.write_text("Local test template")

            with patch.dict(
                os.environ,
                {
                    "LANGFUSE_SECRET_KEY": "secret",
                    "LANGFUSE_PUBLIC_KEY": "public",
                    "LANGFUSE_HOST": "https://cloud.langfuse.com",
                },
            ):
                obs = LangfuseObservability(prompts_dir=temp_dir)

                # Pull the prompt
                result = obs.pull_prompt("langfuse-test")

                # Assert Langfuse get_prompt was called - use the actual default value (600 seconds)
                mock_langfuse.get_prompt.assert_called_once_with(name="langfuse-test", cache_ttl_seconds=600)

                # Assert that result is the ChatPromptTemplate returned by get_langchain_prompt
                assert result == chat_prompt
                assert isinstance(result, ChatPromptTemplate)

                # Test with return_with_prompt_object=True
                result, obj = obs.pull_prompt("langfuse-test", return_with_prompt_object=True)
                assert result == chat_prompt
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

    def test_compute_prompt_hash(self):
        """Test the hash computation for different prompt types."""
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_SECRET_KEY": "secret",
                "LANGFUSE_PUBLIC_KEY": "public",
                "LANGFUSE_HOST": "https://cloud.langfuse.com",
            },
        ):
            obs = LangfuseObservability()

            # Test string hash
            str_hash = obs._compute_prompt_hash("Test prompt")
            assert isinstance(str_hash, str)
            assert len(str_hash) > 0

            # Test list hash
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "human", "content": "Help with a task"},
            ]
            list_hash = obs._compute_prompt_hash(messages)
            assert isinstance(list_hash, str)
            assert len(list_hash) > 0

            # Test chat prompt hash
            chat_prompt = ChatPromptTemplate.from_template("Test template")
            obj_hash = obs._compute_prompt_hash(chat_prompt)
            assert isinstance(obj_hash, str)
            assert len(obj_hash) > 0

    @patch("langgraph_agent_toolkit.core.observability.langfuse.Langfuse")
    @patch("langgraph_agent_toolkit.core.observability.base.joblib.dump")
    def test_push_prompt_with_unchanged_content(self, mock_dump, mock_langfuse_cls):
        """Test prompt is not recreated when content is unchanged (same hash)."""
        # Setup mock
        mock_langfuse = MagicMock()

        # Create a simple prompt to get its hash
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "human", "content": "Help with {{ topic }}"},
        ]

        # Create a temporary LangfuseObservability instance just to get the hash
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_SECRET_KEY": "secret",
                "LANGFUSE_PUBLIC_KEY": "public",
                "LANGFUSE_HOST": "https://cloud.langfuse.com",
            },
        ):
            temp_obs = LangfuseObservability()
            expected_hash = temp_obs._compute_prompt_hash(messages)

        # Create a mock prompt with matching hash
        mock_existing_prompt = MagicMock()
        mock_existing_prompt.id = "existing_prompt_id_456"
        mock_existing_prompt.commit_message = expected_hash  # Same hash = unchanged content
        mock_langfuse.get_prompt.return_value = mock_existing_prompt

        mock_langfuse_cls.return_value = mock_langfuse
        mock_dump.return_value = None

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

                # Push the prompt with force_create_new_version=False to avoid creating new version
                # even though content is already unchanged
                obs.push_prompt("unchanged-prompt", messages, force_create_new_version=False)

                # Assert that get_prompt was called
                mock_langfuse.get_prompt.assert_called_once_with(name="unchanged-prompt")

                # Content is unchanged and force_create_new_version=False, so no new version should be created
                mock_langfuse.create_prompt.assert_not_called()

    @patch("langgraph_agent_toolkit.core.observability.langfuse.Langfuse")
    @patch("langgraph_agent_toolkit.core.observability.base.joblib.dump")
    def test_push_prompt_with_changed_content(self, mock_dump, mock_langfuse_cls):
        """Test new prompt version is created when content changes."""
        # Setup mock
        mock_langfuse = MagicMock()

        # Create a mock prompt with a different hash
        mock_existing_prompt = MagicMock()
        mock_existing_prompt.id = "existing_prompt_id_456"
        mock_existing_prompt.commit_message = "different_hash_value"  # Different hash = changed content
        mock_langfuse.get_prompt.return_value = mock_existing_prompt

        # Mock for new prompt
        mock_new_prompt = MagicMock()
        mock_new_prompt.id = "new_prompt_id_123"
        mock_langfuse.create_prompt.return_value = mock_new_prompt

        mock_langfuse_cls.return_value = mock_langfuse
        mock_dump.return_value = None

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

                # Push the prompt with default force_create_new_version=True
                obs.push_prompt("changed-prompt", messages)

                # Assert that get_prompt was called
                mock_langfuse.get_prompt.assert_called_once_with(name="changed-prompt")

                # Content has changed and force_create_new_version=True, so create_prompt should be called
                mock_langfuse.create_prompt.assert_called_once()
                call_kwargs = mock_langfuse.create_prompt.call_args[1]
                assert call_kwargs["name"] == "changed-prompt"
                # Check that we passed the hash in commit_message
                prompt_hash = obs._compute_prompt_hash(messages)
                assert call_kwargs["commit_message"] == prompt_hash

    @patch("langgraph_agent_toolkit.core.observability.langfuse.Langfuse")
    @patch("langgraph_agent_toolkit.core.observability.base.joblib.dump")
    def test_push_prompt_force_new_version(self, mock_dump, mock_langfuse_cls):
        """Test new version is created when force_create_new_version=True even if content unchanged."""
        # Setup mock
        mock_langfuse = MagicMock()

        # Create a simple prompt to get its hash
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "human", "content": "Help with {{ topic }}"},
        ]

        # Create a temporary LangfuseObservability instance just to get the hash
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_SECRET_KEY": "secret",
                "LANGFUSE_PUBLIC_KEY": "public",
                "LANGFUSE_HOST": "https://cloud.langfuse.com",
            },
        ):
            temp_obs = LangfuseObservability()
            expected_hash = temp_obs._compute_prompt_hash(messages)

        # Create a mock prompt with matching hash
        mock_existing_prompt = MagicMock()
        mock_existing_prompt.id = "existing_prompt_id_456"
        mock_existing_prompt.commit_message = expected_hash  # Same hash = unchanged content
        mock_langfuse.get_prompt.return_value = mock_existing_prompt

        # Mock for new prompt
        mock_new_prompt = MagicMock()
        mock_new_prompt.id = "new_prompt_id_123"
        mock_langfuse.create_prompt.return_value = mock_new_prompt

        mock_langfuse_cls.return_value = mock_langfuse
        mock_dump.return_value = None

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

                # Force new version creation even with unchanged content
                obs.push_prompt("force-new-version", messages, force_create_new_version=True)

                # Assert that get_prompt was called
                mock_langfuse.get_prompt.assert_called_once_with(name="force-new-version")

                # force_create_new_version=True should force new version even if content unchanged
                mock_langfuse.create_prompt.assert_called_once()
                # Check that we passed the hash in commit_message
                call_kwargs = mock_langfuse.create_prompt.call_args[1]
                assert call_kwargs["commit_message"] == expected_hash

    @patch("langgraph_agent_toolkit.core.observability.langfuse.Langfuse")
    @patch("langgraph_agent_toolkit.core.observability.base.joblib.dump")
    def test_push_prompt_remote_first_existing_remote(self, mock_dump, mock_langfuse_cls):
        """Test push_prompt with remote_first=True when remote prompt exists."""
        # Setup mock
        mock_langfuse = MagicMock()

        # Mock existing remote prompt
        mock_existing_prompt = MagicMock()
        mock_existing_prompt.id = "remote_prompt_id_789"
        mock_langfuse.get_prompt.return_value = mock_existing_prompt

        mock_langfuse_cls.return_value = mock_langfuse
        mock_dump.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                os.environ,
                {
                    "LANGFUSE_SECRET_KEY": "secret",
                    "LANGFUSE_PUBLIC_KEY": "public",
                    "LANGFUSE_HOST": "https://cloud.langfuse.com",
                },
            ):
                obs = LangfuseObservability(prompts_dir=temp_dir, remote_first=True)

                messages: list[ChatMessageDict] = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "human", "content": "Help with {{ topic }}"},
                ]

                # Push the prompt - should use existing remote prompt
                obs.push_prompt("remote-first-test", messages)

                # Should check for existing remote prompt first
                mock_langfuse.get_prompt.assert_called_once_with(name="remote-first-test")

                # Should NOT create new prompt since remote exists and remote_first=True
                mock_langfuse.create_prompt.assert_not_called()

    @patch("langgraph_agent_toolkit.core.observability.langfuse.Langfuse")
    @patch("langgraph_agent_toolkit.core.observability.base.joblib.dump")
    def test_push_prompt_remote_first_no_remote(self, mock_dump, mock_langfuse_cls):
        """Test push_prompt with remote_first=True when remote prompt doesn't exist."""
        # Setup mock
        mock_langfuse = MagicMock()

        # Mock that remote prompt doesn't exist
        mock_langfuse.get_prompt.side_effect = Exception("Prompt not found")

        # Mock new prompt creation
        mock_new_prompt = MagicMock()
        mock_new_prompt.id = "new_remote_prompt_id_101"
        mock_langfuse.create_prompt.return_value = mock_new_prompt

        mock_langfuse_cls.return_value = mock_langfuse
        mock_dump.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                os.environ,
                {
                    "LANGFUSE_SECRET_KEY": "secret",
                    "LANGFUSE_PUBLIC_KEY": "public",
                    "LANGFUSE_HOST": "https://cloud.langfuse.com",
                },
            ):
                obs = LangfuseObservability(prompts_dir=temp_dir, remote_first=True)

                messages: list[ChatMessageDict] = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "human", "content": "Help with {{ topic }}"},
                ]

                # Push the prompt - should create new since remote doesn't exist
                obs.push_prompt("remote-first-new-test", messages)

                # Should check for existing remote prompt first - called twice (remote_first + regular logic)
                assert mock_langfuse.get_prompt.call_count == 2
                mock_langfuse.get_prompt.assert_any_call(name="remote-first-new-test")

                # Should create new prompt since remote doesn't exist
                mock_langfuse.create_prompt.assert_called_once()
                call_kwargs = mock_langfuse.create_prompt.call_args[1]
                assert call_kwargs["name"] == "remote-first-new-test"


def test_observability_factory():
    """Test creating observability instances based on backend type."""
    from langgraph_agent_toolkit.core.observability.factory import ObservabilityFactory

    # Test with EMPTY backend
    empty_obs = ObservabilityFactory.create(ObservabilityBackend.EMPTY)
    assert isinstance(empty_obs, EmptyObservability)
