import functools
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

import joblib
from jinja2 import Template
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph_agent_toolkit.core.observability.types import PromptReturnType, PromptTemplateType


T = TypeVar("T")


class BaseObservabilityPlatform(ABC):
    """Base class for observability platforms."""

    __default_required_vars = []

    def __init__(self, prompts_dir: Optional[str] = None):
        """Initialize the observability platform."""
        self._required_vars = self.__default_required_vars.copy()

        if prompts_dir:
            self._prompts_dir = Path(prompts_dir)
        else:
            temp_base = Path(tempfile.gettempdir())
            self._prompts_dir = temp_base / "langgraph_prompts"

        self._prompts_dir.mkdir(exist_ok=True, parents=True)

    @property
    def prompts_dir(self) -> Path:
        """Get the directory where prompts are stored."""
        return self._prompts_dir

    @prompts_dir.setter
    def prompts_dir(self, path: str) -> None:
        """Set the directory where prompts are stored."""
        self._prompts_dir = Path(path)
        self._prompts_dir.mkdir(exist_ok=True, parents=True)

    @property
    def required_vars(self) -> List[str]:
        """Return the name of the observability platform."""
        return self._required_vars

    @required_vars.setter
    def required_vars(self, value: List[str]) -> None:
        """Set the name of the observability platform."""
        self._required_vars = value

    def validate_environment(self) -> bool:
        """Validate that all necessary environment variables are set."""
        missing_vars = [var for var in self._required_vars if not os.environ.get(var)]

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        return True

    @staticmethod
    def requires_env_vars(func: Callable[..., T]) -> Callable[..., T]:
        """Validate environment variables before executing a function."""

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Call validate_environment directly to raise the exception
            self.validate_environment()
            return func(self, *args, **kwargs)

        return wrapper

    @abstractmethod
    def get_callback_handler(self, **kwargs) -> Any:
        """Get the callback handler for the observability platform."""
        pass

    @abstractmethod
    def before_shutdown(self) -> None:
        """Perform any necessary cleanup before shutdown."""
        pass

    @abstractmethod
    def record_feedback(self, run_id: str, key: str, score: float, **kwargs) -> None:
        """Record feedback for a run."""
        pass

    def _convert_to_chat_prompt(self, prompt_template: PromptTemplateType) -> ChatPromptTemplate:
        """Convert different prompt formats to a ChatPromptTemplate."""
        if isinstance(prompt_template, str):
            return ChatPromptTemplate.from_template(prompt_template)
        elif isinstance(prompt_template, list) and all(isinstance(msg, dict) for msg in prompt_template):
            messages = []
            for msg in prompt_template:
                if msg["role"] == "system":
                    messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "human":
                    messages.append(HumanMessage(content=msg["content"]))
                # Add other role types if needed
            return ChatPromptTemplate.from_messages(messages)
        else:
            # If it's already a prompt template or another object, return as is
            return cast(ChatPromptTemplate, prompt_template)

    def push_prompt(
        self,
        name: str,
        prompt_template: PromptTemplateType,
        metadata: Optional[Dict[str, Any]] = None,
        create_new_version: bool = True,
    ) -> None:
        """Push a prompt to the observability platform."""
        self._prompts_dir.mkdir(exist_ok=True, parents=True)

        file_path = self._prompts_dir / f"{name}.jinja2"
        metadata_path = self._prompts_dir / f"{name}.metadata.joblib"

        if create_new_version:
            if file_path.exists():
                file_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()

        # Handle different prompt formats
        chat_prompt = self._convert_to_chat_prompt(prompt_template)

        # Get template string and class info
        if isinstance(prompt_template, str):
            template_str = prompt_template
        elif isinstance(prompt_template, list) and all(isinstance(msg, dict) for msg in prompt_template):
            # For list of message dicts, serialize to a readable format
            template_str = ""
            for msg in prompt_template:
                template_str += f"[{msg['role']}]: {msg['content']}\n\n"
        else:
            # Try to extract template string
            if hasattr(chat_prompt, "template"):
                template_str = chat_prompt.template
            else:
                template_str = str(chat_prompt)

        # Save the template
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(template_str))

        # Save metadata with original prompt for exact reconstruction
        full_metadata = metadata.copy() if metadata else {}
        if not isinstance(prompt_template, str):
            full_metadata["original_prompt"] = chat_prompt
            full_metadata["original_format"] = "chat_message_dict" if isinstance(prompt_template, list) else "other"
            joblib.dump(full_metadata, metadata_path)
        elif metadata:
            joblib.dump(full_metadata, metadata_path)

    def pull_prompt(self, name: str) -> PromptReturnType:
        """Pull a prompt from the observability platform.

        Args:
            name: Name of the prompt to pull

        Returns:
            The prompt object or template string

        """
        file_path = self._prompts_dir / f"{name}.jinja2"

        if not file_path.exists():
            raise ValueError(f"Prompt '{name}' not found at {file_path}")

        # Read the template
        with open(file_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        # Try to load metadata with joblib
        metadata_path = self._prompts_dir / f"{name}.metadata.joblib"

        if metadata_path.exists():
            try:
                metadata = joblib.load(metadata_path)
                original_prompt = metadata.get("original_prompt")
                if original_prompt:
                    return original_prompt
            except Exception:
                pass

        return ChatPromptTemplate.from_template(template_content, template_format="jinja2")

    def get_template(self, name: str) -> str:
        """Get just the template string for a prompt.

        Args:
            name: Name of the prompt

        Returns:
            The template string

        """
        file_path = self._prompts_dir / f"{name}.jinja2"

        if not file_path.exists():
            raise ValueError(f"Prompt '{name}' not found at {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def render_prompt(self, prompt_name: str, **variables) -> str:
        """Render a prompt with provided variables.

        Args:
            prompt_name: Name of the prompt to render
            **variables: Variables to use in rendering

        Returns:
            Rendered prompt string

        """
        template_content = self.get_template(prompt_name)
        template = Template(template_content)
        return template.render(**variables)

    def delete_prompt(self, name: str) -> None:
        """Delete a prompt from storage.

        Args:
            name: Name of the prompt to delete

        """
        file_path = self._prompts_dir / f"{name}.jinja2"
        metadata_path = self._prompts_dir / f"{name}.metadata.joblib"
        json_metadata_path = self._prompts_dir / f"{name}.metadata.json"

        if file_path.exists():
            file_path.unlink()

        if metadata_path.exists():
            metadata_path.unlink()

        if json_metadata_path.exists():
            json_metadata_path.unlink()
