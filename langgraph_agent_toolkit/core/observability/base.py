import functools
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar, cast

import joblib
from jinja2 import Template
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import (
    AIMessagePromptTemplate,
    BaseMessage,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from langgraph_agent_toolkit.core.observability.types import MessageRole, PromptReturnType, PromptTemplateType
from langgraph_agent_toolkit.helper.logging import logger


T = TypeVar("T")


class BaseObservabilityPlatform(ABC):
    """Base class for observability platforms."""

    __default_required_vars = []

    def __init__(self, prompts_dir: Optional[str] = None):
        self._required_vars = self.__default_required_vars.copy()

        if prompts_dir:
            self._prompts_dir = Path(prompts_dir)
        else:
            temp_base = Path(tempfile.gettempdir())
            self._prompts_dir = temp_base / "langgraph_prompts"

        self._prompts_dir.mkdir(exist_ok=True, parents=True)

    @property
    def prompts_dir(self) -> Path:
        return self._prompts_dir

    @prompts_dir.setter
    def prompts_dir(self, path: str) -> None:
        self._prompts_dir = Path(path)
        self._prompts_dir.mkdir(exist_ok=True, parents=True)

    @property
    def required_vars(self) -> List[str]:
        return self._required_vars

    @required_vars.setter
    def required_vars(self, value: List[str]) -> None:
        self._required_vars = value

    def validate_environment(self) -> bool:
        missing_vars = [var for var in self._required_vars if not os.environ.get(var)]

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        return True

    @staticmethod
    def requires_env_vars(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            self.validate_environment()
            return func(self, *args, **kwargs)

        return wrapper

    @abstractmethod
    def get_callback_handler(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def before_shutdown(self) -> None:
        pass

    @abstractmethod
    def record_feedback(self, run_id: str, key: str, score: float, **kwargs) -> None:
        pass

    def _handle_existing_prompt(
        self,
        name: str,
        create_new_version: bool = True,
        client: Any = None,
        client_pull_method: Optional[str] = None,
        client_delete_method: Optional[str] = None,
    ) -> Tuple[Any, Any]:
        existing_prompt = None
        url = None

        if not client or not client_pull_method or not client_delete_method:
            return (existing_prompt, url)

        pull_method = getattr(client, client_pull_method, None)
        delete_method = getattr(client, client_delete_method, None)

        if not pull_method or not delete_method:
            return (existing_prompt, url)

        if not create_new_version:
            try:
                existing_prompt = pull_method(name=name)
                url = getattr(existing_prompt, "url", None)
                logger.debug(f"Using existing prompt '{name}' as create_new_version is False")
            except Exception:
                logger.debug(f"Existing prompt '{name}' not found, will create a new one")
        else:
            try:
                pull_method(name=name)
                delete_method(name=name)
                logger.debug(f"Deleted existing prompt '{name}' to create new version")
            except Exception:
                pass

        return (existing_prompt, url)

    def _convert_to_chat_prompt(self, prompt_template: PromptTemplateType) -> ChatPromptTemplate:
        if isinstance(prompt_template, str):
            return ChatPromptTemplate.from_template(prompt_template)
        elif isinstance(prompt_template, list) and all(isinstance(msg, dict) for msg in prompt_template):
            messages = []
            for msg in prompt_template:
                role = msg.get("role", "")
                content = msg.get("content", "")

                match role.lower():
                    case MessageRole.SYSTEM:
                        messages.append(SystemMessage(content=content))
                    case MessageRole.HUMAN | MessageRole.USER:
                        messages.append(HumanMessage(content=content))
                    case MessageRole.AI | MessageRole.ASSISTANT:
                        messages.append(AIMessage(content=content))
                    case MessageRole.PLACEHOLDER | MessageRole.MESSAGES_PLACEHOLDER:
                        messages.append(MessagesPlaceholder(variable_name=content))
                    case _:
                        raise ValueError(f"Unknown message role: {role}")

            return ChatPromptTemplate.from_messages(messages)
        else:
            return cast(ChatPromptTemplate, prompt_template)

    def _process_messages_from_prompt(
        self, messages: List[Any], template_format: Literal["f-string", "mustache", "jinja2"] = "f-string"
    ) -> List[Any]:
        MESSAGE_TYPE_MAP = {
            MessageRole.SYSTEM: SystemMessagePromptTemplate,
            MessageRole.HUMAN: HumanMessagePromptTemplate,
            MessageRole.USER: HumanMessagePromptTemplate,
            MessageRole.AI: AIMessagePromptTemplate,
            MessageRole.ASSISTANT: AIMessagePromptTemplate,
        }

        processed_messages = []

        for msg in messages:
            if isinstance(msg, MessagesPlaceholder):
                processed_messages.append(msg)
                continue

            if isinstance(msg, BaseMessage):
                msg_type = MessageRole(msg.type)
                if msg_type in MESSAGE_TYPE_MAP:
                    template_class = MESSAGE_TYPE_MAP[msg_type]
                    processed_messages.append(
                        template_class.from_template(msg.content, template_format=template_format)
                    )
                continue

            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                role, content = msg["role"], msg["content"]

                if role in MESSAGE_TYPE_MAP:
                    template_class = MESSAGE_TYPE_MAP[role]
                    processed_messages.append(template_class.from_template(content, template_format=template_format))
                    continue

                if role.lower() in (MessageRole.PLACEHOLDER, MessageRole.MESSAGES_PLACEHOLDER):
                    processed_messages.append(MessagesPlaceholder(variable_name=content))
                    continue

            if isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                if role in MESSAGE_TYPE_MAP:
                    template_class = MESSAGE_TYPE_MAP[role]
                    processed_messages.append(template_class.from_template(content, template_format=template_format))
                    continue

                if role.lower() in (MessageRole.PLACEHOLDER, MessageRole.MESSAGES_PLACEHOLDER):
                    processed_messages.append(MessagesPlaceholder(variable_name=content))
                    continue

            processed_messages.append(msg)

        return processed_messages

    def _process_prompt_object(
        self, prompt_obj: Any, template_format: Literal["f-string", "mustache", "jinja2"] = "f-string"
    ) -> ChatPromptTemplate:
        if isinstance(prompt_obj, ChatPromptTemplate):
            return prompt_obj

        if hasattr(prompt_obj, "messages") and isinstance(prompt_obj.messages, list):
            processed_messages = self._process_messages_from_prompt(
                prompt_obj.messages, template_format=template_format
            )
            if processed_messages:
                return ChatPromptTemplate.from_messages(processed_messages)

        elif isinstance(prompt_obj, list):
            if all(isinstance(item, dict) and "role" in item and "content" in item for item in prompt_obj):
                processed_messages = self._process_messages_from_prompt(prompt_obj, template_format=template_format)
                if processed_messages:
                    return ChatPromptTemplate.from_messages(processed_messages)

        elif isinstance(prompt_obj, str):
            return ChatPromptTemplate.from_template(prompt_obj, template_format=template_format)

        else:
            raise ValueError(f"Could not process prompt object of type {type(prompt_obj)}")

    def _extract_template_string(self, prompt_template: PromptTemplateType, prompt_obj: Any) -> str:
        if isinstance(prompt_template, str):
            return prompt_template
        elif isinstance(prompt_template, list) and all(isinstance(msg, dict) for msg in prompt_template):
            template_str = ""
            for msg in prompt_template:
                template_str += f"[{msg['role']}]: {msg['content']}\n\n"
            return template_str
        else:
            if hasattr(prompt_obj, "template"):
                return prompt_obj.template
            return str(prompt_obj)

    def _local_pull_prompt(self, name: str, template_format: str = "f-string", **kwargs) -> PromptReturnType:
        """Local implementation of pull_prompt that reads from the file system."""
        file_path = self._prompts_dir / f"{name}.jinja2"

        if not file_path.exists():
            raise ValueError(f"Prompt '{name}' not found at {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        metadata_path = self._prompts_dir / f"{name}.metadata.joblib"

        if metadata_path.exists():
            try:
                metadata = joblib.load(metadata_path)
                original_prompt = metadata.get("original_prompt")
                if original_prompt:
                    return original_prompt
            except Exception:
                pass

        return ChatPromptTemplate.from_template(template_content, template_format=template_format)

    def pull_prompt(
        self, name: str, template_format: Literal["f-string", "mustache", "jinja2"] = "f-string", **kwargs
    ) -> PromptReturnType:
        """Pull a prompt from the observability platform."""
        # Use the local implementation
        return self._local_pull_prompt(name, template_format=template_format, **kwargs)

    def push_prompt(
        self,
        name: str,
        prompt_template: PromptTemplateType,
        metadata: Optional[Dict[str, Any]] = None,
        create_new_version: bool = True,
    ) -> None:
        self._prompts_dir.mkdir(exist_ok=True, parents=True)

        file_path = self._prompts_dir / f"{name}.jinja2"
        metadata_path = self._prompts_dir / f"{name}.metadata.joblib"

        if create_new_version:
            if file_path.exists():
                file_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()

        chat_prompt = self._convert_to_chat_prompt(prompt_template)
        template_str = self._extract_template_string(prompt_template, chat_prompt)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(template_str))

        full_metadata = metadata.copy() if metadata else {}
        if not isinstance(prompt_template, str):
            full_metadata["original_prompt"] = chat_prompt
            full_metadata["original_format"] = "chat_message_dict" if isinstance(prompt_template, list) else "other"
            joblib.dump(full_metadata, metadata_path)
        elif metadata:
            joblib.dump(full_metadata, metadata_path)

    def get_template(self, name: str) -> str:
        file_path = self._prompts_dir / f"{name}.jinja2"

        if not file_path.exists():
            raise ValueError(f"Prompt '{name}' not found at {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def render_prompt(self, prompt_name: str, **variables) -> str:
        template_content = self.get_template(prompt_name)
        template = Template(template_content)
        return template.render(**variables)

    def delete_prompt(self, name: str) -> None:
        file_path = self._prompts_dir / f"{name}.jinja2"
        metadata_path = self._prompts_dir / f"{name}.metadata.joblib"
        json_metadata_path = self._prompts_dir / f"{name}.metadata.json"

        if file_path.exists():
            file_path.unlink()

        if metadata_path.exists():
            metadata_path.unlink()

        if json_metadata_path.exists():
            json_metadata_path.unlink()
