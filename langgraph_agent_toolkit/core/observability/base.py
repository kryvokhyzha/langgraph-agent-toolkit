import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, cast

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


T = TypeVar("T")


class BaseObservabilityPlatform(ABC):
    """Base class for observability platforms.

    This is a lightweight base class that provides common utilities for
    observability platforms. It does NOT perform any disk I/O operations
    to avoid blocking in async contexts.
    """

    def __init__(self, remote_first: bool = False):
        """Initialize the observability platform.

        Args:
            remote_first: If True, prioritize remote prompts over local cache.

        """
        self._required_vars: List[str] = []
        self._remote_first = remote_first

    @property
    def remote_first(self) -> bool:
        return self._remote_first

    @property
    def required_vars(self) -> List[str]:
        return self._required_vars

    @required_vars.setter
    def required_vars(self, value: List[str]) -> None:
        self._required_vars = value

    def validate_environment(self) -> bool:
        """Validate that required environment variables are set."""
        missing_vars = [var for var in self._required_vars if not os.environ.get(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        return True

    @staticmethod
    def requires_env_vars(func: Callable[..., T]) -> Callable[..., T]:
        """Validate environment variables before calling a method."""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
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

    @abstractmethod
    def push_prompt(
        self,
        name: str,
        prompt_template: PromptTemplateType,
        metadata: Optional[Dict[str, Any]] = None,
        force_create_new_version: bool = True,
    ) -> None:
        """Push a prompt to the observability platform."""
        pass

    @abstractmethod
    def pull_prompt(
        self,
        name: str,
        template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
        **kwargs,
    ) -> PromptReturnType:
        """Pull a prompt from the observability platform."""
        pass

    @abstractmethod
    def delete_prompt(self, name: str) -> None:
        """Delete a prompt from the observability platform."""
        pass

    @contextmanager
    def trace_context(self, run_id: str, **kwargs):
        """Create a trace context for the execution.

        Override in subclasses for platform-specific implementation.

        Args:
            run_id: The run ID to use as trace ID
            **kwargs: Additional context parameters (user_id, input, etc.)

        Yields:
            None (or platform-specific context object)

        """
        yield

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
        self,
        messages: List[Any],
        template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
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
        self,
        prompt_obj: Any,
        template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
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

        if isinstance(prompt_obj, str):
            return ChatPromptTemplate.from_template(prompt_obj, template_format=template_format)

        raise ValueError(f"Could not process prompt object of type {type(prompt_obj)}")
