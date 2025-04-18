import inspect
import re
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from langchain_core.prompt_values import PromptValue
from langchain_core.prompts.chat import (
    AIMessagePromptTemplate,
    BaseChatPromptTemplate,
    BaseMessage,
    BaseMessagePromptTemplate,
    ChatPromptTemplate,
    ChatPromptValue,
    HumanMessagePromptTemplate,
    MessageLikeRepresentation,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.string import get_template_variables
from pydantic import Field

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform
from langgraph_agent_toolkit.core.observability.factory import ObservabilityFactory
from langgraph_agent_toolkit.core.observability.types import MessageRole, ObservabilityBackend, PromptReturnType
from langgraph_agent_toolkit.helper.constants import DEFAULT_CACHE_TTL_SECOND
from langgraph_agent_toolkit.helper.logging import logger


def _convert_template_format(content: str, target_format: str) -> str:
    """Convert template string between different formats."""
    if not content or not isinstance(content, str):
        return content

    if target_format == "jinja2" and "{" in content and "{{" not in content:
        # Convert f-string format to Jinja2 format
        return re.sub(r"{(\w+)}", r"{{ \1 }}", content)
    elif target_format == "f-string" and "{{" in content:
        # Convert Jinja2 format to f-string format
        return re.sub(r"{{\s*(\w+)\s*}}", r"{\1}", content)
    return content


class ObservabilityChatPromptTemplate(ChatPromptTemplate):
    """A chat prompt template that loads prompts from observability platforms."""

    prompt_name: Optional[str] = Field(default=None, description="Name of the prompt to load")
    prompt_version: Optional[int] = Field(default=None, description="Version of the prompt")
    prompt_label: Optional[str] = Field(default=None, description="Label of the prompt")
    load_at_runtime: bool = Field(default=False, description="Whether to load prompt at runtime")
    observability_backend: Optional[ObservabilityBackend] = Field(
        default=None, description="Observability backend to use"
    )
    cache_ttl_seconds: int = Field(default=DEFAULT_CACHE_TTL_SECOND, description="Cache TTL for prompts")
    template_format: str = Field(default="f-string", description="Format of the template")

    _observability_platform: Optional[BaseObservabilityPlatform] = None
    _loaded_prompt: Any = None
    _last_load_time: float = 0

    model_config = {"extra": "allow"}

    def __init__(
        self,
        messages: Optional[Sequence[MessageLikeRepresentation]] = None,
        *,
        prompt_name: Optional[str] = None,
        prompt_version: Optional[int] = None,
        prompt_label: Optional[str] = None,
        load_at_runtime: bool = False,
        observability_platform: Optional[BaseObservabilityPlatform] = None,
        observability_backend: Optional[Union[ObservabilityBackend, str]] = None,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECOND,
        template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
        input_variables: Optional[List[str]] = None,
        partial_variables: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize ObservabilityChatPromptTemplate."""
        # Process observability platform/backend
        _observability_platform = observability_platform
        _observability_backend = observability_backend

        if not observability_platform and observability_backend:
            _observability_backend = (
                ObservabilityBackend(observability_backend)
                if isinstance(observability_backend, str)
                else observability_backend
            )
            _observability_platform = ObservabilityFactory.create(_observability_backend)

        # Load messages if needed
        _messages = messages or []
        if not load_at_runtime and prompt_name and _observability_platform:
            try:
                self._loaded_prompt = loaded_prompt = self._load_prompt_from_platform(
                    _observability_platform,
                    prompt_name=prompt_name,
                    prompt_version=prompt_version,
                    prompt_label=prompt_label,
                    cache_ttl_seconds=cache_ttl_seconds,
                    template_format=template_format,
                )

                # Process returned prompt based on type
                if not _messages:
                    if hasattr(loaded_prompt, "messages"):
                        _messages = self._process_messages_from_prompt(loaded_prompt.messages, template_format)
                    elif isinstance(loaded_prompt, BaseChatPromptTemplate):
                        _messages = loaded_prompt.messages
                    elif isinstance(loaded_prompt, list):
                        processed_messages = self._process_list_prompt(loaded_prompt, template_format)
                        if processed_messages:
                            _messages = processed_messages
            except Exception as e:
                logger.warning(f"Failed to load prompt {prompt_name}: {e}")

        # Save input variables and partial variables
        _input_variables = list(input_variables) if input_variables else []
        _partial_variables = dict(partial_variables) if partial_variables else {}

        # Initialize parent class with messages only
        super().__init__(messages=_messages)

        # Set attributes
        self.prompt_name = prompt_name
        self.prompt_version = prompt_version
        self.prompt_label = prompt_label
        self.load_at_runtime = load_at_runtime
        self.observability_backend = _observability_backend
        self.cache_ttl_seconds = cache_ttl_seconds
        self.template_format = template_format

        # Explicitly set input variables and partial variables
        if _input_variables:
            self.input_variables = _input_variables

        if _partial_variables:
            self.partial_variables = _partial_variables

        # Set private attributes
        self._observability_platform = _observability_platform
        self._last_load_time = time.time()

    @property
    def observability_platform(self) -> Optional[BaseObservabilityPlatform]:
        """Get the observability platform."""
        return self._observability_platform

    @observability_platform.setter
    def observability_platform(self, platform: BaseObservabilityPlatform) -> None:
        """Set the observability platform."""
        self._observability_platform = platform
        self._loaded_prompt = None
        self._last_load_time = 0

    def _load_prompt_from_platform(
        self,
        platform: BaseObservabilityPlatform,
        prompt_name: str,
        prompt_version: Optional[int] = None,
        prompt_label: Optional[str] = None,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECOND,
        template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
    ) -> PromptReturnType:
        """Load prompt from observability platform."""
        kwargs = {}

        try:
            sig = inspect.signature(platform.pull_prompt).parameters
            if "cache_ttl_seconds" in sig:
                kwargs["cache_ttl_seconds"] = cache_ttl_seconds
            if "template_format" in sig:
                kwargs["template_format"] = template_format
        except (ValueError, TypeError):
            pass

        if prompt_version is not None:
            kwargs["version"] = prompt_version
        if prompt_label is not None:
            kwargs["label"] = prompt_label

        return platform.pull_prompt(prompt_name, **kwargs)

    def _load_prompt_from_observability(self) -> PromptReturnType:
        """Load prompt from observability platform."""
        if not self._observability_platform:
            raise ValueError("No observability platform set")

        if not self.prompt_name:
            raise ValueError("No prompt name provided")

        return self._load_prompt_from_platform(
            self._observability_platform,
            prompt_name=self.prompt_name,
            prompt_version=self.prompt_version,
            prompt_label=self.prompt_label,
            cache_ttl_seconds=self.cache_ttl_seconds,
            template_format=self.template_format,
        )

    def _update_messages_from_loaded_prompt(self) -> None:
        """Update the messages from the loaded prompt."""
        if self._loaded_prompt is None:
            return

        MESSAGE_TYPE_MAP = {
            "system": SystemMessagePromptTemplate,
            "human": HumanMessagePromptTemplate,
            "ai": AIMessagePromptTemplate,
            "assistant": AIMessagePromptTemplate,
        }

        if hasattr(self._loaded_prompt, "messages"):
            processed_messages = []
            for msg in self._loaded_prompt.messages:
                if isinstance(msg, BaseMessage) and msg.type in MESSAGE_TYPE_MAP:
                    content = _convert_template_format(msg.content, self.template_format)
                    template_class = MESSAGE_TYPE_MAP[msg.type]
                    processed_messages.append(
                        template_class.from_template(content, template_format=self.template_format)
                    )
                else:
                    processed_messages.append(msg)

            self.messages = processed_messages
        elif isinstance(self._loaded_prompt, list):
            processed_messages = self._process_list_prompt(self._loaded_prompt, self.template_format)
            if processed_messages:
                self.messages = processed_messages
        elif isinstance(self._loaded_prompt, BaseChatPromptTemplate):
            self.messages = self._loaded_prompt.messages

    def _process_messages_from_prompt(self, messages: Any, template_format: str) -> List[MessageLikeRepresentation]:
        """Process messages from a loaded prompt."""
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
                # Preserve MessagesPlaceholder objects
                processed_messages.append(msg)
            elif isinstance(msg, BaseMessage) and msg.type in MESSAGE_TYPE_MAP:
                content = _convert_template_format(msg.content, template_format)
                template_class = MESSAGE_TYPE_MAP[MessageRole(msg.type)]
                processed_messages.append(template_class.from_template(content, template_format=template_format))
            elif isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                if role in MESSAGE_TYPE_MAP:
                    content = _convert_template_format(content, template_format)
                    template_class = MESSAGE_TYPE_MAP[role]
                    processed_messages.append(template_class.from_template(content, template_format=template_format))
            elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                role, content = msg["role"], msg["content"]
                if role in MESSAGE_TYPE_MAP:
                    content = _convert_template_format(content, template_format)
                    template_class = MESSAGE_TYPE_MAP[role]
                    processed_messages.append(template_class.from_template(content, template_format=template_format))
            else:
                processed_messages.append(msg)

        return processed_messages

    def _process_list_prompt(
        self, prompt_list: List[Any], template_format: str
    ) -> Optional[List[MessageLikeRepresentation]]:
        """Process a list prompt from an observability platform."""
        MESSAGE_TYPE_MAP = {
            MessageRole.SYSTEM: SystemMessagePromptTemplate,
            MessageRole.HUMAN: HumanMessagePromptTemplate,
            MessageRole.USER: HumanMessagePromptTemplate,
            MessageRole.AI: AIMessagePromptTemplate,
            MessageRole.ASSISTANT: AIMessagePromptTemplate,
        }

        processed_messages = []

        # Handle list of tuples (role, content)
        if all(isinstance(item, tuple) and len(item) == 2 for item in prompt_list):
            for role, content in prompt_list:
                if role in MESSAGE_TYPE_MAP:
                    content = _convert_template_format(content, template_format)
                    template_class = MESSAGE_TYPE_MAP[role]
                    processed_messages.append(template_class.from_template(content, template_format=template_format))
            return processed_messages

        # Handle list of dicts with role and content
        if all(isinstance(item, dict) and "role" in item and "content" in item for item in prompt_list):
            for item in prompt_list:
                role, content = item["role"], item["content"]
                if role in MESSAGE_TYPE_MAP:
                    content = _convert_template_format(content, template_format)
                    template_class = MESSAGE_TYPE_MAP[role]
                    processed_messages.append(template_class.from_template(content, template_format=template_format))
                # Handle MessagesPlaceholder
                elif role.lower() in (MessageRole.PLACEHOLDER, MessageRole.MESSAGES_PLACEHOLDER):
                    # Create a MessagesPlaceholder with the content as variable name
                    processed_messages.append(MessagesPlaceholder(variable_name=content))
            return processed_messages

        return processed_messages or None

    def _format_message_with_input(self, msg: Any, input_dict: Dict[str, Any]) -> List[BaseMessage]:
        """Format a message with input."""
        full_input_dict = dict(self.partial_variables or {})
        full_input_dict.update(input_dict)

        try:
            # Special handling for MessagesPlaceholder
            if isinstance(msg, MessagesPlaceholder):
                var_name = msg.variable_name
                if var_name in full_input_dict:
                    messages_value = full_input_dict[var_name]
                    if isinstance(messages_value, list):
                        return messages_value if all(isinstance(m, BaseMessage) for m in messages_value) else [msg]
                    return [messages_value] if isinstance(messages_value, BaseMessage) else [msg]
                return []

            if hasattr(msg, "format") and callable(msg.format):
                return [msg.format(**full_input_dict)]
            elif hasattr(msg, "format_messages") and callable(msg.format_messages):
                return msg.format_messages(**full_input_dict)
        except Exception as e:
            logger.warning(f"Error formatting message: {e}")

        return [msg]

    async def _aformat_message_with_input(self, msg: Any, input_dict: Dict[str, Any]) -> List[BaseMessage]:
        """Asynchronously format a message with input."""
        full_input_dict = dict(self.partial_variables or {})
        full_input_dict.update(input_dict)

        try:
            # Special handling for MessagesPlaceholder
            if isinstance(msg, MessagesPlaceholder):
                var_name = msg.variable_name
                if var_name in full_input_dict:
                    messages_value = full_input_dict[var_name]
                    if isinstance(messages_value, list):
                        return messages_value if all(isinstance(m, BaseMessage) for m in messages_value) else [msg]
                    return [messages_value] if isinstance(messages_value, BaseMessage) else [msg]
                return []

            if hasattr(msg, "aformat") and callable(msg.aformat):
                return [await msg.aformat(**full_input_dict)]
            elif hasattr(msg, "aformat_messages") and callable(msg.aformat_messages):
                return await msg.aformat_messages(**full_input_dict)
            elif hasattr(msg, "format") and callable(msg.format):
                return [msg.format(**full_input_dict)]
            elif hasattr(msg, "format_messages") and callable(msg.format_messages):
                return msg.format_messages(**full_input_dict)
        except Exception:
            pass

        return [msg]

    def _ensure_messages_loaded(self) -> None:
        """Ensure messages are loaded from observability platform if needed."""
        if not self.load_at_runtime or not self.prompt_name or not self._observability_platform:
            return

        current_time = time.time()
        if self._loaded_prompt is None or current_time - self._last_load_time > self.cache_ttl_seconds:
            try:
                self._loaded_prompt = self._load_prompt_from_observability()
                self._last_load_time = current_time
                self._update_messages_from_loaded_prompt()
            except Exception as e:
                logger.error(f"Failed to load prompt: {e}")
                if not self.messages:
                    raise ValueError(f"Failed to load prompt and no fallback available: {e}")

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> PromptValue:
        """Invoke the prompt template."""
        self._ensure_messages_loaded()

        if isinstance(input, dict):
            formatted_messages = []
            for msg in self.messages:
                formatted_messages.extend(self._format_message_with_input(msg, input))
            return ChatPromptValue(messages=formatted_messages)

        return super().invoke(input=input, config=config, **kwargs)

    async def ainvoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> PromptValue:
        """Asynchronously invoke the prompt template."""
        self._ensure_messages_loaded()

        if isinstance(input, dict):
            formatted_messages = []
            for msg in self.messages:
                formatted_messages.extend(await self._aformat_message_with_input(msg, input))
            return ChatPromptValue(messages=formatted_messages)

        return await super().ainvoke(input=input, config=config, **kwargs)

    @classmethod
    def from_observability_platform(
        cls,
        prompt_name: str,
        observability_platform: BaseObservabilityPlatform,
        *,
        prompt_version: Optional[int] = None,
        prompt_label: Optional[str] = None,
        load_at_runtime: bool = True,
        **kwargs: Any,
    ) -> "ObservabilityChatPromptTemplate":
        """Create a chat prompt template from an observability platform."""
        return cls(
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            prompt_label=prompt_label,
            load_at_runtime=load_at_runtime,
            observability_platform=observability_platform,
            **kwargs,
        )

    @classmethod
    def from_observability_backend(
        cls,
        prompt_name: str,
        observability_backend: Union[ObservabilityBackend, str],
        *,
        prompt_version: Optional[int] = None,
        prompt_label: Optional[str] = None,
        load_at_runtime: bool = True,
        **kwargs: Any,
    ) -> "ObservabilityChatPromptTemplate":
        """Create a chat prompt template from an observability backend."""
        backend = (
            ObservabilityBackend(observability_backend)
            if isinstance(observability_backend, str)
            else observability_backend
        )
        platform = ObservabilityFactory.create(backend)

        return cls(
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            prompt_label=prompt_label,
            load_at_runtime=load_at_runtime,
            observability_platform=platform,
            observability_backend=backend,
            **kwargs,
        )

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """Combine two prompt templates."""
        if isinstance(other, ChatPromptTemplate):
            # Create a copy of messages from both templates
            combined_messages = list(self.messages)

            # Process messages from the other template
            other_messages = []
            for msg in other.messages:
                # Special handling for MessagesPlaceholder
                if isinstance(msg, MessagesPlaceholder):
                    other_messages.append(msg)
                    continue

                if isinstance(msg, BaseMessagePromptTemplate):
                    other_messages.append(msg)
                elif isinstance(msg, BaseMessage):
                    content = msg.content
                    if isinstance(content, str):
                        template_vars = get_template_variables(content, self.template_format)
                        if template_vars:
                            template_class = {
                                MessageRole.SYSTEM: SystemMessagePromptTemplate,
                                MessageRole.HUMAN: HumanMessagePromptTemplate,
                                MessageRole.AI: AIMessagePromptTemplate,
                                MessageRole.ASSISTANT: AIMessagePromptTemplate,
                            }.get(MessageRole(msg.type))

                            if template_class:
                                other_messages.append(
                                    template_class.from_template(content, template_format=self.template_format)
                                )
                                continue

                    other_messages.append(msg)
                else:
                    other_messages.append(msg)

            combined_messages.extend(other_messages)

            # Collect all input variables
            all_vars = set(self.input_variables or [])
            other_vars = set(other.input_variables or [])
            all_vars.update(other_vars)

            # Get variables from MessagesPlaceholder
            for msg in combined_messages:
                if isinstance(msg, MessagesPlaceholder):
                    all_vars.add(msg.variable_name)
                elif hasattr(msg, "input_variables"):
                    all_vars.update(msg.input_variables)

            # Create new partial variables dict
            combined_partial_vars = dict(self.partial_variables or {})
            if hasattr(other, "partial_variables") and other.partial_variables:
                for k, v in other.partial_variables.items():
                    if k not in combined_partial_vars:
                        combined_partial_vars[k] = v

            # Create the combined template
            return ChatPromptTemplate(
                messages=combined_messages,
                input_variables=list(all_vars),
                partial_variables=combined_partial_vars,
            )

        elif isinstance(other, (BaseMessagePromptTemplate, BaseMessage)):
            return self + ChatPromptTemplate.from_messages([other])
        elif isinstance(other, (list, tuple)):
            return self + ChatPromptTemplate.from_messages(other)
        elif isinstance(other, str):
            return self + ChatPromptTemplate.from_template(other)
        else:
            raise NotImplementedError(f"Unsupported operand type for +: {type(other)}")


__all__ = ["ObservabilityChatPromptTemplate"]
