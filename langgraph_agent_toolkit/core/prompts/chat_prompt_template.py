import asyncio
import inspect
import re
import threading
import time
from concurrent.futures import Future
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from jinja2.sandbox import SandboxedEnvironment
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts.chat import (
    AIMessagePromptTemplate,
    BaseChatPromptTemplate,
    BaseMessage,
    BaseMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessageLikeRepresentation,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.string import get_template_variables
from pydantic import Field, PrivateAttr

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform, PromptReturnType
from langgraph_agent_toolkit.core.observability.factory import ObservabilityFactory
from langgraph_agent_toolkit.core.observability.types import MessageRole, ObservabilityBackend
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.logging import logger


# Keep dictionary attribute access and block access to Python internals.
_JINJA2_ENV = SandboxedEnvironment(autoescape=False)

# Map message roles to prompt template classes.
_MESSAGE_TYPE_MAP = {
    MessageRole.SYSTEM: SystemMessagePromptTemplate,
    MessageRole.HUMAN: HumanMessagePromptTemplate,
    MessageRole.USER: HumanMessagePromptTemplate,
    MessageRole.AI: AIMessagePromptTemplate,
    MessageRole.ASSISTANT: AIMessagePromptTemplate,
}

# Map `BaseMessage.type` strings to prompt template classes.
_STRING_TYPE_MAP = {
    "system": SystemMessagePromptTemplate,
    "human": HumanMessagePromptTemplate,
    "ai": AIMessagePromptTemplate,
    "assistant": AIMessagePromptTemplate,
}


def _convert_template_format(content: str, target_format: str) -> str:
    """Convert a template string between formats."""
    if not content or not isinstance(content, str):
        return content

    if target_format == "jinja2" and "{" in content and "{{" not in content:
        # Convert f-string format to Jinja2 format.
        return re.sub(r"{(\w+)}", r"{{ \1 }}", content)
    elif target_format == "f-string" and "{{" in content:
        # Convert Jinja2 format to f-string format.
        return re.sub(r"{{\s*(\w+)\s*}}", r"{\1}", content)
    return content


class ObservabilityChatPromptTemplate(ChatPromptTemplate):
    """Chat prompt template that loads prompts from observability platforms."""

    prompt_name: Optional[str] = Field(default=None, description="Name of the prompt to load")
    prompt_version: Optional[int] = Field(default=None, description="Version of the prompt")
    prompt_label: Optional[str] = Field(default=None, description="Label of the prompt")
    load_at_runtime: bool = Field(default=False, description="Whether to load prompt at runtime")
    observability_backend: Optional[ObservabilityBackend] = Field(
        default=None,
        description="Observability backend to use",
    )
    cache_ttl_seconds: int = Field(
        default=settings.LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS,
        description="Cache TTL for prompts",
    )
    template_format: str = Field(default="f-string", description="Format of the template")

    _observability_platform: Optional[BaseObservabilityPlatform] = None
    _loaded_prompt: Any = None
    _last_load_time: float = 0
    _jinja2_template_cache: Dict[str, Any] = {}
    _load_state_lock: Any = PrivateAttr(default_factory=threading.Lock)
    _load_future: Any = PrivateAttr(default=None)
    _declared_input_variables: List[str] = PrivateAttr(default_factory=list)

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
        cache_ttl_seconds: int = settings.LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS,
        template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
        input_variables: Optional[List[str]] = None,
        partial_variables: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize ObservabilityChatPromptTemplate."""
        # Process the observability platform and backend.
        _observability_platform = observability_platform
        _observability_backend = observability_backend

        if not observability_platform and observability_backend:
            _observability_backend = (
                ObservabilityBackend(observability_backend)
                if isinstance(observability_backend, str)
                else observability_backend
            )
            _observability_platform = ObservabilityFactory.create(_observability_backend)

        # Load messages when required.
        _messages = messages or []
        loaded_prompt = None
        if not load_at_runtime and prompt_name and _observability_platform:
            try:
                loaded_prompt = self._load_prompt_from_platform(
                    _observability_platform,
                    prompt_name=prompt_name,
                    prompt_version=prompt_version,
                    prompt_label=prompt_label,
                    cache_ttl_seconds=cache_ttl_seconds,
                    template_format=template_format,
                )

                # Process the returned prompt by type.
                if not _messages:
                    if hasattr(loaded_prompt, "messages"):
                        _messages = self._process_messages_from_prompt(loaded_prompt.messages, template_format)
                    elif isinstance(loaded_prompt, BaseChatPromptTemplate):
                        _messages = loaded_prompt.messages
                    elif isinstance(loaded_prompt, list):
                        processed_messages = self._process_list_prompt(loaded_prompt, template_format)
                        if processed_messages:
                            _messages = processed_messages
                    elif isinstance(loaded_prompt, str):
                        _messages = [("human", loaded_prompt)]
                if not _messages:
                    raise ValueError("The prompt backend returned no messages")
            except Exception as e:
                logger.warning(f"Failed to load prompt {prompt_name}: {e}")
                if not _messages:
                    raise ValueError(f"Failed to load prompt and no fallback available: {e}") from e

        # Save input and partial variables.
        _input_variables = list(input_variables) if input_variables else []
        _partial_variables = dict(partial_variables) if partial_variables else {}

        # Let the parent derive required variables and process partial values.
        super().__init__(
            messages=_messages,
            template_format=template_format,
            partial_variables=_partial_variables,
            **kwargs,
        )

        # Set attributes.
        self.prompt_name = prompt_name
        self.prompt_version = prompt_version
        self.prompt_label = prompt_label
        self.load_at_runtime = load_at_runtime
        self.observability_backend = _observability_backend
        self.cache_ttl_seconds = cache_ttl_seconds
        self.template_format = template_format

        self._declared_input_variables = _input_variables
        self.input_variables = sorted(
            (set(self.input_variables) | set(_input_variables))
            - set(self.partial_variables)
            - set(self.optional_variables)
        )

        # Set private attributes.
        self._observability_platform = _observability_platform
        self._loaded_prompt = loaded_prompt
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
        cache_ttl_seconds: int = settings.LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS,
        template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
    ) -> PromptReturnType:
        """Load prompt from observability platform."""
        kwargs = {}

        try:
            sig = inspect.signature(platform.pull_prompt).parameters
            accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in sig.values())
            if "cache_ttl_seconds" in sig or accepts_kwargs:
                kwargs["cache_ttl_seconds"] = cache_ttl_seconds
            if "template_format" in sig or accepts_kwargs:
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

    def _accept_loaded_prompt(self, loaded_prompt: Any) -> None:
        """Validate a refreshed prompt before replacing the active messages."""
        if hasattr(loaded_prompt, "messages"):
            messages = self._process_messages_from_prompt(loaded_prompt.messages, self.template_format)
        elif isinstance(loaded_prompt, list):
            messages = self._process_list_prompt(loaded_prompt, self.template_format)
        elif isinstance(loaded_prompt, str):
            messages = [("human", loaded_prompt)]
        else:
            raise ValueError("The prompt backend returned an unsupported prompt")
        if not messages:
            raise ValueError("The prompt backend returned no messages")
        template = ChatPromptTemplate(
            messages=messages,
            template_format=self.template_format,
            partial_variables=self.partial_variables,
        )
        self.messages = template.messages
        self.partial_variables = template.partial_variables
        self.optional_variables = template.optional_variables
        self.input_variables = sorted(
            (set(template.input_variables) | set(self._declared_input_variables))
            - set(template.partial_variables)
            - set(template.optional_variables)
        )
        self._jinja2_template_cache.clear()
        self._loaded_prompt = loaded_prompt
        self._last_load_time = time.time()

    def _process_messages_from_prompt(self, messages: Any, template_format: str) -> List[MessageLikeRepresentation]:
        """Process messages from a loaded prompt."""
        processed_messages = []
        for msg in messages:
            if isinstance(msg, MessagesPlaceholder):
                # Keep `MessagesPlaceholder` objects.
                processed_messages.append(msg)
            elif isinstance(msg, BaseMessage) and msg.type in _STRING_TYPE_MAP:
                content = _convert_template_format(msg.content, template_format)
                template_class = _STRING_TYPE_MAP[msg.type]
                processed_messages.append(template_class.from_template(content, template_format=template_format))
            elif isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                if role in _MESSAGE_TYPE_MAP:
                    content = _convert_template_format(content, template_format)
                    template_class = _MESSAGE_TYPE_MAP[role]
                    processed_messages.append(template_class.from_template(content, template_format=template_format))
            elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                role, content = msg["role"], msg["content"]
                if role in _MESSAGE_TYPE_MAP:
                    content = _convert_template_format(content, template_format)
                    template_class = _MESSAGE_TYPE_MAP[role]
                    processed_messages.append(template_class.from_template(content, template_format=template_format))
            else:
                processed_messages.append(msg)

        return processed_messages

    def _process_list_prompt(
        self, prompt_list: List[Any], template_format: str
    ) -> Optional[List[MessageLikeRepresentation]]:
        """Process a list prompt from an observability platform."""
        processed_messages = []

        # Process a list of `(role, content)` tuples.
        if all(isinstance(item, tuple) and len(item) == 2 for item in prompt_list):
            for role, content in prompt_list:
                if role in _MESSAGE_TYPE_MAP:
                    content = _convert_template_format(content, template_format)
                    template_class = _MESSAGE_TYPE_MAP[role]
                    processed_messages.append(template_class.from_template(content, template_format=template_format))
            return processed_messages

        # Process a list of dictionaries with `role` and `content`.
        if all(isinstance(item, dict) and "role" in item and "content" in item for item in prompt_list):
            for item in prompt_list:
                role, content = item["role"], item["content"]
                if role in _MESSAGE_TYPE_MAP:
                    content = _convert_template_format(content, template_format)
                    template_class = _MESSAGE_TYPE_MAP[role]
                    processed_messages.append(template_class.from_template(content, template_format=template_format))
                # Process a `MessagesPlaceholder`.
                elif role.lower() in (MessageRole.PLACEHOLDER, MessageRole.MESSAGES_PLACEHOLDER):
                    # Use `content` as the `MessagesPlaceholder` variable name.
                    processed_messages.append(MessagesPlaceholder(variable_name=content))
            return processed_messages

        return processed_messages or None

    def _should_reload_prompt(self) -> bool:
        """Return whether cache TTL requires a prompt reload."""
        if not self.load_at_runtime or not self.prompt_name or not self._observability_platform:
            return False
        current_time = time.time()
        return self._loaded_prompt is None or current_time - self._last_load_time > self.cache_ttl_seconds

    def _begin_load(self) -> tuple[Optional[Future], bool]:
        """Share one prompt load across synchronous and asynchronous callers."""
        with self._load_state_lock:
            if self._load_future is not None:
                return self._load_future, False
            if not self._should_reload_prompt():
                return None, False
            self._load_future = Future()
            return self._load_future, True

    def _finish_load(self, future: Future, error: Optional[BaseException] = None) -> None:
        with self._load_state_lock:
            if error is None:
                future.set_result(None)
            else:
                future.set_exception(error)
            self._load_future = None

    def _load_error(self, error: BaseException) -> Optional[BaseException]:
        if not isinstance(error, Exception):
            return error
        logger.error(f"Failed to load prompt: {error}")
        if not self.messages:
            return ValueError(f"Failed to load prompt and no fallback available: {error}")
        return None

    def _ensure_messages_loaded(self) -> None:
        """Load messages from the observability platform when required."""
        future, owner = self._begin_load()
        if future is None:
            return
        if not owner:
            future.result()
            return
        try:
            self._accept_loaded_prompt(self._load_prompt_from_observability())
        except BaseException as error:
            failure = self._load_error(error)
            self._finish_load(future, failure)
            if failure is not None:
                raise failure from error
        else:
            self._finish_load(future)

    async def _aensure_messages_loaded(self) -> None:
        """Load messages without blocking the event loop."""
        future, owner = self._begin_load()
        if future is None:
            return
        if not owner:
            await asyncio.shield(asyncio.wrap_future(future))
            return
        try:
            loaded_prompt = await self._observability_platform.apull_prompt(
                name=self.prompt_name,
                cache_ttl_seconds=self.cache_ttl_seconds,
                template_format=self.template_format,
                version=self.prompt_version,
                label=self.prompt_label,
            )
            self._accept_loaded_prompt(loaded_prompt)
        except BaseException as error:
            failure = self._load_error(error)
            self._finish_load(future, failure)
            if failure is not None:
                raise failure from error
        else:
            self._finish_load(future)

    def _get_compiled_template(self, template_content: str):
        """Get a compiled Jinja2 template from cache or create one.

        The cache avoids parsing templates for each render.
        """
        template = self._jinja2_template_cache.get(template_content)
        if template is None:
            template = _JINJA2_ENV.from_string(template_content)
            self._jinja2_template_cache[template_content] = template
        return template

    def _render_jinja2_template(self, template_content: str, variables: Dict[str, Any]) -> str:
        """Render a cached template inside the Jinja2 sandbox."""
        template = self._get_compiled_template(template_content)
        return template.render(**variables)

    def _format_messages_with_jinja2(self, input_values: Dict[str, Any]) -> List[BaseMessage]:
        """Format Jinja2 text and keep standard message behavior."""
        formatted_messages = []
        message_types = {
            SystemMessagePromptTemplate: SystemMessage,
            HumanMessagePromptTemplate: HumanMessage,
            AIMessagePromptTemplate: AIMessage,
        }
        for msg in self.messages:
            if isinstance(msg, BaseMessage):
                formatted_messages.append(msg)
                continue
            prompt = getattr(msg, "prompt", None)
            message_type = next((value for key, value in message_types.items() if isinstance(msg, key)), None)
            if message_type and getattr(prompt, "template_format", None) == "jinja2":
                variables = prompt._merge_partial_and_user_variables(**input_values)
                content = self._render_jinja2_template(prompt.template, variables)
                formatted_messages.append(message_type(content=content, **msg.additional_kwargs))
            else:
                formatted_messages.extend(msg.format_messages(**input_values))
        return formatted_messages

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format messages with standard partial and placeholder handling."""
        if self.template_format != "jinja2":
            return super().format_messages(**kwargs)
        input_values = self._merge_partial_and_user_variables(**kwargs)
        return self._format_messages_with_jinja2(input_values)

    async def aformat_messages(self, **kwargs: Any) -> List[BaseMessage]:
        if self.template_format != "jinja2":
            return await super().aformat_messages(**kwargs)
        return self.format_messages(**kwargs)

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> PromptValue:
        """Invoke the prompt with standard input validation and callbacks."""
        self._ensure_messages_loaded()
        return super().invoke(input=input, config=config, **kwargs)

    async def ainvoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> PromptValue:
        """Asynchronously invoke the prompt with standard callbacks."""
        await self._aensure_messages_loaded()
        return await super().ainvoke(input=input, config=config, **kwargs)

    def partial(self, **kwargs: Any) -> "ObservabilityChatPromptTemplate":
        """Keep the remote backend when binding partial variables."""
        template = super().partial(**kwargs)
        template._observability_platform = self._observability_platform
        template._loaded_prompt = self._loaded_prompt
        template._last_load_time = self._last_load_time
        return template

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
            # Copy messages from both templates.
            combined_messages = list(self.messages)

            # Process messages from the other template.
            other_messages = []
            for msg in other.messages:
                # Process `MessagesPlaceholder`.
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
                            template_class = _STRING_TYPE_MAP.get(msg.type)

                            if template_class:
                                other_messages.append(
                                    template_class.from_template(content, template_format=self.template_format)
                                )
                                continue

                    other_messages.append(msg)
                else:
                    other_messages.append(msg)

            combined_messages.extend(other_messages)

            # Collect input variables.
            all_vars = set(self.input_variables or [])
            other_vars = set(other.input_variables or [])
            all_vars.update(other_vars)

            # Get variables from `MessagesPlaceholder`.
            for msg in combined_messages:
                if isinstance(msg, MessagesPlaceholder):
                    all_vars.add(msg.variable_name)
                elif hasattr(msg, "input_variables"):
                    all_vars.update(msg.input_variables)

            # Create the partial variable dictionary.
            combined_partial_vars = dict(self.partial_variables or {})
            if hasattr(other, "partial_variables") and other.partial_variables:
                for k, v in other.partial_variables.items():
                    if k not in combined_partial_vars:
                        combined_partial_vars[k] = v

            # Create the combined template.
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
