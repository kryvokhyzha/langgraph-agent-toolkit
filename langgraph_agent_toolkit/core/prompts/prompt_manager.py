import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from langgraph_agent_toolkit.core.observability.factory import ObservabilityFactory
from langgraph_agent_toolkit.core.observability.types import ChatMessageDict, MessageRole
from langgraph_agent_toolkit.core.prompts.chat_prompt_template import ObservabilityChatPromptTemplate
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.utils import read_file


class PromptManager:
    """Manages prompt templates for assistant."""

    def __init__(
        self,
        observability_backend: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
        force_create_new_version: bool = False,
        template_format: str = "jinja2",
        load_at_runtime: bool = True,
    ):
        """Initialize the PromptManager with configurable parameters.

        Args:
            observability_backend: Backend type for observability (defaults to settings.OBSERVABILITY_BACKEND)
            prompts_dir: Directory containing prompt templates (defaults to ./prompts)
            force_create_new_version: Whether to force creation of new prompt versions
            template_format: Format for prompt templates (default: "jinja2")
            load_at_runtime: Whether to load prompts at runtime (default: True)

        """
        self._prompt_cache: Dict[str, ObservabilityChatPromptTemplate] = {}
        self._observability = None
        self._observability_backend = observability_backend or settings.OBSERVABILITY_BACKEND
        self._prompts_dir = prompts_dir or (Path(__file__).parent / "prompts")
        self._force_create_new_version = force_create_new_version
        self._template_format = template_format
        self._load_at_runtime = load_at_runtime

    @property
    def observability(self):
        """Lazy-loaded observability platform."""
        if self._observability is None:
            self._observability = ObservabilityFactory.create(self._observability_backend)
        return self._observability

    def _build_prompt_template(
        self,
        template_content: str,
        has_messages_placeholder: bool = False,
    ) -> List[ChatMessageDict]:
        """Build prompt template messages from template content."""
        prompt_template = [
            ChatMessageDict(
                role=MessageRole.SYSTEM,
                content=template_content,
            )
        ]

        if has_messages_placeholder:
            prompt_template.append(
                ChatMessageDict(
                    role=MessageRole.PLACEHOLDER,
                    content="messages",
                )
            )

        return prompt_template

    def _cache_prompt(
        self,
        prompt_name: str,
        input_variables: List[str],
        partial_variables: Optional[Dict[str, str]] = None,
    ) -> ObservabilityChatPromptTemplate:
        """Create and cache the prompt from observability platform."""
        prompt = ObservabilityChatPromptTemplate.from_observability_platform(
            prompt_name=prompt_name,
            observability_platform=self.observability,
            load_at_runtime=self._load_at_runtime,
            template_format=self._template_format,
            input_variables=input_variables,
            partial_variables=partial_variables or {},
        )
        self._prompt_cache[prompt_name] = prompt
        return prompt

    def _create_prompt_template(
        self,
        prompt_name: str,
        template_path: Path,
        input_variables: List[str],
        has_messages_placeholder: bool = False,
        partial_variables: Optional[Dict[str, str]] = None,
    ) -> ObservabilityChatPromptTemplate:
        """Create and cache a prompt template from a file."""
        template_content = read_file(template_path)
        prompt_template = self._build_prompt_template(template_content, has_messages_placeholder)

        self.observability.push_prompt(
            name=prompt_name,
            prompt_template=prompt_template,
            force_create_new_version=self._force_create_new_version,
        )

        return self._cache_prompt(prompt_name, input_variables, partial_variables)

    async def _acreate_prompt_template(
        self,
        prompt_name: str,
        template_path: Path,
        input_variables: List[str],
        has_messages_placeholder: bool = False,
        partial_variables: Optional[Dict[str, str]] = None,
    ) -> ObservabilityChatPromptTemplate:
        """Create and cache a prompt template from a file (async version)."""
        template_content = await asyncio.to_thread(read_file, template_path)
        prompt_template = self._build_prompt_template(template_content, has_messages_placeholder)

        await asyncio.to_thread(
            self.observability.push_prompt,
            name=prompt_name,
            prompt_template=prompt_template,
            force_create_new_version=self._force_create_new_version,
        )

        return self._cache_prompt(prompt_name, input_variables, partial_variables)

    def _get_or_create_prompt(
        self,
        prompt_name: str,
        template_filename: str,
        input_variables: List[str],
        has_messages_placeholder: bool = False,
        partial_variables: Optional[Dict[str, str]] = None,
    ) -> ObservabilityChatPromptTemplate:
        """Get cached prompt or create it if it doesn't exist."""
        if prompt_name not in self._prompt_cache:
            template_path = self._prompts_dir / template_filename
            self._create_prompt_template(
                prompt_name=prompt_name,
                template_path=template_path,
                input_variables=input_variables,
                has_messages_placeholder=has_messages_placeholder,
                partial_variables=partial_variables,
            )
        return self._prompt_cache[prompt_name]

    async def _aget_or_create_prompt(
        self,
        prompt_name: str,
        template_filename: str,
        input_variables: List[str],
        has_messages_placeholder: bool = False,
        partial_variables: Optional[Dict[str, str]] = None,
    ) -> ObservabilityChatPromptTemplate:
        """Get cached prompt or create it if it doesn't exist (async version)."""
        if prompt_name not in self._prompt_cache:
            template_path = self._prompts_dir / template_filename
            await self._create_prompt_template_async(
                prompt_name=prompt_name,
                template_path=template_path,
                input_variables=input_variables,
                has_messages_placeholder=has_messages_placeholder,
                partial_variables=partial_variables,
            )
        return self._prompt_cache[prompt_name]

    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._prompt_cache.clear()

    def get_cached_prompt_names(self) -> List[str]:
        """Get list of cached prompt names."""
        return list(self._prompt_cache.keys())

    def set_prompts_directory(self, prompts_dir: Path) -> None:
        """Update the prompts directory and clear cache."""
        self._prompts_dir = prompts_dir
        self.clear_cache()
