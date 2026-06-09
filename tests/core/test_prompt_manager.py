import pytest

from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.core.observability.types import MessageRole, ObservabilityBackend
from langgraph_agent_toolkit.core.prompts.chat_prompt_template import ObservabilityChatPromptTemplate
from langgraph_agent_toolkit.core.prompts.prompt_manager import PromptManager


class CountingEmptyObservability(EmptyObservability):
    """EmptyObservability that counts push_prompt calls (for cache-hit assertions, no mocking needed)."""

    def __init__(self, remote_first: bool = False):
        super().__init__(remote_first)
        self.push_count = 0

    def push_prompt(self, *args, **kwargs):
        self.push_count += 1
        return super().push_prompt(*args, **kwargs)


@pytest.fixture
def pm(tmp_path):
    """Build a PromptManager backed by a counting in-memory observability platform and a real template file."""
    (tmp_path / "greeting.j2").write_text("You are a {{ role }} assistant.")
    manager = PromptManager(
        observability_backend=ObservabilityBackend.EMPTY,
        prompts_dir=tmp_path,
        template_format="jinja2",
    )
    # Inject a counting backend so push behavior is observable without a remote platform.
    manager._observability = CountingEmptyObservability()
    return manager


async def test_async_create_and_cache_regression(pm):
    """Regression for the async cache-miss path (commit that fixed the misnamed method call).

    `_aget_or_create_prompt` -> `_acreate_prompt_template` must create, push, cache, and return a
    template. Before the fix it called a nonexistent method and raised AttributeError on the first
    (cache-miss) call, so this test simply completing is the regression guard.
    """
    prompt = await pm._aget_or_create_prompt("greet", "greeting.j2", input_variables=["role"])

    assert isinstance(prompt, ObservabilityChatPromptTemplate)
    assert "greet" in pm.get_cached_prompt_names()
    assert pm._observability.push_count == 1
    assert "helpful assistant" in prompt.invoke({"role": "helpful"}).to_messages()[0].content


def test_sync_create_and_cache(pm):
    """`_get_or_create_prompt` reads the file, pushes it, and caches a rendered template."""
    prompt = pm._get_or_create_prompt("greet", "greeting.j2", input_variables=["role"])

    assert isinstance(prompt, ObservabilityChatPromptTemplate)
    assert pm.get_cached_prompt_names() == ["greet"]
    assert "helpful assistant" in prompt.invoke({"role": "helpful"}).to_messages()[0].content


def test_cache_hit_does_not_push_again(pm):
    """A second request for a cached prompt returns the same object and does not re-push."""
    first = pm._get_or_create_prompt("greet", "greeting.j2", input_variables=["role"])
    second = pm._get_or_create_prompt("greet", "greeting.j2", input_variables=["role"])

    assert first is second
    assert pm._observability.push_count == 1


def test_clear_cache_forces_recreate(pm):
    """clear_cache empties the cache so the next request re-creates (and re-pushes)."""
    pm._get_or_create_prompt("greet", "greeting.j2", input_variables=["role"])
    assert pm.get_cached_prompt_names() == ["greet"]

    pm.clear_cache()
    assert pm.get_cached_prompt_names() == []

    pm._get_or_create_prompt("greet", "greeting.j2", input_variables=["role"])
    assert pm._observability.push_count == 2


def test_set_prompts_directory_clears_cache(pm, tmp_path):
    """set_prompts_directory swaps the directory AND clears the cache."""
    pm._get_or_create_prompt("greet", "greeting.j2", input_variables=["role"])
    assert pm.get_cached_prompt_names() == ["greet"]

    new_dir = tmp_path / "other"
    new_dir.mkdir()
    pm.set_prompts_directory(new_dir)

    assert pm.get_cached_prompt_names() == []
    assert pm._prompts_dir == new_dir


def test_build_prompt_template_with_and_without_placeholder(pm):
    """_build_prompt_template returns [system] or [system, placeholder('messages')]."""
    without = pm._build_prompt_template("hi", has_messages_placeholder=False)
    assert len(without) == 1
    assert without[0]["role"] == MessageRole.SYSTEM
    assert without[0]["content"] == "hi"

    with_placeholder = pm._build_prompt_template("hi", has_messages_placeholder=True)
    assert len(with_placeholder) == 2
    assert with_placeholder[1]["role"] == MessageRole.PLACEHOLDER
    assert with_placeholder[1]["content"] == "messages"


def test_observability_property_is_lazy_and_memoized():
    """The observability platform is built lazily on first access, then memoized."""
    manager = PromptManager(observability_backend=ObservabilityBackend.EMPTY)
    assert manager._observability is None

    first = manager.observability
    assert first is not None
    assert manager.observability is first
