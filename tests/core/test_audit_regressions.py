import asyncio
import base64
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from jinja2.exceptions import SecurityError
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import SecretStr

from langgraph_agent_toolkit.core._base_settings import Settings
from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory, _ConfigurableModelCustom
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.core.observability.langfuse import LangfuseObservability
from langgraph_agent_toolkit.core.prompts.chat_prompt_template import ObservabilityChatPromptTemplate
from langgraph_agent_toolkit.core.prompts.prompt_manager import PromptManager
from langgraph_agent_toolkit.helper.constants import DEFAULT_MODEL_PARAMETER_VALUES


def test_factory_defaults_do_not_retain_credentials():
    defaults = dict(DEFAULT_MODEL_PARAMETER_VALUES)
    try:
        with patch.object(CompletionModelFactory, "init_chat_model", side_effect=lambda **kwargs: kwargs):
            CompletionModelFactory.create("openai", "first", api_key="dummy-a", base_url="https://example.invalid/a")
            second = CompletionModelFactory.create("openai", "second")
        assert "api_key" not in second
        assert "base_url" not in second
        assert DEFAULT_MODEL_PARAMETER_VALUES == defaults
    finally:
        DEFAULT_MODEL_PARAMETER_VALUES.clear()
        DEFAULT_MODEL_PARAMETER_VALUES.update(defaults)


def test_openai_wrapper_accepts_dictionary_responses():
    from langgraph_agent_toolkit.core.models.chat_openai import ChatOpenAIPatched

    model = ChatOpenAIPatched(model="dummy", api_key="dummy")
    result = model._create_chat_result(
        {"choices": [{"message": {"role": "assistant_custom", "content": "answer"}, "finish_reason": "stop"}]}
    )
    assert result.generations[0].message.content == "answer"


@pytest.mark.asyncio
async def test_async_backend_accepts_keyword_only_template_format():
    class Backend(EmptyObservability):
        def pull_prompt(self, name, *, template_format="f-string", **kwargs):
            return ChatPromptTemplate.from_template("Hello {{ name }}", template_format=template_format)

    prompt = await Backend().apull_prompt("test", template_format="jinja2")
    assert prompt.invoke("Jane").to_messages()[0].content == "Hello Jane"


def test_fake_import_without_openai_extra():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['openai'] = None; sys.modules['langchain_openai'] = None; "
            "from langgraph_agent_toolkit.core.models import CompletionModelFactory, FakeToolModel; "
            "assert isinstance(CompletionModelFactory.create('fake'), FakeToolModel)",
        ],
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "PYTHON_DOTENV_DISABLED": "1"},
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("operation", [lambda m: m.bind_tools([]), lambda m: m.with_config(tags=["test"])])
def test_model_operations_preserve_patched_construction(operation):
    model = CompletionModelFactory.create("openai", "dummy")
    configured = operation(model)
    assert isinstance(configured, _ConfigurableModelCustom)
    with patch.object(CompletionModelFactory, "_init_chat_model_helper", return_value=MagicMock()) as create:
        configured._model()
    create.assert_called_once()


@pytest.mark.parametrize("field", ["AUTH_SECRET", "POSTGRES_PASSWORD", "OPENAI_API_KEY"])
def test_prefixed_secrets_remain_typed(field, monkeypatch):
    monkeypatch.setenv(f"LANGGRAPH_{field}", "replacement")
    settings = Settings(_env_file=None, **{field: SecretStr("original")})
    settings._apply_langgraph_env_overrides()
    value = getattr(settings, field)
    assert isinstance(value, SecretStr)
    assert value.get_secret_value() == "replacement"


def test_prefixed_enum_and_optional_types(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_ENV_MODE", "development")
    monkeypatch.setenv("LANGGRAPH_POSTGRES_PORT", "5432")
    monkeypatch.setenv("LANGGRAPH_DEFAULT_MAX_TOKENS_HISTORY_LENGTH", "4096")
    settings = Settings(_env_file=None)
    settings.setup()
    assert settings.ENV_MODE.value == "development"
    assert settings.POSTGRES_PORT == 5432
    assert settings.DEFAULT_MAX_TOKENS_HISTORY_LENGTH == 4096


def test_prefixed_invalid_field_constraint_fails(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_CLEAR_INTERMEDIATE_TOOL_CALLS_KEEP_LAST_N", "0")
    settings = Settings(_env_file=None)
    with pytest.raises(ValueError):
        settings._apply_langgraph_env_overrides()


def test_settings_mapping_overrides_are_atomic_and_typed():
    settings = Settings(_env_file=None, PORT=8080)
    with pytest.raises(ValueError):
        settings.apply_overrides({"PORT": 9000, "CLEAR_INTERMEDIATE_TOOL_CALLS_KEEP_LAST_N": 0})
    assert settings.PORT == 8080
    settings.apply_overrides({"AUTH_SECRET": "dummy", "AGENT_PATHS": '["example:agent"]', "PORT": "9000"})
    assert settings.AUTH_SECRET == SecretStr("dummy")
    assert settings.AGENT_PATHS == ["example:agent"]
    assert settings.PORT == 9000
    with pytest.raises(ValueError, match="Unknown"):
        settings.apply_overrides({"UNKNOWN_SETTING": "x"})


def test_prefixed_optional_value_can_be_cleared(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_AUTH_SECRET", "null")
    settings = Settings(_env_file=None, AUTH_SECRET="dummy")
    settings._apply_langgraph_env_overrides()
    assert settings.AUTH_SECRET is None


@pytest.mark.parametrize("kind", ["MODEL", "DB"])
@pytest.mark.parametrize("source", ["constructor", "dotenv", "base64", "path"])
def test_setup_preserves_configuration_sources(kind, source, tmp_path, monkeypatch):
    for suffix in ["", "_BASE64", "_PATH"]:
        monkeypatch.delenv(f"{kind}_CONFIGS{suffix}", raising=False)
    value = {"example": {"name": "dummy"}}
    raw = '{"example":{"name":"dummy"}}'
    kwargs = {"_env_file": None}
    if source == "constructor":
        kwargs[f"{kind}_CONFIGS"] = value
    elif source == "dotenv":
        path = tmp_path / "test.env"
        path.write_text(f"{kind}_CONFIGS={raw}\n")
        kwargs["_env_file"] = path
    elif source == "base64":
        kwargs[f"{kind}_CONFIGS_BASE64"] = base64.b64encode(raw.encode()).decode()
    else:
        path = tmp_path / "config.json"
        path.write_text(raw)
        kwargs[f"{kind}_CONFIGS_PATH"] = str(path)
    settings = Settings(**kwargs)
    settings.setup()
    assert getattr(settings, f"{kind}_CONFIGS") == value


def make_prompt(content="Hello {{ name }}", **kwargs):
    backend = EmptyObservability()
    backend.push_prompt("test", [{"role": "system", "content": content}])
    return ObservabilityChatPromptTemplate(
        prompt_name="test", observability_platform=backend, template_format="jinja2", **kwargs
    )


def test_jinja_sandbox_preserves_dictionary_attributes():
    prompt = make_prompt("Hello {{ item.name }}")
    assert prompt.invoke({"item": {"name": "Jane"}}).to_messages()[0].content == "Hello Jane"
    unsafe = make_prompt("{{ cycler.__init__.__globals__.os.name }}")
    with pytest.raises(SecurityError):
        unsafe.invoke({})


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_jinja_invocation_keeps_parent_input_semantics(use_async):
    prompt = make_prompt()
    result = await prompt.ainvoke("Jane") if use_async else prompt.invoke("Jane")
    assert result.to_messages()[0].content == "Hello Jane"
    with pytest.raises(KeyError):
        if use_async:
            await prompt.ainvoke({})
        else:
            prompt.invoke({})


@pytest.mark.parametrize("template_format", ["jinja2", "f-string"])
def test_partial_variables_are_evaluated_and_not_required(template_format):
    content = "Hello {{ name }}" if template_format == "jinja2" else "Hello {name}"
    prompt = ObservabilityChatPromptTemplate(
        messages=[("system", content)],
        template_format=template_format,
        partial_variables={"name": lambda: "Jane"},
    )
    assert prompt.invoke({}).to_messages()[0].content == "Hello Jane"


def test_jinja_placeholder_keeps_conversion_and_window():
    prompt = ObservabilityChatPromptTemplate(
        messages=[MessagesPlaceholder("messages", n_messages=1)], template_format="jinja2"
    )
    result = prompt.invoke({"messages": [("human", "first"), ("human", "last")]})
    assert result.to_messages() == [HumanMessage(content="last")]


def test_prompt_invocation_keeps_callbacks():
    class Handler(BaseCallbackHandler):
        starts = 0

        def on_chain_start(self, *args, **kwargs):
            self.starts += 1

    handler = Handler()
    make_prompt().invoke({"name": "Jane"}, config={"callbacks": [handler]})
    assert handler.starts == 1


def test_missing_startup_prompt_requires_explicit_fallback():
    with pytest.raises(ValueError, match="fallback"):
        ObservabilityChatPromptTemplate(prompt_name="missing", observability_platform=EmptyObservability())
    prompt = ObservabilityChatPromptTemplate(
        messages=[("system", "fallback")], prompt_name="missing", observability_platform=EmptyObservability()
    )
    assert prompt.invoke({}).to_messages()[0].content == "fallback"


def test_empty_startup_prompt_is_rejected():
    backend = EmptyObservability()
    with patch.object(backend, "pull_prompt", return_value=None):
        with pytest.raises(ValueError, match="fallback"):
            ObservabilityChatPromptTemplate(prompt_name="empty", observability_platform=backend)


@pytest.mark.asyncio
async def test_runtime_partial_keeps_the_prompt_backend():
    prompt = make_prompt(load_at_runtime=True).partial(name="Jane")
    result = await prompt.ainvoke({})
    assert result.to_messages()[0].content == "Hello Jane"


def test_configuration_source_can_be_set_after_initial_setup(monkeypatch):
    monkeypatch.delenv("MODEL_CONFIGS", raising=False)
    settings = Settings(_env_file=None)
    settings.setup()
    settings.apply_overrides({"MODEL_CONFIGS_BASE64": "eyJleGFtcGxlIjp7Im5hbWUiOiJkdW1teSJ9fQ=="})
    settings.setup()
    assert settings.MODEL_CONFIGS == {"example": {"name": "dummy"}}


def test_langfuse_pinned_prompt_does_not_fall_back():
    backend = LangfuseObservability()
    client = MagicMock()
    client.get_prompt.side_effect = [ValueError("version unavailable"), MagicMock(prompt="wrong version")]
    with (
        patch.object(backend, "validate_environment"),
        patch("langgraph_agent_toolkit.core.observability.langfuse._get_langfuse_client", return_value=client),
    ):
        with pytest.raises(ValueError, match="version unavailable"):
            backend.pull_prompt("test", version=42)
    assert client.get_prompt.call_count == 1


@pytest.mark.asyncio
async def test_runtime_prompt_refresh_is_shared():
    class Backend(EmptyObservability):
        calls = 0

        async def apull_prompt(self, *args, **kwargs):
            self.calls += 1
            await asyncio.sleep(0.01)
            return ChatPromptTemplate.from_messages([("system", "Hello {name}")])

    backend = Backend()
    prompt = ObservabilityChatPromptTemplate(prompt_name="test", observability_platform=backend, load_at_runtime=True)
    results = await asyncio.gather(*(prompt.ainvoke("Jane") for _ in range(6)))
    assert backend.calls == 1
    assert all(result.to_messages()[0].content == "Hello Jane" for result in results)


@pytest.mark.asyncio
async def test_cancelled_prompt_waiter_does_not_cancel_shared_refresh():
    started = asyncio.Event()
    release = asyncio.Event()

    class Backend(EmptyObservability):
        calls = 0

        async def apull_prompt(self, *args, **kwargs):
            self.calls += 1
            started.set()
            await release.wait()
            return ChatPromptTemplate.from_messages([("system", "Hello {name}")])

    backend = Backend()
    prompt = ObservabilityChatPromptTemplate(prompt_name="test", observability_platform=backend, load_at_runtime=True)
    owner = asyncio.create_task(prompt.ainvoke("Jane"))
    await started.wait()
    waiter = asyncio.create_task(prompt.ainvoke("Jane"))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    assert (await owner).to_messages()[0].content == "Hello Jane"
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_async_prompt_manager_moves_all_io_off_event_loop(tmp_path):
    event_loop_thread = threading.get_ident()

    class Backend(EmptyObservability):
        pushes = 0

        def push_prompt(self, *args, **kwargs):
            assert threading.get_ident() != event_loop_thread
            self.pushes += 1
            return super().push_prompt(*args, **kwargs)

        def pull_prompt(self, *args, **kwargs):
            assert threading.get_ident() != event_loop_thread
            return super().pull_prompt(*args, **kwargs)

    (tmp_path / "test.j2").write_text("Hello {{ name }}")
    manager = PromptManager(observability_backend="empty", prompts_dir=tmp_path)
    manager._observability = Backend()
    results = await asyncio.gather(*(manager._aget_or_create_prompt("test", "test.j2", ["name"]) for _ in range(6)))
    assert manager._observability.pushes == 1
    assert all(prompt is results[0] for prompt in results)
    assert results[0].invoke("Jane").to_messages()[0].content == "Hello Jane"


def test_sync_prompt_manager_creates_once(tmp_path):
    class Backend(EmptyObservability):
        pushes = 0

        def push_prompt(self, *args, **kwargs):
            self.pushes += 1
            return super().push_prompt(*args, **kwargs)

    (tmp_path / "test.j2").write_text("Hello {{ name }}")
    manager = PromptManager(observability_backend="empty", prompts_dir=tmp_path)
    manager._observability = Backend()
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: manager._get_or_create_prompt("test", "test.j2", ["name"]), range(12)))
    assert manager._observability.pushes == 1
    assert all(prompt is results[0] for prompt in results)
