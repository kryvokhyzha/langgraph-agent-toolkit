"""Exercise native tool selection with real graphs and isolated requests."""

import asyncio
from typing import Any

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from pydantic import Field, ValidationError

from langgraph_agent_toolkit.agents.blueprints.create_agent._shared import build_tool_graph
from langgraph_agent_toolkit.core._base_settings import Settings
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.core.tool_selection import (
    ToolSelectionConfig,
    ToolSelectionMiddleware,
    create_tool_selection_middleware,
)


class SelectorModel(BaseChatModel):
    responses: list[Any] = Field(default_factory=list)
    calls: int = 0
    schemas: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "scripted-tool-selector"

    def _generate(self, *args, **kwargs):
        raise AssertionError("Use structured output for selection")

    def with_structured_output(self, schema, **kwargs):
        self.schemas.append(schema)

        def choose(messages):
            self.calls += 1
            if self.responses:
                return self.responses.pop(0)
            text = messages[-1].content
            return {"tools": ["multiply" if "multiply" in text else "add"]}

        async def choose_async(messages):
            await asyncio.sleep(0)
            return choose(messages)

        return RunnableLambda(choose, afunc=choose_async)


class ExecutingModel(BaseChatModel):
    seen: list[tuple[str, list[str]]] = Field(default_factory=list)
    write_first: bool = False

    @property
    def _llm_type(self):
        return "scripted-tool-executor"

    def bind_tools(self, tools, **kwargs):
        return self.bind(available_tools=[item.name if hasattr(item, "name") else item["name"] for item in tools])

    def _generate(self, messages, stop=None, run_manager=None, available_tools=(), **kwargs):
        user = next(item.content for item in reversed(messages) if isinstance(item, HumanMessage))
        self.seen.append((user, list(available_tools)))
        tool_results = [item for item in messages if isinstance(item, ToolMessage)]
        if self.write_first and not tool_results:
            name, arguments = "write_file", {"file_path": "/notes.txt", "content": "Saved notes."}
        elif not tool_results or (self.write_first and tool_results[-1].name == "write_file"):
            name = "multiply" if "multiply" in user else "add"
            arguments = {"a": 2, "b": 3}
        else:
            return ChatResult(generations=[ChatGeneration(message=AIMessage("Completed: " + tool_results[-1].content))])
        if name not in available_tools:
            return ChatResult(generations=[ChatGeneration(message=AIMessage("No matching tool is visible."))])
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage("", tool_calls=[{"name": name, "args": arguments, "id": name}]))
            ]
        )


@tool
def unrelated_lookup(key: str) -> str:
    """Read an unrelated reference."""
    raise AssertionError("The unrelated tool must not run")


@pytest.mark.parametrize(
    "configuration",
    [
        {"mode": "unknown"},
        {"max_tools": 0},
        {"max_retries": -1},
        {"max_retries": 4},
        {"always_include": [""]},
        {"always_include": ["add", "add"]},
        {"always_include": [" add"]},
        {"model_config_key": ""},
        {"on_parsing_failure": "ignore"},
        {"unknown": True},
    ],
)
def test_invalid_configuration_fails_before_model_calls(configuration):
    with pytest.raises(ValidationError):
        ToolSelectionConfig(**configuration)


def test_settings_validate_json_overrides_and_default_to_off(mock_env):
    configuration = Settings(_env_file=None)
    assert configuration.TOOL_SELECTION.mode == "off"
    configuration.apply_overrides({"TOOL_SELECTION": {"mode": "llm", "max_tools": 2}})
    assert isinstance(configuration.TOOL_SELECTION, ToolSelectionConfig)
    assert configuration.TOOL_SELECTION.max_tools == 2
    with pytest.raises(ValidationError):
        configuration.apply_overrides({"TOOL_SELECTION": {"mode": "llm", "max_tools": 0}})
    assert configuration.TOOL_SELECTION.max_tools == 2


def test_disabled_configuration_does_not_resolve_selector_models():
    assert create_tool_selection_middleware(ToolSelectionConfig(model_config_key="missing")) == []


def test_named_selector_configuration_must_exist(monkeypatch):
    monkeypatch.setattr(settings, "MODEL_CONFIGS", {})
    with pytest.raises(ValueError, match="model_config_key is not in MODEL_CONFIGS"):
        create_tool_selection_middleware(ToolSelectionConfig(mode="llm", model_config_key="missing"))


def test_named_selector_model_uses_the_model_factory(monkeypatch):
    monkeypatch.setattr(settings, "MODEL_CONFIGS", {"selector": {"provider": "fake", "name": "selector"}})
    middleware = create_tool_selection_middleware(ToolSelectionConfig(mode="llm", model_config_key="selector"))[0]
    assert isinstance(middleware.selector_model, BaseChatModel)
    with pytest.raises(ValueError, match="not both"):
        create_tool_selection_middleware(
            ToolSelectionConfig(mode="llm", model_config_key="selector"), selector_model=SelectorModel()
        )


@pytest.mark.parametrize("mode,max_tools,expected_calls", [("off", 1, 0), ("llm", 8, 0), ("llm", 1, 2)])
async def test_native_graph_completes_with_optional_selection(mode, max_tools, expected_calls):
    selector, model = SelectorModel(), ExecutingModel()
    graph = build_tool_graph(
        model,
        [unrelated_lookup],
        tool_selection=ToolSelectionConfig(mode=mode, max_tools=max_tools),
        selector_model=selector,
    )
    result = await graph.ainvoke({"messages": [HumanMessage("add two numbers")]})
    assert result["messages"][-1].content == "Completed: 5.0"
    assert len([item for item in result["messages"] if isinstance(item, ToolMessage)]) == 1
    assert selector.calls == expected_calls
    if expected_calls:
        assert all(names == ["add"] for _, names in model.seen)
    else:
        assert all({"add", "multiply", "unrelated_lookup"} <= set(names) for _, names in model.seen)


def test_synchronous_native_graph_uses_the_same_selector():
    selector, model = SelectorModel(), ExecutingModel()
    graph = build_tool_graph(
        model, tool_selection=ToolSelectionConfig(mode="llm", max_tools=1), selector_model=selector
    )
    result = graph.invoke({"messages": [HumanMessage("multiply two numbers")]})
    assert result["messages"][-1].content == "Completed: 6.0"
    assert selector.calls == 2
    assert all(names == ["multiply"] for _, names in model.seen)


async def test_concurrent_graph_requests_keep_separate_tool_views():
    selector, model = SelectorModel(), ExecutingModel()
    graph = build_tool_graph(
        model, tool_selection=ToolSelectionConfig(mode="llm", max_tools=1), selector_model=selector
    )
    inputs = ["add" if index % 2 else "multiply" for index in range(8)]
    results = await asyncio.gather(*(graph.ainvoke({"messages": [HumanMessage(text)]}) for text in inputs))
    assert [item["messages"][-1].content for item in results] == [
        "Completed: 5.0" if text == "add" else "Completed: 6.0" for text in inputs
    ]
    assert all(names == [user] for user, names in model.seen)
    assert selector.calls == 16


async def test_caller_filter_runs_before_selection_and_retains_the_request_boundary():
    class PermitAddition(AgentMiddleware):
        async def awrap_model_call(self, request, handler):
            return await handler(request.override(tools=[item for item in request.tools if item.name == "add"]))

    selector = SelectorModel(responses=[{"tools": ["multiply"]}])
    model = ExecutingModel()
    graph = build_tool_graph(
        model,
        tool_selection=ToolSelectionConfig(mode="llm", max_tools=1),
        selector_model=selector,
        extra_middleware=[PermitAddition()],
    )
    result = await graph.ainvoke({"messages": [HumanMessage("add")]})
    assert result["messages"][-1].content == "Completed: 5.0"
    assert selector.calls == 0
    assert all(names == ["add"] for _, names in model.seen)


@pytest.mark.parametrize("policy,expected", [("all", {"add", "multiply"}), ("none", {"add"})])
async def test_malformed_selection_fallback_keeps_required_tools_and_never_executes_tools(policy, expected):
    from langgraph_agent_toolkit.agents.components.tools import add, multiply

    selector = SelectorModel(responses=[None, None])
    middleware = ToolSelectionMiddleware(
        ToolSelectionConfig(mode="llm", max_tools=1, always_include=["add"], on_parsing_failure=policy),
        selector_model=selector,
    )
    # Include two optional tools so the small-catalog fast path does not apply.
    request = ModelRequest(
        model=ExecutingModel(), messages=[HumanMessage("add")], tools=[add, multiply, unrelated_lookup]
    )
    result = await middleware.awrap_model_call(request, lambda item: asyncio.sleep(0, result=item))
    expected = expected | ({"unrelated_lookup"} if policy == "all" else set())
    assert {item.name for item in result.tools} == expected
    assert selector.calls == 2
    assert {item.name for item in request.tools} == {"add", "multiply", "unrelated_lookup"}


@pytest.mark.parametrize("response", [None, {"tools": ["not_registered"]}])
async def test_invalid_selection_fails_without_calling_the_execution_model(response):
    selector, model = SelectorModel(responses=[response, response]), ExecutingModel()
    graph = build_tool_graph(
        model, tool_selection=ToolSelectionConfig(mode="llm", max_tools=1), selector_model=selector
    )
    with pytest.raises(ValueError, match="malformed|invalid tools"):
        await graph.ainvoke({"messages": [HumanMessage("add")]})
    assert model.seen == []


async def test_required_names_are_validated_without_mutating_the_configuration():
    from langgraph_agent_toolkit.agents.components.tools import add

    configuration = ToolSelectionConfig(mode="llm", always_include=["missing"])
    middleware = ToolSelectionMiddleware(configuration, selector_model=SelectorModel())
    configuration.always_include.clear()
    request = ModelRequest(model=ExecutingModel(), messages=[HumanMessage("add")], tools=[add])
    with pytest.raises(ValueError, match="always_include names are not available"):
        await middleware.awrap_model_call(request, lambda item: asyncio.sleep(0, result=item))


async def test_immediate_answer_bypasses_selector_and_does_not_restore_tools(monkeypatch):
    monkeypatch.setattr(settings, "DEFAULT_RECURSION_LIMIT", 2)
    selector, model = SelectorModel(), ExecutingModel()
    graph = build_tool_graph(
        model,
        tool_selection=ToolSelectionConfig(mode="llm", max_tools=1),
        selector_model=selector,
    )
    result = await graph.ainvoke({"messages": [HumanMessage("add")]})
    assert result["messages"][-1].content == "No matching tool is visible."
    assert model.seen == [("add", [])]
    assert selector.calls == 0


async def test_deep_agent_keeps_file_and_planning_tools_with_selection():
    pytest.importorskip("deepagents")
    from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph

    selector, model = SelectorModel(), ExecutingModel(write_first=True)
    graph = build_graph(
        [unrelated_lookup],
        model=model,
        tool_selection=ToolSelectionConfig(mode="llm", max_tools=1),
        selector_model=selector,
    )
    result = await graph.ainvoke({"messages": [HumanMessage("add after saving notes")]})
    assert result["messages"][-1].content == "Completed: 5.0"
    assert result["files"]["/notes.txt"]["content"] == "Saved notes."
    assert selector.calls == 3
    required = {"task", "write_todos", "ls", "read_file", "write_file", "edit_file", "glob", "grep"}
    assert all(required <= set(names) for _, names in model.seen)
    assert all("multiply" not in names and "unrelated_lookup" not in names for _, names in model.seen)


async def test_hitl_selection_preserves_approval_and_runs_the_tool_once(monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from langgraph_agent_toolkit.agents.blueprints.hitl_agent import agent as blueprint

    class EmailModel(ExecutingModel):
        def _generate(self, messages, stop=None, run_manager=None, available_tools=(), **kwargs):
            self.seen.append(("email", list(available_tools)))
            if isinstance(messages[-1], ToolMessage):
                message = AIMessage("Completed the approved operation.")
            else:
                message = AIMessage(
                    "",
                    tool_calls=[
                        {
                            "name": "send_email",
                            "args": {"recipient": "user@example.com", "subject": "Test", "body": "Synthetic message"},
                            "id": "approved-operation",
                        }
                    ],
                )
            return ChatResult(generations=[ChatGeneration(message=message)])

    model = EmailModel()
    monkeypatch.setattr(blueprint, "model", model)
    selector = SelectorModel(responses=[{"tools": ["send_email"]}, {"tools": ["send_email"]}])
    graph = blueprint.build_graph(
        [unrelated_lookup],
        tool_selection=ToolSelectionConfig(mode="llm", max_tools=1),
        selector_model=selector,
    )
    graph.checkpointer = InMemorySaver()
    configuration = {"configurable": {"thread_id": "tool-selection-approval"}}
    paused = await graph.ainvoke({"messages": [HumanMessage("Send a synthetic message.")]}, configuration)
    assert paused["__interrupt__"]
    assert not any(isinstance(item, ToolMessage) for item in paused["messages"])
    finished = await graph.ainvoke(Command(resume={"decisions": [{"type": "approve"}]}), configuration)
    assert finished["messages"][-1].content == "Completed the approved operation."
    assert len([item for item in finished["messages"] if isinstance(item, ToolMessage)]) == 1
    assert model.seen == [("email", ["send_email"]), ("email", ["send_email"])]
