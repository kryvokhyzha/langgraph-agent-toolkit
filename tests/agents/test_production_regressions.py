"""Exercise graph regressions without model or retrieval services."""

import importlib
import os
import subprocess
import sys
from typing import Annotated
from unittest.mock import AsyncMock, Mock

import pytest
from langchain.chat_models.base import _ConfigurableModel
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import Field

from langgraph_agent_toolkit.agents.components.creators.create_react_agent import _get_model, create_react_agent
from langgraph_agent_toolkit.core.settings import settings


class RecordingModel(FakeMessagesListChatModel):
    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


class TenantState(AgentState):
    tenant: str


def test_immediate_model_removes_tools_from_configurable_bindings(monkeypatch):
    model = RecordingModel(responses=[AIMessage("finished")])
    configurable = _ConfigurableModel(default_config={})
    monkeypatch.setattr(_ConfigurableModel, "_model", lambda self, config: model.bind(tools=["lookup"]))
    assert _get_model(configurable, {}) is model


def build_tool_graph(version):
    @tool
    def lookup(
        key: str,
        tenant: Annotated[str, InjectedState("tenant")],
        store: Annotated[BaseStore, InjectedStore()],
    ) -> str:
        """Read a value from the current tenant's store."""
        return store.get((tenant,), key).value["value"]

    store = InMemoryStore()
    store.put(("tenant-a",), "one", {"value": "first result"})
    store.put(("tenant-a",), "two", {"value": "second result"})
    model = RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "lookup", "args": {"key": key}, "id": key, "type": "tool_call"} for key in ("one", "two")
                ],
            ),
            AIMessage(content="finished"),
        ]
    )
    return create_react_agent(model, [lookup], version=version, state_schema=TenantState, store=store)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_custom_creator_executes_tools_with_injected_state_and_store(version):
    graph = build_tool_graph(version)
    result = graph.invoke({"messages": [HumanMessage("look up both keys")], "tenant": "tenant-a"})
    assert {m.content for m in result["messages"] if isinstance(m, ToolMessage)} == {
        "first result",
        "second result",
    }
    assert result["messages"][-1].content == "finished"


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_custom_creator_executes_tools_asynchronously(version):
    graph = build_tool_graph(version)
    result = await graph.ainvoke({"messages": [HumanMessage("look up both keys")], "tenant": "tenant-a"})
    assert len([m for m in result["messages"] if isinstance(m, ToolMessage)]) == 2
    assert result["messages"][-1].content == "finished"


@pytest.mark.parametrize("prompt_template", [False, True])
@pytest.mark.parametrize("async_execution", [False, True])
@pytest.mark.asyncio
async def test_immediate_fallback_preserves_configured_prompt(prompt_template, async_execution):
    instruction = "Always answer in Ukrainian and redact secrets."
    prompt = (
        ChatPromptTemplate.from_messages([("system", instruction), ("placeholder", "{messages}")])
        if prompt_template
        else instruction
    )
    model = RecordingModel(responses=[AIMessage("finished")])
    graph = create_react_agent(model, [], prompt=prompt)
    inputs = {"messages": [HumanMessage("hello")]}
    config = {"recursion_limit": 4}
    if async_execution:
        await graph.ainvoke(inputs, config=config)
    else:
        graph.invoke(inputs, config=config)
    assert isinstance(model.seen[-1][0], SystemMessage)
    assert instruction in model.seen[-1][0].content
    assert "direct answer" in model.seen[-1][0].content
    assert model.seen[-1][-1].content == "hello"


@pytest.mark.asyncio
@pytest.mark.parametrize("second_retrieval", [[], TimeoutError("retrieval failed")])
async def test_kb_clears_previous_documents_after_miss_or_error(monkeypatch, second_retrieval):
    module = importlib.import_module("langgraph_agent_toolkit.agents.blueprints.knowledge_base_agent.agent")
    retriever = Mock()
    retriever.ainvoke = AsyncMock(side_effect=[[Document(page_content="product A only")], second_retrieval])
    monkeypatch.setattr(module, "get_kb_retriever", lambda: retriever)
    model = RecordingModel(responses=[AIMessage("first"), AIMessage("second")])
    monkeypatch.setattr(module.CompletionModelFactory, "create", Mock(return_value=model))
    monkeypatch.setattr(module.kb_agent.graph, "checkpointer", MemorySaver())
    config = {"configurable": {"thread_id": "kb-regression", "model_provider": "fake"}}
    await module.kb_agent.graph.ainvoke({"messages": [HumanMessage("product A")]}, config=config)
    result = await module.kb_agent.graph.ainvoke({"messages": [HumanMessage("product B")]}, config=config)
    assert result["retrieved_documents"] == []
    assert result["kb_documents"] == ""
    assert "product A only" not in model.seen[-1][0].content
    assert "No relevant documents" in model.seen[-1][0].content


@pytest.mark.asyncio
@pytest.mark.parametrize("blueprint", ["chatbot", "knowledge_base_agent", "bg_task_agent"])
async def test_blueprint_uses_top_level_model_config_key(monkeypatch, blueprint):
    module = importlib.import_module(f"langgraph_agent_toolkit.agents.blueprints.{blueprint}.agent")
    model_config = {"provider": "fake", "name": "named-fake", "temperature": 0.4}
    monkeypatch.setattr(settings, "MODEL_CONFIGS", {"named": model_config})
    model = RecordingModel(responses=[AIMessage("configured response")])
    from_config = Mock(return_value=model)
    monkeypatch.setattr(module.CompletionModelFactory, "get_model_from_config", from_config)
    monkeypatch.setattr(module.CompletionModelFactory, "create", Mock(side_effect=AssertionError("wrong factory")))
    state = {"messages": [HumanMessage("hello")]}
    config = {"configurable": {"model_config_key": "named"}}
    if blueprint == "chatbot":
        result = await module.chatbot.ainvoke(state, config=config)
    else:
        result = await module.acall_model(state, config)
    from_config.assert_called_once_with(model_config)
    assert result["messages"][-1].content == "configured response"


def test_blueprints_import_offline_and_allow_service_saver_injection(tmp_path):
    """Import all blueprints without cached modules, credentials, or network access."""
    script = """
import os
if os.environ.get("COVERAGE_PROCESS_START") or os.environ.get("COVERAGE_PROCESS_CONFIG"):
    import coverage
    if coverage.Coverage.current() is None:
        coverage.process_startup()

import importlib
import importlib.util
import socket
import dotenv

dotenv.find_dotenv = lambda *args, **kwargs: ""
dotenv.load_dotenv = lambda *args, **kwargs: False

def reject_connection(*args, **kwargs):
    raise AssertionError("Blueprint imports must not open network connections.")

socket.socket.connect = reject_connection
socket.create_connection = reject_connection

from langgraph.checkpoint.memory import MemorySaver
from langgraph_agent_toolkit.agents.agent import Agent

blueprints = [
    "chatbot", "command_agent", "interrupt_agent", "bg_task_agent",
    "knowledge_base_agent", "create_agent", "create_agent_structured",
    "hitl_agent", "react", "supervisor_agent",
]
if importlib.util.find_spec("deepagents") is not None:
    blueprints.append("deep_agent")
for blueprint in blueprints:
    module = importlib.import_module(f"langgraph_agent_toolkit.agents.blueprints.{blueprint}.agent")
    agents = [value for value in vars(module).values() if isinstance(value, Agent)]
    assert agents, blueprint
    for agent in agents:
        assert agent.graph.checkpointer is None, blueprint
        saver = MemorySaver()
        agent.graph.checkpointer = saver
        assert agent.graph.checkpointer is saver, blueprint
"""
    # Keep coverage collection when the child excludes the parent environment.
    child_env = {
        key: os.environ[key]
        for key in ("COVERAGE_PROCESS_START", "COVERAGE_PROCESS_CONFIG", "COVERAGE_FILE")
        if key in os.environ
    }
    child_env.update(PATH=os.defpath, HOME=str(tmp_path), ENV_MODE="development", USE_FAKE_MODEL="true")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
