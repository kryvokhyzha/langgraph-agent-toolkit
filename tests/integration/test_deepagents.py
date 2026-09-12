"""Run DeepAgents tools through real graphs and the toolkit service."""

import importlib
import sys
from types import ModuleType
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import Field, SecretStr

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.core.memory.types import MemoryBackends
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper import constants
from langgraph_agent_toolkit.helper.types import EnvironmentMode
from langgraph_agent_toolkit.service.auth import storage_thread_id
from langgraph_agent_toolkit.service.handler import create_app


pytest.importorskip("deepagents")
blueprint = importlib.import_module("langgraph_agent_toolkit.agents.blueprints.deep_agent.agent")
HEADERS = {"Authorization": "Bearer deepagents-test-credential"}


class ScriptedModel(FakeMessagesListChatModel):
    seen: list = Field(default_factory=list)
    bound_tool_names: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names.append({tool.name for tool in tools})
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)

    def get_num_tokens(self, text):
        return len(text.split())

    def get_num_tokens_from_messages(self, messages, tools=None):
        return sum(self.get_num_tokens(str(message.content)) for message in messages)


def call(name, **arguments):
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": uuid4().hex}])


def tool_results(messages, name):
    return [message for message in messages if isinstance(message, ToolMessage) and message.name == name]


def configure_service(monkeypatch, tmp_path, graph):
    module = ModuleType("lat_deepagents_test_" + uuid4().hex)
    module.agent = Agent("deep-agent", "A DeepAgents integration test.", graph)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for key, value in {
        "ENV_MODE": EnvironmentMode.PRODUCTION,
        "AUTH_MODE": "trusted",
        "AUTH_SECRET": SecretStr("deepagents-test-credential"),
        "AUTH_USERS": {},
        "MEMORY_BACKEND": MemoryBackends.SQLITE,
        "SQLITE_DB_PATH": str(tmp_path / "deepagents.sqlite"),
        "OBSERVABILITY_BACKEND": ObservabilityBackend.EMPTY,
        "AGENT_PATHS": [f"{module.__name__}:agent"],
        "DEFAULT_AGENT": "deep-agent",
        "MCP_SERVERS": {},
        "MCP_AGENT_SERVERS": {},
    }.items():
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(constants, "_runtime_default_agent", "deep-agent")
    return module


def invoke(client, message, *, thread="first-thread", user="user-1"):
    response = client.post(
        "/deep-agent/invoke",
        headers=HEADERS,
        json={"input": {"message": message}, "thread_id": thread, "user_id": user},
    )
    assert response.status_code == 200, response.text
    return response.json()


async def test_planning_and_virtual_files_do_not_write_to_the_host(tmp_path):
    host_path = tmp_path / "host-must-not-change.txt"
    model = ScriptedModel(
        responses=[
            call("write_todos", todos=[{"content": "Prepare note", "status": "in_progress"}]),
            call("write_file", file_path=str(host_path), content="The virtual note."),
            call("read_file", file_path=str(host_path)),
            call("write_todos", todos=[{"content": "Prepare note", "status": "completed"}]),
            AIMessage(content="The note is ready."),
        ]
    )
    graph = blueprint.build_graph(model=model)
    result = await graph.ainvoke({"messages": [HumanMessage("Prepare a note.")]})

    assert result["messages"][-1].content == "The note is ready."
    assert "The virtual note." in tool_results(result["messages"], "read_file")[0].content
    assert result["todos"] == [{"content": "Prepare note", "status": "completed"}]
    assert not host_path.exists()
    assert model.bound_tool_names
    assert all("execute" not in names for names in model.bound_tool_names)
    assert graph.checkpointer is None


def test_virtual_files_survive_service_restart_and_remain_private(monkeypatch, tmp_path):
    writer = ScriptedModel(
        responses=[
            call("write_file", file_path="/private-note.txt", content="User one private note."),
            AIMessage("Saved."),
        ]
    )
    module = configure_service(monkeypatch, tmp_path, blueprint.build_graph(model=writer))
    with TestClient(create_app()) as client:
        assert invoke(client, "Save a private note.")["content"] == "Saved."

    reader = ScriptedModel(responses=[call("read_file", file_path="/private-note.txt"), AIMessage("Read complete.")])
    module.agent.graph = blueprint.build_graph(model=reader)
    with TestClient(create_app()) as client:
        invoke(client, "Read my note.")
        recovered = tool_results(reader.seen[-1], "read_file")[-1]
        assert recovered.status == "success"
        assert "User one private note." in recovered.content

        invoke(client, "Read that path.", thread="second-thread")
        separate_thread = tool_results(reader.seen[-1], "read_file")[-1]
        assert separate_thread.status == "error"
        assert all("User one private note." not in message.model_dump_json() for message in reader.seen[-1])

        invoke(client, "Read that path.", user="user-2")
        separate_user = tool_results(reader.seen[-1], "read_file")[-1]
        assert separate_user.status == "error"
        assert all("User one private note." not in message.model_dump_json() for message in reader.seen[-1])


async def test_subagent_returns_a_file_without_exposing_its_private_conversation():
    specialist = ScriptedModel(
        responses=[
            call("write_file", file_path="/result.txt", content="The specialist result."),
            AIMessage(content="The result is in /result.txt."),
        ]
    )
    parent = ScriptedModel(
        responses=[
            call("task", subagent_type="specialist", description="Write the requested result."),
            call("read_file", file_path="/result.txt"),
            AIMessage(content="The delegated task is complete."),
        ]
    )
    graph = blueprint.build_graph(
        model=parent,
        subagents=[{"name": "specialist", "description": "Write a result file.", "model": specialist}],
    )
    result = await graph.ainvoke({"messages": [HumanMessage("Parent-only detail: private planning context.")]})

    delegated = tool_results(result["messages"], "task")
    assert len(delegated) == 1
    assert "The result is in /result.txt." in delegated[0].content
    assert "The specialist result." in tool_results(result["messages"], "read_file")[0].content
    assert not tool_results(result["messages"], "write_file")
    assert all("Parent-only detail" not in str(message.content) for message in specialist.seen[0])
    assert result["messages"][-1].content == "The delegated task is complete."


def test_file_approval_survives_restart_and_executes_only_after_resume(monkeypatch, tmp_path):
    model = ScriptedModel(
        responses=[call("write_file", file_path="/approved.txt", content="Approved content."), AIMessage("Written.")]
    )
    module = configure_service(
        monkeypatch, tmp_path, blueprint.build_graph(model=model, interrupt_on={"write_file": True})
    )
    with TestClient(create_app()) as client:
        paused = invoke(client, "Write the file after approval.")
        requests = paused["custom_data"]["interrupts"][0]["value"]["action_requests"]
        assert requests[0]["name"] == "write_file"
        graph = client.app.state.agent_executor.get_agent("deep-agent").graph
        config = {"configurable": {"thread_id": storage_thread_id("user-1", "deep-agent", "first-thread")}}
        state = client.portal.call(graph.aget_state, config)
        assert "/approved.txt" not in state.values.get("files", {})

    resumed_model = ScriptedModel(
        responses=[call("read_file", file_path="/approved.txt"), AIMessage("Approved and read.")]
    )
    module.agent.graph = blueprint.build_graph(model=resumed_model, interrupt_on={"write_file": True})
    with TestClient(create_app()) as client:
        completed = invoke(client, "approve")
        assert completed["content"] == "Approved and read."
        writes = tool_results(resumed_model.seen[-1], "write_file")
        assert len(writes) == 1
        assert writes[0].status == "success"
        assert "Approved content." in tool_results(resumed_model.seen[-1], "read_file")[0].content


@pytest.mark.parametrize("fail_after_write", [False, True])
async def test_mcp_tool_runs_once_and_closes_its_connection(monkeypatch, fail_after_write):
    fastmcp = pytest.importorskip("fastmcp")
    adapter_class = pytest.importorskip("langchain.mcp").MCPAdapter
    writes = []
    server = fastmcp.FastMCP("deepagents-local-test")

    @server.tool
    async def remote_write(value: str) -> dict:
        """Store one value in the test server."""
        writes.append(value)
        if fail_after_write:
            raise RuntimeError("The write completed but the response failed.")
        return {"saved": value}

    adapter = adapter_class(fastmcp.Client(server, timeout=5, init_timeout=5))
    tools = await adapter.list_tools()
    assert not adapter.client.is_connected()
    model = ScriptedModel(responses=[call("remote_write", value="one"), AIMessage("Tool attempt complete.")])
    monkeypatch.setattr(blueprint, "create_model", lambda: model)
    graph = blueprint.deep_agent.graph_factory(tools)

    result = await graph.ainvoke({"messages": [HumanMessage("Store one value.")]})

    assert writes == ["one"]
    messages = tool_results(result["messages"], "remote_write")
    assert len(messages) == 1
    assert messages[0].status == ("error" if fail_after_write else "success")
    if not fail_after_write:
        assert messages[0].artifact == {"structured_content": {"saved": "one"}}
    assert not adapter.client.is_connected()
