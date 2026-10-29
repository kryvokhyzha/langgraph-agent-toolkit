"""Recover MCP questions and tool results through HTTP and durable SQLite."""

import json
import sys
from types import ModuleType
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import SecretStr

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.core.memory.types import MemoryBackends
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper import constants
from langgraph_agent_toolkit.helper.types import EnvironmentMode
from langgraph_agent_toolkit.schema import ChatMessage
from langgraph_agent_toolkit.service.handler import create_app


ALICE = {"Authorization": "Bearer mcp-history-alice"}
BOB = {"Authorization": "Bearer mcp-history-bob"}


def configure_service(monkeypatch, tmp_path, graph):
    module = ModuleType("lat_mcp_history_" + uuid4().hex)
    module.agent = Agent("mcp-history", "A durable MCP test agent.", graph)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for key, value in {
        "ENV_MODE": EnvironmentMode.PRODUCTION,
        "AUTH_MODE": "token",
        "AUTH_SECRET": None,
        "AUTH_USERS": {"alice": SecretStr("mcp-history-alice"), "bob": SecretStr("mcp-history-bob")},
        "MEMORY_BACKEND": MemoryBackends.SQLITE,
        "SQLITE_DB_PATH": str(tmp_path / "mcp-history.sqlite"),
        "OBSERVABILITY_BACKEND": ObservabilityBackend.EMPTY,
        "AGENT_PATHS": [f"{module.__name__}:agent"],
        "DEFAULT_AGENT": "mcp-history",
        "MCP_SERVERS": {},
    }.items():
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(constants, "_runtime_default_agent", "mcp-history")


def read_history(http, *, headers=ALICE, path="/history", **params):
    response = http.get(path, headers=headers, params={"thread_id": "shared", **params})
    assert response.status_code == 200, response.text
    return response.json()


def mcp_tool_graph(server, count=1):
    from langchain.mcp import MCPAdapter

    builder = StateGraph(MessagesState)
    for index in range(count):
        label = str(index)

        async def node(state, label=label):
            tools = await MCPAdapter(server).list_tools()
            call = {"name": tools[0].name, "args": {"label": label}, "id": f"call-{label}", "type": "tool_call"}
            result = await tools[0].ainvoke(call)
            return {"messages": [AIMessage("", tool_calls=[call]), result, AIMessage(f"Completed {label}.")]}

        builder.add_node(label, node)
        builder.add_edge(START, label)
        builder.add_edge(label, END)
    return builder.compile()


@pytest.mark.parametrize("count", [1, 2])
def test_saved_mcp_questions_recover_after_sqlite_reopen_without_executing_tools(count, monkeypatch, tmp_path):
    fastmcp = pytest.importorskip("fastmcp")
    from fastmcp import Context
    from mcp.types import ElicitRequest, ElicitRequestFormParams, InputRequiredResult

    server = fastmcp.FastMCP("Durable questions")
    requests = []
    completed = []

    @server.tool
    async def request_answer(label: str, ctx: Context) -> Any:
        """Request an answer before the tool changes external state."""
        requests.append(label)
        if ctx.input_responses is None:
            return InputRequiredResult(
                input_requests={
                    "details": ElicitRequest(
                        params=ElicitRequestFormParams(
                            message=f"Answer for {label}.",
                            requested_schema={
                                "type": "object",
                                "properties": {"answer": {"type": "string"}},
                                "required": ["answer"],
                            },
                        )
                    )
                },
                request_state=label,
            )
        assert ctx.request_state == label
        value = ctx.input_responses["details"].content["answer"]
        completed.append((label, value))
        return {"label": label, "answer": value}

    configure_service(monkeypatch, tmp_path, mcp_tool_graph(server, count))
    with TestClient(create_app()) as http:
        paused = http.post("/invoke", headers=ALICE, json={"thread_id": "shared", "input": {"message": "Begin."}})
        assert paused.status_code == 200, paused.text
        live_pending = paused.json()["custom_data"]["interrupts"]
        assert len(live_pending) == count
        before_read = list(requests)
        # A lost response can be recovered from the saved state on any history page.
        page = read_history(http, limit=1)
        assert page["pending_interrupts"] == live_pending
        assert [message["content"] for message in page["messages"]] == ["Begin."]
        assert read_history(http, offset=99, limit=1)["pending_interrupts"] == live_pending
        assert read_history(http, path="/mcp-history/history")["pending_interrupts"] == live_pending
        assert read_history(http, headers=BOB)["pending_interrupts"] == []
        spoof = http.get("/history", headers=BOB, params={"thread_id": "shared", "user_id": "alice"})
        assert spoof.status_code == 403
        assert requests == before_read
        assert completed == []

        discarded = http.post("/invoke", headers=ALICE, json={"thread_id": "discarded", "input": {"message": "Begin."}})
        assert discarded.status_code == 200, discarded.text
        assert len(read_history(http, thread_id="discarded")["pending_interrupts"]) == count
        cleared = http.request("DELETE", "/history/clear", headers=ALICE, json={"thread_id": "discarded"})
        assert cleared.status_code == 200, cleared.text
        assert read_history(http, thread_id="discarded")["pending_interrupts"] == []
        assert completed == []
        before_read = list(requests)

    # Service shutdown closes the saver. A new service reopens the same database.
    with TestClient(create_app()) as http:
        assert read_history(http, thread_id="discarded")["pending_interrupts"] == []
        with AgentClient(
            agent="mcp-history", get_info=False, http_client=http, auth_secret="mcp-history-alice"
        ) as client:
            recovered = client.get_history("shared", offset=99, limit=1)
        assert recovered.messages == []
        assert recovered.pending_interrupts == live_pending
        assert requests == before_read
        answers = {
            item["id"]: {"responses": {"details": {"action": "accept", "content": {"answer": item["id"]}}}}
            for item in recovered.pending_interrupts
        }
        resumed = http.post("/invoke", headers=ALICE, json={"thread_id": "shared", "input": {"resume": answers}})
        assert resumed.status_code == 200, resumed.text
        assert len(completed) == count
        assert {answer for _, answer in completed} == {item["id"] for item in live_pending}
        before_read = list(requests)
        history = read_history(http, limit=1)
        assert history["total"] == 1 + 3 * count
        assert history["next_offset"] == 1
        assert history["pending_interrupts"] == []
        assert requests == before_read
        cleared = http.request("DELETE", "/history/clear", headers=ALICE, json={"thread_id": "shared"})
        assert cleared.status_code == 200, cleared.text
        assert read_history(http)["pending_interrupts"] == []
        assert len(completed) == count


@pytest.mark.parametrize("protocol", ["stream", "stream_jsonl"])
@pytest.mark.parametrize("failed", [False, True])
def test_real_mcp_tool_metadata_survives_stream_history_and_sqlite_reopen(protocol, failed, monkeypatch, tmp_path):
    fastmcp = pytest.importorskip("fastmcp")
    from mcp.types import CallToolResult, TextContent

    server = fastmcp.FastMCP("Tool result metadata")

    @server.tool
    async def read_result(label: str) -> Any:
        """Return a structured result or an MCP tool error."""
        if failed:
            return CallToolResult(content=[TextContent(type="text", text="Remote tool failed.")], is_error=True)
        return {"label": label, "nested": {"items": [1, None, True]}, "empty": {}}

    configure_service(monkeypatch, tmp_path, mcp_tool_graph(server))
    with TestClient(create_app()) as http:
        with AgentClient(
            agent="mcp-history", get_info=False, http_client=http, auth_secret="mcp-history-alice"
        ) as client:
            events = list(getattr(client, protocol)({"message": "Read."}, thread_id="shared"))
        messages = [event for event in events if isinstance(event, ChatMessage)]
        tools = [message for message in messages if message.type == "tool"]
        assert len(tools) == 1
        tool = tools[0]
        assert tool.tool_status == ("error" if failed else "success")
        expected_artifact = (
            None
            if failed
            else {"structured_content": {"label": "0", "nested": {"items": [1, None, True]}, "empty": {}}}
        )
        assert tool.tool_artifact == expected_artifact
        assert json.loads(tool.model_dump_json())["tool_artifact"] == expected_artifact
        saved = [message for message in read_history(http)["messages"] if message["type"] == "tool"]
        assert saved[0]["tool_status"] == tool.tool_status
        assert saved[0]["tool_artifact"] == expected_artifact

    with TestClient(create_app()) as http:
        saved = [message for message in read_history(http)["messages"] if message["type"] == "tool"]
        assert saved[0]["tool_status"] == tool.tool_status
        assert saved[0]["tool_artifact"] == expected_artifact
        assert read_history(http)["pending_interrupts"] == []
