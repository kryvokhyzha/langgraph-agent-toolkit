"""Check saved Functional API history through the real service and SQLite."""

import asyncio
import json
import sys
from copy import copy
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.func import entrypoint
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import SecretStr

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import add_graph_history, get_graph_history
from langgraph_agent_toolkit.core.memory.types import MemoryBackends
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper import constants
from langgraph_agent_toolkit.helper.exceptions import InputValidationError
from langgraph_agent_toolkit.helper.types import EnvironmentMode
from langgraph_agent_toolkit.service.auth import storage_thread_id
from langgraph_agent_toolkit.service.handler import create_app


HEADERS = {"Authorization": "Bearer history-test-credential"}


async def reply(messages):
    count = sum(isinstance(message, HumanMessage) for message in messages)
    return await FakeListChatModel(responses=[f"answer {count}"]).ainvoke(messages)


@pytest.fixture(params=["functional", "stategraph"])
def history_service(request, monkeypatch, tmp_path):
    if request.param == "functional":

        @entrypoint()
        async def workflow(inputs, *, previous=None):
            saved = previous or {"messages": [], "retained_field": "keep me"}
            messages = saved["messages"] + inputs["messages"]
            response = await reply(messages)
            return entrypoint.final(
                value={"messages": [response]},
                save={**saved, "messages": messages + [response]},
            )

        graph = workflow
    else:

        async def node(state):
            return {"messages": [await reply(state["messages"])]}

        builder = StateGraph(MessagesState)
        builder.add_node("reply", node)
        builder.add_edge(START, "reply")
        builder.add_edge("reply", END)
        graph = builder.compile()
    module = ModuleType("lat_history_agent_" + uuid4().hex)
    module.agent = Agent("history-agent", "A history contract agent.", graph)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    database = tmp_path / "history.sqlite"
    for key, value in {
        "ENV_MODE": EnvironmentMode.PRODUCTION,
        "AUTH_MODE": "token",
        "AUTH_SECRET": None,
        "AUTH_USERS": {"owner": SecretStr("history-test-credential")},
        "MEMORY_BACKEND": MemoryBackends.SQLITE,
        "SQLITE_DB_PATH": str(database),
        "OBSERVABILITY_BACKEND": ObservabilityBackend.EMPTY,
        "AGENT_PATHS": [f"{module.__name__}:agent"],
        "DEFAULT_AGENT": "history-agent",
    }.items():
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(constants, "_runtime_default_agent", "history-agent")
    return SimpleNamespace(graph=graph, database=database, app=create_app(), kind=request.param)


def read_history(client):
    response = client.get("/history", params={"thread_id": "conversation"}, headers=HEADERS)
    assert response.status_code == 200, response.text
    return [message["content"] for message in response.json()["messages"]]


def test_existing_checkpoint_append_stream_and_clear_keep_the_history_contract(history_service):
    """Read old checkpoints without rewriting their output or replaying answers."""
    config = {"configurable": {"thread_id": storage_thread_id("owner", "history-agent", "conversation")}}

    async def seed_existing_checkpoint():
        async with AsyncSqliteSaver.from_conn_string(str(history_service.database)) as saver:
            graph = copy(history_service.graph)
            graph.checkpointer = saver
            for text in ("first", "second"):
                await graph.ainvoke({"messages": [HumanMessage(text)]}, config)
            if history_service.kind == "functional":
                state = await graph.aget_state(config)
                assert [message.content for message in state.values["messages"]] == ["answer 2"]

    asyncio.run(seed_existing_checkpoint())

    with TestClient(history_service.app) as client:
        assert read_history(client) == ["first", "answer 1", "second", "answer 2"]
        added = client.post(
            "/history/add_messages",
            headers=HEADERS,
            json={
                "thread_id": "conversation",
                "messages": [
                    {"type": "human", "content": "imported question"},
                    {"type": "ai", "content": "imported answer"},
                ],
            },
        )
        assert added.status_code == 201, added.text
        assert read_history(client) == [
            "first",
            "answer 1",
            "second",
            "answer 2",
            "imported question",
            "imported answer",
        ]
        response = client.post(
            "/stream", headers=HEADERS, json={"thread_id": "conversation", "input": {"message": "next"}}
        )
        assert response.status_code == 200, response.text
        lines = [line for line in response.text.splitlines() if line]
        assert lines[-1] == "data: [DONE]"
        events = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
        assert not any(event["type"] == "error" for event in events)
        assert [event["content"]["content"] for event in events if event["type"] == "message"] == ["answer 4"]
        assert "".join(event["content"] for event in events if event["type"] == "token") == "answer 4"
        assert read_history(client) == [
            "first",
            "answer 1",
            "second",
            "answer 2",
            "imported question",
            "imported answer",
            "next",
            "answer 4",
        ]

    with TestClient(history_service.app) as client:
        assert len(read_history(client)) == 8
        cleared = client.request("DELETE", "/history/clear", headers=HEADERS, json={"thread_id": "conversation"})
        assert cleared.status_code == 200, cleared.text
        assert read_history(client) == []
        response = client.post(
            "/invoke", headers=HEADERS, json={"thread_id": "conversation", "input": {"message": "fresh"}}
        )
        assert response.status_code == 200, response.text
        assert response.json()["content"] == "answer 1"
        assert read_history(client) == ["fresh", "answer 1"]


async def test_functional_append_preserves_other_saved_fields_and_message_ids(tmp_path):
    @entrypoint()
    async def workflow(inputs, *, previous=None):
        return entrypoint.final(value={"messages": []}, save={"messages": inputs["messages"], "other": "keep"})

    config = {"configurable": {"thread_id": "saved-fields"}}
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "saved.sqlite")) as saver:
        workflow.checkpointer = saver
        await workflow.ainvoke({"messages": [HumanMessage("original", id="message-1")]}, config)
        await add_graph_history(workflow, config, [HumanMessage("edited", id="message-1"), HumanMessage("new")])
        assert [message.content for message in await get_graph_history(workflow, config)] == ["edited", "new"]
        checkpoint = await saver.aget_tuple(config)
        assert checkpoint.checkpoint["channel_values"]["__previous__"]["other"] == "keep"


@pytest.mark.parametrize("saved", ["not an object", {"messages": "not a list"}])
async def test_invalid_functional_history_is_rejected_without_replacing_the_checkpoint(tmp_path, saved):
    @entrypoint()
    async def workflow(inputs):
        return entrypoint.final(value={"messages": []}, save=saved)

    config = {"configurable": {"thread_id": "invalid"}}
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "invalid.sqlite")) as saver:
        workflow.checkpointer = saver
        await workflow.ainvoke({}, config)
        before = await saver.aget_tuple(config)
        with pytest.raises(InputValidationError, match="messages list"):
            await get_graph_history(workflow, config)
        with pytest.raises(InputValidationError, match="messages list"):
            await add_graph_history(workflow, config, [HumanMessage("new")])
        after = await saver.aget_tuple(config)
        assert after.config == before.config
        assert after.checkpoint == before.checkpoint


async def test_append_preserves_history_without_a_messages_reducer(tmp_path):
    from typing import TypedDict

    class State(TypedDict):
        messages: list
        retained_field: str

    builder = StateGraph(State)
    builder.add_node("reply", lambda state: state)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    config = {"configurable": {"thread_id": "replace-channel"}}
    database = str(tmp_path / "replace-channel.sqlite")
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        graph = builder.compile(checkpointer=saver)
        await graph.ainvoke(
            {
                "messages": [HumanMessage("original", id="old"), HumanMessage("keep", id="keep")],
                "retained_field": "keep",
            },
            config,
        )
        await add_graph_history(graph, config, [HumanMessage("edited", id="old"), HumanMessage("new", id="new")])

    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        graph = builder.compile(checkpointer=saver)
        assert [message.content for message in await get_graph_history(graph, config)] == ["edited", "keep", "new"]
        state = await graph.aget_state(config)
        assert state.values["retained_field"] == "keep"
