"""Check mid-run interrupts through HTTP with real checkpoint storage."""

import json
import sys

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.types import EnvironmentMode


@pytest.fixture(params=["memory", "sqlite"])
async def saver(request, tmp_path):
    """Keep the same saver available for the question and its answer."""
    if request.param == "memory":
        yield InMemorySaver()
    else:
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "interrupt.sqlite")) as checkpointer:
            yield checkpointer


async def submit(client, protocol, message):
    response = await client.post(
        f"/mid-run/{protocol}",
        json={"thread_id": "conversation", "input": {"message": message}, "stream_tokens": False},
    )
    assert response.status_code == 200, response.text
    if protocol == "invoke":
        return [response.json()]
    lines = [line for line in response.text.splitlines() if line]
    if protocol == "stream":
        assert lines.pop() == "data: [DONE]"
        lines = [line.removeprefix("data: ") for line in lines]
    events = [json.loads(line) for line in lines]
    assert all(event["type"] == "message" for event in events), events
    return [event["content"] for event in events]


@pytest.mark.parametrize("protocol", ["invoke", "stream", "stream/jsonl"])
async def test_mid_run_question_takes_priority_and_resume_does_not_repeat_completed_nodes(
    app, monkeypatch, saver, protocol
):
    prepared = []
    answers = []

    async def prepare(state):
        prepared.append("prepared")
        return {"messages": [AIMessage("I will check.")]}

    async def choose(state):
        answer = interrupt("Which action should run?")
        answers.append(answer)
        return {"messages": [AIMessage(f"Completed: {answer['message']}.")]}

    builder = StateGraph(MessagesState)
    builder.add_node("prepare", prepare)
    builder.add_node("choose", choose)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "choose")
    builder.add_edge("choose", END)
    agent = Agent("mid-run", "A graph with a saved question.", builder.compile(checkpointer=saver))
    monkeypatch.setattr(sys.modules[__name__], "mid_run_agent", agent, raising=False)
    monkeypatch.setattr("langgraph_agent_toolkit.helper.constants._runtime_default_agent", agent.name)
    monkeypatch.setattr(settings, "ENV_MODE", EnvironmentMode.DEVELOPMENT)
    monkeypatch.setattr(settings, "AUTH_SECRET", None)
    monkeypatch.setattr(settings, "AUTH_USERS", {})
    monkeypatch.setattr(settings, "CHECK_INTERRUPTS", True)
    app.state.agent_executor = AgentExecutor(f"{__name__}:mid_run_agent")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        messages = await submit(client, protocol, "Begin.")
        expected = ["Which action should run?"]
        if protocol != "invoke":
            expected.insert(0, "I will check.")
        assert [message["content"] for message in messages] == expected
        assert all(message["type"] == "ai" for message in messages)
        assert all(message["run_id"] and message["thread_id"] == "conversation" for message in messages)
        pending = messages[-1]["custom_data"]["interrupts"]
        assert len(pending) == 1
        assert pending[0]["id"] and pending[0]["value"] == "Which action should run?"

        saved = await client.get("/mid-run/history", params={"thread_id": "conversation"})
        assert saved.status_code == 200, saved.text
        assert [message["content"] for message in saved.json()["messages"]] == ["Begin.", "I will check."]
        assert saved.json()["pending_interrupts"] == pending
        assert prepared == ["prepared"]
        assert answers == []

        resumed = await submit(client, protocol, "continue")
        assert [message["content"] for message in resumed] == ["Completed: continue."]
        assert not resumed[0]["custom_data"].get("interrupts")
        assert prepared == ["prepared"]
        assert answers == [{"message": "continue"}]
        saved = await client.get("/mid-run/history", params={"thread_id": "conversation"})
        assert saved.status_code == 200, saved.text
        assert [message["content"] for message in saved.json()["messages"]] == [
            "Begin.",
            "I will check.",
            "Completed: continue.",
        ]
        assert saved.json()["pending_interrupts"] == []
