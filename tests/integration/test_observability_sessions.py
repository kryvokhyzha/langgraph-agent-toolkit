"""Keep public trace sessions separate from private checkpoint identifiers."""

import sys
from contextlib import contextmanager
from types import ModuleType
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import SecretStr

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.core.memory.concurrency import ConversationCoordinator
from langgraph_agent_toolkit.core.memory.types import MemoryBackends
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper import constants
from langgraph_agent_toolkit.helper.types import EnvironmentMode
from langgraph_agent_toolkit.schema import ChatMessage
from langgraph_agent_toolkit.service.handler import create_app


class CapturingObservability(EmptyObservability):
    def __init__(self):
        super().__init__()
        self.callbacks = []
        self.roots = []

    def get_callback_handler(self, **kwargs):
        self.callbacks.append(kwargs)
        return None

    @contextmanager
    def trace_context(self, run_id, **kwargs):
        self.roots.append(kwargs)
        yield None


def graph_with_capture(records):
    def reply(state, config):
        user_messages = [item for item in state["messages"] if isinstance(item, HumanMessage)]
        records.append(
            {
                "storage": config["configurable"]["thread_id"],
                "user": config["configurable"]["user_id"],
                "session": config["metadata"]["langfuse_session_id"],
            }
        )
        return {"messages": [AIMessage(f"Turn {len(user_messages)}: {user_messages[-1].content}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile()


@pytest.mark.parametrize("protocol", ["invoke", "stream", "stream_jsonl"])
def test_http_sessions_use_the_public_thread_and_checkpoints_keep_owner_and_agent_scope(
    monkeypatch, tmp_path, protocol
):
    observed = CapturingObservability()
    records = []
    module = ModuleType("lat_session_test_" + uuid4().hex)
    module.first = Agent("first", "First session test agent.", graph_with_capture(records), observed)
    module.second = Agent("second", "Second session test agent.", graph_with_capture(records), observed)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    credential = "session-test-client-credential"
    for key, value in {
        "ENV_MODE": EnvironmentMode.PRODUCTION,
        "AUTH_MODE": "trusted",
        "AUTH_SECRET": SecretStr(credential),
        "AUTH_USERS": {},
        "MEMORY_BACKEND": MemoryBackends.SQLITE,
        "SQLITE_DB_PATH": str(tmp_path / "sessions.sqlite"),
        "OBSERVABILITY_BACKEND": ObservabilityBackend.EMPTY,
        "AGENT_PATHS": [f"{module.__name__}:first", f"{module.__name__}:second"],
        "DEFAULT_AGENT": "first",
        "MCP_SERVERS": {},
        "MCP_AGENT_SERVERS": {},
    }.items():
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(constants, "_runtime_default_agent", "first")

    with TestClient(create_app()) as http:
        for agent in ("first", "second"):
            with AgentClient(agent=agent, get_info=False, http_client=http, auth_secret=credential) as client:
                for user in ("user-one", "user-two"):
                    for turn in (1, 2):
                        text = f"Private request for {agent}/{user}/{turn}."
                        arguments = {"thread_id": "shared-public-thread", "user_id": user}
                        if protocol == "invoke":
                            response = client.invoke({"message": text}, **arguments)
                        else:
                            events = list(getattr(client, protocol)({"message": text}, **arguments))
                            response = next(
                                item for item in reversed(events) if isinstance(item, ChatMessage) and item.type == "ai"
                            )
                        assert response.content == f"Turn {turn}: {text}"
                        assert response.thread_id == "shared-public-thread"

    assert len(records) == len(observed.callbacks) == len(observed.roots) == 8
    assert {item["session_id"] for item in observed.callbacks} == {"shared-public-thread"}
    assert {item["session_id"] for item in observed.roots} == {"shared-public-thread"}
    assert {item["user_id"] for item in observed.roots} == {"user-one", "user-two"}
    assert {item["session"] for item in records} == {"shared-public-thread"}
    storage_keys = {item["storage"] for item in records}
    assert len(storage_keys) == 4
    assert all(key.startswith("lat:v1:") for key in storage_keys)
    assert "shared-public-thread" not in storage_keys

    with TestClient(create_app()) as http:
        for agent in ("first", "second"):
            with AgentClient(agent=agent, get_info=False, http_client=http, auth_secret=credential) as client:
                for user in ("user-one", "user-two"):
                    history = client.get_history("shared-public-thread", user_id=user)
                    human = [item.content for item in history.messages if item.type == "human"]
                    assert human == [f"Private request for {agent}/{user}/{turn}." for turn in (1, 2)]
    assert len(records) == 8


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("public_session", [None, "public-session"])
async def test_embedded_executor_keeps_the_thread_default_and_accepts_an_explicit_trace_session(
    streaming, public_session
):
    records = []
    observed = CapturingObservability()
    executor = AgentExecutor.__new__(AgentExecutor)
    executor.agents = {"embedded": Agent("embedded", "Embedded session test.", graph_with_capture(records), observed)}
    executor.concurrency = ConversationCoordinator()
    arguments = dict(
        agent_id="embedded",
        input={"message": "Hello."},
        thread_id="private-checkpoint-thread",
        user_id="user-one",
        trace_session_id=public_session,
    )
    if streaming:
        events = [item async for item in executor.stream(**arguments)]
        assert any(isinstance(item, ChatMessage) and item.content == "Turn 1: Hello." for item in events)
    else:
        assert (await executor.invoke(**arguments)).content == "Turn 1: Hello."
    expected = public_session or "private-checkpoint-thread"
    assert records == [{"storage": "private-checkpoint-thread", "user": "user-one", "session": expected}]
    assert observed.callbacks[0]["session_id"] == observed.roots[0]["session_id"] == expected
