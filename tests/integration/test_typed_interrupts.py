"""Keep typed interrupt questions and answers across HTTP and saver restart."""

import sys
from types import ModuleType
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field, SecretStr

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.core.memory.types import MemoryBackends
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper import constants
from langgraph_agent_toolkit.helper.types import EnvironmentMode
from langgraph_agent_toolkit.schema import ChatMessage
from langgraph_agent_toolkit.service.handler import create_app


CREDENTIAL = "typed-interrupt-test-credential"
HEADERS = {"Authorization": f"Bearer {CREDENTIAL}"}


class Decision(BaseModel):
    quantity: int = Field(ge=1, le=5, strict=True)


def configure_service(monkeypatch, tmp_path, graph):
    module = ModuleType("lat_typed_interrupt_" + uuid4().hex)
    module.agent = Agent("typed-interrupt", "A typed interrupt test agent.", graph)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for key, value in {
        "ENV_MODE": EnvironmentMode.PRODUCTION,
        "AUTH_MODE": "trusted",
        "AUTH_SECRET": SecretStr(CREDENTIAL),
        "AUTH_USERS": {},
        "MEMORY_BACKEND": MemoryBackends.SQLITE,
        "SQLITE_DB_PATH": str(tmp_path / "typed-interrupt.sqlite"),
        "OBSERVABILITY_BACKEND": ObservabilityBackend.EMPTY,
        "AGENT_PATHS": [f"{module.__name__}:agent"],
        "DEFAULT_AGENT": "typed-interrupt",
        "MCP_SERVERS": {},
        "MCP_AGENT_SERVERS": {},
    }.items():
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(constants, "_runtime_default_agent", "typed-interrupt")


def typed_graph(prepared, completed, count):
    def prepare(state):
        prepared.append("prepared")
        return {}

    builder = StateGraph(MessagesState)
    builder.add_node("prepare", prepare)
    builder.add_edge(START, "prepare")
    for index in range(count):
        label = str(index)

        def choose(state, label=label):
            answer = interrupt(
                {"label": label, "question": f"Choose a quantity for {label}."}, response_schema=Decision
            )
            completed.append((label, answer.quantity))
            return {"messages": [AIMessage(f"Quantity for {label}: {answer.quantity}.")]}

        builder.add_node(label, choose)
        builder.add_edge("prepare", label)
        builder.add_edge(label, END)
    return builder.compile()


def history(http):
    response = http.get("/history", headers=HEADERS, params={"thread_id": "shared"})
    assert response.status_code == 200, response.text
    return response.json()


def invoke(http, value):
    return http.post("/invoke", headers=HEADERS, json={"thread_id": "shared", "input": value})


@pytest.mark.parametrize("protocol", ["invoke", "stream", "stream_jsonl"])
@pytest.mark.parametrize("count", [1, 2])
def test_typed_questions_survive_http_and_restart_without_replaying_completed_nodes(
    monkeypatch, tmp_path, protocol, count
):
    prepared, completed = [], []
    configure_service(monkeypatch, tmp_path, typed_graph(prepared, completed, count))
    with TestClient(create_app()) as http:
        with AgentClient(agent="typed-interrupt", get_info=False, http_client=http, auth_secret=CREDENTIAL) as client:
            if protocol == "invoke":
                message = client.invoke({"message": "Begin."}, thread_id="shared")
            else:
                events = list(getattr(client, protocol)({"message": "Begin."}, thread_id="shared"))
                message = next(
                    event for event in events if isinstance(event, ChatMessage) and event.custom_data.get("interrupts")
                )
        pending = message.custom_data["interrupts"]
        assert len(pending) == count
        assert all(item["response_schema"] == Decision.model_json_schema() for item in pending)
        assert {item["id"]: item for item in history(http)["pending_interrupts"]} == {
            item["id"]: item for item in pending
        }
        assert prepared == ["prepared"]
        assert completed == []

    with TestClient(create_app()) as http:
        recovered = history(http)["pending_interrupts"]
        assert {item["id"]: item for item in recovered} == {item["id"]: item for item in pending}
        assert prepared == ["prepared"]
        assert completed == []
        answers = {item["id"]: {"quantity": index + 1} for index, item in enumerate(recovered)}
        response = invoke(http, {"resume": answers})
        assert response.status_code == 200, response.text
        assert prepared == ["prepared"]
        assert len(completed) == count
        assert set(completed) == {(item["value"]["label"], answers[item["id"]]["quantity"]) for item in recovered}
        page = history(http)
        assert page["pending_interrupts"] == []
        assert len([message for message in page["messages"] if message["type"] == "ai"]) == count


def test_invalid_typed_answer_keeps_the_question_for_a_later_valid_answer(monkeypatch, tmp_path):
    prepared, completed = [], []
    configure_service(monkeypatch, tmp_path, typed_graph(prepared, completed, 1))
    with TestClient(create_app()) as http:
        response = invoke(http, {"message": "Begin."})
        assert response.status_code == 200, response.text
        pending = history(http)["pending_interrupts"]
        interrupt_id = pending[0]["id"]

        missing = invoke(http, {"resume": {}})
        assert missing.status_code == 422, missing.text
        assert history(http)["pending_interrupts"] == pending

        invalid = invoke(http, {"resume": {interrupt_id: {"quantity": 0}}})
        assert invalid.status_code == 400, invalid.text
        assert invalid.json()["detail"] == "Invalid request"
        assert history(http)["pending_interrupts"] == pending
        assert prepared == ["prepared"]
        assert completed == []

    with TestClient(create_app()) as http:
        assert history(http)["pending_interrupts"] == pending
        valid = invoke(http, {"resume": {interrupt_id: {"quantity": 3}}})
        assert valid.status_code == 200, valid.text
        assert valid.json()["content"] == "Quantity for 0: 3."
        assert prepared == ["prepared"]
        assert completed == [("0", 3)]
        assert history(http)["pending_interrupts"] == []


def test_raw_interrupt_inputs_keep_the_existing_contract(monkeypatch, tmp_path):
    received = []

    def choose(state):
        answer = interrupt("Choose the next action.")
        received.append(answer)
        return {"messages": [AIMessage(answer["message"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("choose", choose)
    builder.add_edge(START, "choose")
    builder.add_edge("choose", END)
    configure_service(monkeypatch, tmp_path, builder.compile())
    with TestClient(create_app()) as http:
        paused = invoke(http, {"message": "Begin."})
        assert paused.status_code == 200, paused.text
        assert set(paused.json()["custom_data"]["interrupts"][0]) == {"id", "value"}
        payload = {"message": "Continue.", "resume": "application-owned-value"}
        resumed = invoke(http, payload)
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["content"] == "Continue."
        assert received == [payload]
        assert history(http)["pending_interrupts"] == []
