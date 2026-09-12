"""Keep provider usage in public messages and saved history."""

import sys
from types import ModuleType
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import SecretStr, ValidationError

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.core.memory.types import MemoryBackends
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper import constants
from langgraph_agent_toolkit.helper.types import EnvironmentMode
from langgraph_agent_toolkit.helper.utils import langchain_to_chat_message, sanitize_chat_history
from langgraph_agent_toolkit.schema import ChatMessage, MessageInput
from langgraph_agent_toolkit.service.handler import create_app


USAGE = {
    "input_tokens": 8,
    "output_tokens": 3,
    "total_tokens": 11,
    "input_token_details": {"cache_read": 4},
    "output_token_details": {"reasoning": 1},
}


def test_conversion_keeps_usage_without_changing_provider_metadata():
    message = AIMessage("answer", usage_metadata=USAGE, response_metadata={"finish_reason": "stop"})
    converted = langchain_to_chat_message(message)
    restored = ChatMessage.model_validate_json(converted.model_dump_json())
    assert restored.usage_metadata == USAGE
    assert restored.response_metadata == {"finish_reason": "stop"}
    assert ChatMessage(type="ai", content="unknown usage").usage_metadata is None
    zero_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert langchain_to_chat_message(AIMessage("", usage_metadata=zero_usage)).usage_metadata == zero_usage


def test_sanitizing_an_interrupted_tool_call_keeps_its_billed_usage():
    message = AIMessage(
        "",
        tool_calls=[{"name": "add", "args": {"a": 2, "b": 5}, "id": "interrupted-call"}],
        usage_metadata=USAGE,
    )
    sanitized = sanitize_chat_history([message])
    assert sanitized[0].tool_calls == []
    assert sanitized[0].usage_metadata == USAGE


def test_history_import_requires_complete_usage_for_an_ai_message():
    with pytest.raises(ValidationError, match="usage_metadata"):
        MessageInput(type="ai", content="answer", usage_metadata={"input_tokens": 1})
    with pytest.raises(ValidationError, match="Only AI messages"):
        MessageInput(type="human", content="question", usage_metadata=USAGE)


@pytest.mark.parametrize("content", ["", [], "Visible provider text"])
def test_refusal_text_is_visible_without_changing_the_source(content):
    refusal = "I cannot help with that request."
    source = AIMessage(content, additional_kwargs={"refusal": refusal}, response_metadata={"finish_reason": "stop"})
    converted = langchain_to_chat_message(source)
    assert converted.content == (content or refusal)
    assert converted.response_metadata == {"finish_reason": "stop", "refusal": refusal}
    assert source.content == content
    assert source.response_metadata == {"finish_reason": "stop"}
    assert source.additional_kwargs == {"refusal": refusal}


def test_nontext_refusal_metadata_is_not_exposed_as_an_answer():
    source = AIMessage("", additional_kwargs={"refusal": {"unexpected": "payload"}})
    converted = langchain_to_chat_message(source)
    assert converted.content == ""
    assert converted.response_metadata == {}


@pytest.mark.parametrize("endpoint", ["invoke", "stream", "stream_jsonl"])
def test_usage_survives_http_client_history_import_and_sqlite_reopen(endpoint, monkeypatch, tmp_path):
    metadata = {"finish_reason": "stop", "model_name": "test-model"}
    model = FakeMessagesListChatModel(responses=[AIMessage("answer", usage_metadata=USAGE, response_metadata=metadata)])

    async def reply(state):
        return {"messages": [await model.ainvoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("reply", reply)
    graph.add_edge(START, "reply")
    graph.add_edge("reply", END)
    module = ModuleType("lat_usage_agent_" + uuid4().hex)
    module.agent = Agent("usage-agent", "A local agent with token counts.", graph.compile())
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for key, value in {
        "ENV_MODE": EnvironmentMode.PRODUCTION,
        "AUTH_MODE": "trusted",
        "AUTH_SECRET": SecretStr("usage-test-credential"),
        "AUTH_USERS": {},
        "MEMORY_BACKEND": MemoryBackends.SQLITE,
        "SQLITE_DB_PATH": str(tmp_path / "usage.sqlite"),
        "OBSERVABILITY_BACKEND": ObservabilityBackend.EMPTY,
        "AGENT_PATHS": [f"{module.__name__}:agent"],
        "DEFAULT_AGENT": "usage-agent",
    }.items():
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(constants, "_runtime_default_agent", "usage-agent")
    app = create_app()

    def connect(http):
        return AgentClient(
            "http://testserver", agent="usage-agent", auth_secret="usage-test-credential", http_client=http
        )

    with TestClient(app) as http, connect(http) as client:
        response = getattr(client, endpoint)({"message": "question"}, thread_id="usage-thread", user_id="user-1")
        if endpoint != "invoke":
            response = [event for event in response if isinstance(event, ChatMessage) and event.type == "ai"][-1]
        assert response.usage_metadata == USAGE
        assert response.response_metadata == metadata
        assert response.thread_id == "usage-thread"
        history = client.get_history("usage-thread", user_id="user-1").messages
        assert history[0].usage_metadata is None
        assert history[-1].usage_metadata == USAGE
        client.add_messages([response.model_dump()], thread_id="imported-thread", user_id="user-1")

    with TestClient(app) as http, connect(http) as client:
        for thread in ("usage-thread", "imported-thread"):
            saved = client.get_history(thread, user_id="user-1").messages[-1]
            assert saved.content == "answer"
            assert saved.usage_metadata == USAGE
            assert saved.response_metadata == metadata
