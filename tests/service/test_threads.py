"""Check thread discovery and bounded history through the service."""

import re
import sys
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.func import entrypoint
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import SecretStr

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.components.checkpoint.empty import NoOpSaver
from langgraph_agent_toolkit.core.memory.types import MemoryBackends
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper import constants
from langgraph_agent_toolkit.helper.types import EnvironmentMode
from langgraph_agent_toolkit.service.auth import storage_thread_id
from langgraph_agent_toolkit.service.handler import create_app


SHARED_HEADERS = {"Authorization": "Bearer thread-shared-credential"}
OWNER_HEADERS = {"Authorization": "Bearer thread-owner-credential"}
OTHER_HEADERS = {"Authorization": "Bearer thread-other-credential"}


def conversation_graph(kind="stategraph"):
    if kind == "functional":

        @entrypoint()
        async def workflow(inputs, *, previous=None):
            messages = (previous or {}).get("messages", []) + inputs["messages"]
            response = AIMessage("reply to " + messages[-1].content)
            return entrypoint.final(value={"messages": [response]}, save={"messages": messages + [response]})

        return workflow

    child = None
    if kind == "subgraph":
        child_builder = StateGraph(MessagesState)
        child_builder.add_node("private_reply", lambda state: {"messages": [AIMessage("private child answer")]})
        child_builder.add_edge(START, "private_reply")
        child_builder.add_edge("private_reply", END)
        child = child_builder.compile(checkpointer=True)

    async def reply(state):
        if child is not None:
            await child.ainvoke({"messages": [HumanMessage("private child question")]})
        return {"messages": [AIMessage("reply to " + state["messages"][-1].content)]}

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile()


@pytest.fixture
def thread_service(request, monkeypatch, tmp_path, mock_env):
    kind = getattr(request, "param", "stategraph")
    module = ModuleType("lat_thread_agents_" + uuid4().hex)
    module.first = Agent("first", "First thread test agent.", conversation_graph(kind))
    module.second = Agent("second", "Second thread test agent.", conversation_graph(kind))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    database = tmp_path / "threads.sqlite"
    for key, value in {
        "ENV_MODE": EnvironmentMode.PRODUCTION,
        "AUTH_MODE": "trusted",
        "AUTH_SECRET": SecretStr("thread-shared-credential"),
        "AUTH_SERVICE_USER_ID": "service",
        "AUTH_USERS": {"owner": SecretStr("thread-owner-credential"), "other": SecretStr("thread-other-credential")},
        "MEMORY_BACKEND": MemoryBackends.SQLITE,
        "SQLITE_DB_PATH": str(database),
        "OBSERVABILITY_BACKEND": ObservabilityBackend.EMPTY,
        "AGENT_PATHS": [f"{module.__name__}:first", f"{module.__name__}:second"],
        "DEFAULT_AGENT": "first",
        "HISTORY_MAX_PAGE_SIZE": 1000,
    }.items():
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(constants, "_runtime_default_agent", "first")
    return SimpleNamespace(app=create_app(), module=module, database=database, kind=kind)


def invoke(client, thread_id, *, agent="first", headers=SHARED_HEADERS, message="question", **identity):
    response = client.post(
        f"/{agent}/invoke",
        headers=headers,
        json={"thread_id": thread_id, "input": {"message": message}, **identity},
    )
    assert response.status_code == 200, response.text
    return response.json()


def add_history(client, thread_id, messages, *, agent="first", headers=SHARED_HEADERS, **identity):
    response = client.post(
        f"/{agent}/history/add_messages",
        headers=headers,
        json={"thread_id": thread_id, "messages": messages, **identity},
    )
    assert response.status_code == 201, response.text


def list_threads(client, *, agent=None, headers=SHARED_HEADERS, **params):
    response = client.get(f"/{agent}/threads" if agent else "/threads", headers=headers, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def public_ids(result):
    return [thread["thread_id"] for thread in result["threads"]]


def test_threads_are_scoped_to_the_effective_owner_and_agent(thread_service):
    with TestClient(thread_service.app) as client:
        invoke(client, "service-thread")
        invoke(client, "owner-thread", user_id="owner")
        invoke(client, "other-thread", user_id="other")
        invoke(client, "second-agent-thread", user_id="owner", agent="second")
        assert public_ids(list_threads(client)) == ["service-thread"]
        assert public_ids(list_threads(client, user_id="owner")) == ["owner-thread"]
        assert public_ids(list_threads(client, user_id="other")) == ["other-thread"]
        assert public_ids(list_threads(client, agent="second", user_id="owner")) == ["second-agent-thread"]
        assert public_ids(list_threads(client, headers=OWNER_HEADERS)) == ["owner-thread"]
        assert public_ids(list_threads(client, headers=OTHER_HEADERS)) == ["other-thread"]

        for headers, content in ((OWNER_HEADERS, "owner data"), (OTHER_HEADERS, "other data")):
            invoke(client, "same-public-id", headers=headers, message=content)
            assert set(public_ids(list_threads(client, headers=headers))) == {
                "same-public-id",
                "owner-thread" if headers == OWNER_HEADERS else "other-thread",
            }
            response = client.get("/history", headers=headers, params={"thread_id": "same-public-id"})
            assert response.status_code == 200, response.text
            assert [message["content"] for message in response.json()["messages"]] == [content, "reply to " + content]


def test_threads_reject_unauthorized_queries_before_database_reads(thread_service, monkeypatch):
    with TestClient(thread_service.app) as client:
        saver = client.app.state.agent_executor.get_agent("first").graph.checkpointer
        statements = []
        client.portal.call(saver.conn.set_trace_callback, statements.append)

        async def unexpected_read(*args, **kwargs):
            pytest.fail("Unauthorized thread queries must not read checkpoints.")

        monkeypatch.setattr(saver, "aget_tuple", unexpected_read)
        for headers, params, expected in (
            ({}, {}, 401),
            ({"Authorization": "Bearer wrong-credential"}, {}, 401),
            (OWNER_HEADERS, {"user_id": "other"}, 403),
            (OTHER_HEADERS, {"user_id": "owner"}, 403),
        ):
            response = client.get("/threads", headers=headers, params=params)
            assert response.status_code == expected, response.text
        response = client.get("/missing-agent/threads", headers=OWNER_HEADERS)
        assert response.status_code == 404, response.text
        assert not any("checkpoints" in statement.lower() for statement in statements)


def test_nul_identifiers_are_rejected_before_checkpoint_reads_or_writes(thread_service):
    with TestClient(thread_service.app) as client:
        saver = client.app.state.agent_executor.get_agent("first").graph.checkpointer
        statements = []
        client.portal.call(saver.conn.set_trace_callback, statements.append)
        for identity in ({"thread_id": "invalid\x00thread"}, {"user_id": "invalid\x00owner"}):
            for path, payload in (
                ("/invoke", {"input": {"message": "question"}}),
                ("/history/add_messages", {"messages": [{"type": "human", "content": "imported message"}]}),
            ):
                response = client.post(
                    path, headers=SHARED_HEADERS, json={"thread_id": "valid-thread", **payload, **identity}
                )
                assert response.status_code == 422, response.text
        response = client.get("/threads", headers=SHARED_HEADERS, params={"user_id": "invalid\x00owner"})
        assert response.status_code == 422, response.text
        assert not any("checkpoints" in statement.lower() for statement in statements)


def test_strict_shared_token_lists_its_service_owner_and_rejects_user_override(thread_service, monkeypatch):
    monkeypatch.setattr(settings, "AUTH_MODE", "token")
    with TestClient(thread_service.app) as client:
        invoke(client, "service-thread")
        assert public_ids(list_threads(client)) == ["service-thread"]
        response = client.get("/threads", headers=SHARED_HEADERS, params={"user_id": "other"})
        assert response.status_code == 403, response.text


@pytest.mark.parametrize("thread_service", ["stategraph", "functional", "subgraph"], indirect=True)
def test_threads_include_single_turn_and_imported_history_after_restart_and_clear(thread_service):
    with TestClient(thread_service.app) as client:
        assert public_ids(list_threads(client)) == []
        invoke(client, "single-turn")
        add_history(client, "imported-only", [{"type": "human", "content": "imported message"}])
        result = list_threads(client)
        assert set(public_ids(result)) == {"single-turn", "imported-only"}
        assert len(result["threads"]) == 2
        assert "private child" not in str(result)
        assert "lat:v1:" not in str(result)

    with TestClient(create_app()) as client:
        assert set(public_ids(list_threads(client))) == {"single-turn", "imported-only"}
        response = client.get("/history", headers=SHARED_HEADERS, params={"thread_id": "single-turn"})
        assert response.status_code == 200, response.text
        assert [message["content"] for message in response.json()["messages"]] == ["question", "reply to question"]
        response = client.request("DELETE", "/history/clear", headers=SHARED_HEADERS, json={"thread_id": "single-turn"})
        assert response.status_code == 200, response.text
        assert public_ids(list_threads(client)) == ["imported-only"]
        invoke(client, "single-turn", message="new run")
        assert set(public_ids(list_threads(client))) == {"single-turn", "imported-only"}


def test_stateless_agent_has_no_discoverable_threads(thread_service):
    thread_service.module.first.graph.checkpointer = NoOpSaver()
    with TestClient(thread_service.app) as client:
        invoke(client, "stateless")
        add_history(client, "imported", [{"type": "human", "content": "stateless import"}])
        assert public_ids(list_threads(client)) == []


def test_memory_threads_support_owner_isolation_and_clear(thread_service, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_BACKEND", None)
    with TestClient(thread_service.app) as client:
        invoke(client, "memory", headers=OWNER_HEADERS)
        invoke(client, "other-memory", headers=OTHER_HEADERS)
        assert public_ids(list_threads(client, headers=OWNER_HEADERS)) == ["memory"]
        assert public_ids(list_threads(client, headers=OTHER_HEADERS)) == ["other-memory"]
        response = client.request("DELETE", "/history/clear", headers=OWNER_HEADERS, json={"thread_id": "memory"})
        assert response.status_code == 200, response.text
        assert public_ids(list_threads(client, headers=OWNER_HEADERS)) == []


def test_custom_saver_without_thread_listing_returns_501(thread_service):
    thread_service.module.first.graph.checkpointer = BaseCheckpointSaver()
    with TestClient(thread_service.app) as client:
        response = client.get("/threads", headers=SHARED_HEADERS)
        assert response.status_code == 501, response.text


def test_thread_pages_default_to_100_and_preserve_creation_order(thread_service):
    with TestClient(thread_service.app) as client:
        for number in range(103):
            add_history(client, f"thread-{number}", [{"type": "human", "content": "saved message"}])
        first = list_threads(client)
        assert public_ids(first) == [f"thread-{number}" for number in range(102, 2, -1)]
        assert isinstance(first["next_cursor"], str)
        assert all(set(thread) == {"thread_id", "agent_id"} for thread in first["threads"])
        assert all(thread["agent_id"] == "first" for thread in first["threads"])

        invoke(client, "thread-0", message="new message on an old thread")
        invoke(client, "new-thread")
        second = list_threads(client, cursor=first["next_cursor"])
        assert public_ids(second) == ["thread-2", "thread-1", "thread-0"]
        assert second["next_cursor"] is None
        assert public_ids(list_threads(client, limit=2)) == ["new-thread", "thread-102"]


def test_cursor_remains_valid_after_its_thread_is_deleted_and_the_service_restarts(thread_service):
    with TestClient(thread_service.app) as client:
        for thread_id in ("oldest", "middle", "newest"):
            invoke(client, thread_id)
        page = list_threads(client, limit=1)
        assert public_ids(page) == ["newest"]
        assert page["next_cursor"]
        deleted = client.request("DELETE", "/history/clear", headers=SHARED_HEADERS, json={"thread_id": "newest"})
        assert deleted.status_code == 200, deleted.text

    with TestClient(create_app()) as client:
        page = list_threads(client, cursor=page["next_cursor"])
        assert public_ids(page) == ["middle", "oldest"]
        assert page["next_cursor"] is None


@pytest.mark.parametrize("thread_id", ["conversation/é/💬", "x" * 256])
def test_thread_discovery_preserves_public_identifiers(thread_service, thread_id):
    with TestClient(thread_service.app) as client:
        invoke(client, thread_id, user_id="utilisateur-é")
        assert public_ids(list_threads(client, user_id="utilisateur-é")) == [thread_id]


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 101},
        {"limit": "invalid"},
        {"user_id": ""},
        {"user_id": "x" * 257},
        {"cursor": "not-a-thread-cursor"},
        {"cursor": "x" * 2049},
    ],
)
def test_invalid_thread_page_is_rejected(thread_service, params):
    with TestClient(thread_service.app) as client:
        response = client.get("/threads", headers=SHARED_HEADERS, params=params)
        assert response.status_code == 422, response.text


def test_thread_cursor_cannot_cross_owner_or_agent(thread_service):
    with TestClient(thread_service.app) as client:
        invoke(client, "older", headers=OWNER_HEADERS)
        invoke(client, "newer", headers=OWNER_HEADERS)
        cursor = list_threads(client, headers=OWNER_HEADERS, limit=1)["next_cursor"]
        assert cursor
        for path, headers in (("/threads", OTHER_HEADERS), ("/second/threads", OWNER_HEADERS)):
            response = client.get(path, headers=headers, params={"cursor": cursor})
            assert response.status_code == 422, response.text


def test_legacy_threads_use_bounded_empty_pages_without_losing_later_matches(thread_service):
    with TestClient(thread_service.app) as client:
        invoke(client, "older-visible", headers=OWNER_HEADERS)
        graph = client.app.state.agent_executor.get_agent("first").graph

        async def seed_legacy_threads():
            for number in range(101):
                public_id = f"other-agent-{number}"
                await graph.ainvoke(
                    {"messages": [HumanMessage("legacy message")]},
                    {
                        "configurable": {
                            "thread_id": storage_thread_id("owner", "second", public_id),
                            "user_id": "owner",
                        },
                        "metadata": {"langfuse_session_id": public_id, "langfuse_user_id": "owner"},
                    },
                )

        client.portal.call(seed_legacy_threads)
        first = list_threads(client, headers=OWNER_HEADERS, limit=1)
        assert first["threads"] == []
        assert first["next_cursor"]
        second = list_threads(client, headers=OWNER_HEADERS, limit=1, cursor=first["next_cursor"])
        assert public_ids(second) == ["older-visible"]
        assert second["next_cursor"] is None
        assert public_ids(list_threads(client, headers=OWNER_HEADERS, agent="second", limit=1)) == ["other-agent-100"]


@pytest.mark.parametrize("field", ["lat_user_id", "lat_agent_id", "lat_thread_id"])
def test_request_cannot_override_thread_discovery_identity(thread_service, field):
    with TestClient(thread_service.app) as client:
        response = client.post(
            "/invoke",
            headers=OWNER_HEADERS,
            json={"thread_id": "forged", "input": {"message": "question"}, "agent_config": {field: "other"}},
        )
        assert response.status_code == 422, response.text
        assert public_ids(list_threads(client, headers=OWNER_HEADERS)) == []


@pytest.mark.parametrize("thread_service", ["stategraph", "functional", "subgraph"], indirect=True)
def test_thread_listing_reads_bounded_metadata_without_checkpoint_payloads(thread_service, monkeypatch):
    with TestClient(thread_service.app) as client:
        for number in range(30):
            invoke(client, "many-turns", message=f"question {number}")
        invoke(client, "single-turn")
        saver = client.app.state.agent_executor.get_agent("first").graph.checkpointer

        async def unexpected_snapshot(*args, **kwargs):
            pytest.fail("Thread listing must not load checkpoint payloads.")

        async def unexpected_scan(*args, **kwargs):
            pytest.fail("Thread listing must not scan checkpoint payloads.")
            yield

        monkeypatch.setattr(saver, "aget_tuple", unexpected_snapshot)
        monkeypatch.setattr(saver, "alist", unexpected_scan)
        statements = []
        client.portal.call(saver.conn.set_trace_callback, statements.append)
        assert public_ids(list_threads(client)) == ["single-turn", "many-turns"]
        queries = [sql.lower() for sql in statements if "from checkpoints" in sql.lower()]
        assert queries
        for query in queries:
            projection = query.split("from checkpoints", 1)[0]
            assert not re.search(r"\bcheckpoint\b|\*", projection)
            bound = re.search(r"\blimit\s+(\d+)", query)
            assert bound is not None, query
            assert 0 < int(bound.group(1)) <= 401


@pytest.mark.parametrize("thread_service", ["stategraph", "functional", "subgraph"], indirect=True)
def test_history_returns_100_messages_by_default_and_pages_without_loss(thread_service):
    messages = [{"type": "human", "content": f"saved message {number}"} for number in range(205)]
    with TestClient(thread_service.app) as client:
        add_history(client, "long-history", messages)
        collected = []
        for offset, expected_count, next_offset in ((0, 100, 100), (100, 100, 200), (200, 5, None)):
            response = client.get(
                "/history", headers=SHARED_HEADERS, params={"thread_id": "long-history", "offset": offset}
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert len(body["messages"]) == expected_count
            assert body["total"] == 205
            assert body["next_offset"] == next_offset
            collected.extend(message["content"] for message in body["messages"])
        assert collected == [message["content"] for message in messages]

        response = client.get(
            "/first/history", headers=SHARED_HEADERS, params={"thread_id": "long-history", "limit": 200}
        )
        assert response.status_code == 200, response.text
        assert len(response.json()["messages"]) == 200
        assert response.json()["next_offset"] == 200
