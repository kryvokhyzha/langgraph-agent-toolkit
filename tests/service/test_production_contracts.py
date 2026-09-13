"""Verify authentication, durable history, and startup with real local graphs."""

import asyncio
import json
import sqlite3
import sys
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from pydantic import SecretStr

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.components.checkpoint.empty import NoOpSaver
from langgraph_agent_toolkit.core._base_settings import Settings
from langgraph_agent_toolkit.core.memory.sqlite import SQLiteMemoryBackend
from langgraph_agent_toolkit.core.memory.types import MemoryBackends
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper import constants
from langgraph_agent_toolkit.helper.types import EnvironmentMode
from langgraph_agent_toolkit.service.auth import storage_thread_id
from langgraph_agent_toolkit.service.handler import create_app


ALICE = {"Authorization": "Bearer test-alice-token"}
BOB = {"Authorization": "Bearer test-bob-token"}


@pytest.fixture
def configured_service(monkeypatch, tmp_path, mock_env):
    calls = []
    defaults = Settings(_env_file=None)

    async def reply(state):
        message = state["messages"][-1].content
        calls.append(message)
        return {"messages": [AIMessage("reply to " + message)]}

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.set_entry_point("reply")
    builder.add_edge("reply", END)
    graph = builder.compile()
    module = ModuleType("lat_contract_agents_" + uuid4().hex)
    module.first = Agent("first", "First local agent", graph)
    module.second = Agent("second", "Second local agent", graph)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    database = tmp_path / "history.sqlite"
    for key, value in {
        "ENV_MODE": EnvironmentMode.PRODUCTION,
        "AUTH_MODE": defaults.AUTH_MODE,
        "AUTH_SECRET": None,
        "AUTH_USERS": {"alice": SecretStr("test-alice-token"), "bob": SecretStr("test-bob-token")},
        "MEMORY_BACKEND": MemoryBackends.SQLITE,
        "SQLITE_DB_PATH": str(database),
        "OBSERVABILITY_BACKEND": ObservabilityBackend.EMPTY,
        "AGENT_PATHS": [f"{module.__name__}:first", f"{module.__name__}:second"],
        "DEFAULT_AGENT": "first",
    }.items():
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(constants, "_runtime_default_agent", "first")
    return SimpleNamespace(app=create_app(), database=database, module=module, calls=calls)


@pytest.fixture
def live_service(configured_service):
    with TestClient(configured_service.app, raise_server_exceptions=False) as client:
        configured_service.client = client
        yield configured_service


def invoke(client, message, *, headers=ALICE, agent="first", thread_id="shared", **extra):
    payload = {"input": {"message": message}, "thread_id": thread_id, **extra}
    return client.post(f"/{agent}/invoke", headers=headers, json=payload)


def history(client, *, headers=ALICE, agent="first", thread_id="shared", **extra):
    return client.get(f"/{agent}/history", headers=headers, params={"thread_id": thread_id, **extra})


def contents(response):
    assert response.status_code == 200, response.text
    return [message["content"] for message in response.json()["messages"]]


def test_same_public_thread_is_private_to_each_token(live_service):
    client = live_service.client
    assert invoke(client, "Alice private").status_code == 200
    assert contents(history(client, headers=BOB)) == []
    assert invoke(client, "Bob private", headers=BOB).status_code == 200
    assert contents(history(client)) == ["Alice private", "reply to Alice private"]
    assert contents(history(client, headers=BOB)) == ["Bob private", "reply to Bob private"]

    added = client.post(
        "/first/history/add_messages",
        headers=BOB,
        json={"thread_id": "shared", "messages": [{"type": "human", "content": "Bob note"}]},
    )
    assert added.status_code == 201, added.text
    assert "Bob note" not in contents(history(client))
    deleted = client.request("DELETE", "/first/history/clear", headers=BOB, json={"thread_id": "shared"})
    assert deleted.status_code == 200, deleted.text
    assert contents(history(client, headers=BOB)) == []
    assert contents(history(client)) == ["Alice private", "reply to Alice private"]


def test_agent_namespaces_are_separate_with_shared_sqlite_saver(live_service):
    client = live_service.client
    assert invoke(client, "First agent private").status_code == 200
    assert contents(history(client, agent="second")) == []
    assert invoke(client, "Second agent private", agent="second").status_code == 200
    assert contents(history(client)) == ["First agent private", "reply to First agent private"]
    assert contents(history(client, agent="second")) == ["Second agent private", "reply to Second agent private"]


def test_stateless_agent_remains_valid_with_a_configured_service_backend(configured_service):
    configured_service.module.first.graph.checkpointer = NoOpSaver()
    with TestClient(configured_service.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/first/history/add_messages",
            headers=ALICE,
            json={"thread_id": "stateless", "messages": [{"type": "human", "content": "stateless input"}]},
        )
        assert response.status_code == 201, response.text
        assert contents(history(client, thread_id="stateless")) == []
        assert invoke(client, "stateless run", thread_id="stateless").status_code == 200
        assert contents(history(client, thread_id="stateless")) == []


@pytest.mark.parametrize("endpoint", ["invoke", "stream", "stream/jsonl"])
def test_user_id_spoof_is_rejected_before_execution_and_stream_headers(live_service, endpoint):
    response = live_service.client.post(
        f"/first/{endpoint}",
        headers=BOB,
        json={"input": {"message": "spoof"}, "thread_id": "shared", "user_id": "alice"},
    )
    assert response.status_code == 403, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert live_service.calls == []


@pytest.mark.parametrize("operation", ["read", "add", "clear"])
def test_user_id_spoof_cannot_read_change_or_delete_history(live_service, operation):
    client = live_service.client
    assert invoke(client, "Alice private").status_code == 200
    identity = {"thread_id": "shared", "user_id": "alice"}
    if operation == "read":
        response = client.get("/first/history", headers=BOB, params=identity)
    elif operation == "add":
        response = client.post(
            "/first/history/add_messages",
            headers=BOB,
            json={**identity, "messages": [{"type": "human", "content": "attack"}]},
        )
    else:
        response = client.request("DELETE", "/first/history/clear", headers=BOB, json=identity)
    assert response.status_code == 403, response.text
    assert contents(history(client)) == ["Alice private", "reply to Alice private"]


@pytest.mark.parametrize("endpoint", ["invoke", "stream", "stream/jsonl"])
@pytest.mark.parametrize(
    "reserved", ["thread_id", "user_id", "checkpoint_id", "checkpoint_ns", "__pregel_checkpointer"]
)
def test_reserved_configuration_is_rejected_before_execution(live_service, endpoint, reserved):
    response = live_service.client.post(
        f"/first/{endpoint}",
        headers=ALICE,
        json={"input": {"message": "bad config"}, "agent_config": {reserved: "forged"}},
    )
    assert response.status_code == 422, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert live_service.calls == []


@pytest.mark.parametrize("endpoint", ["invoke", "stream", "stream/jsonl"])
def test_generated_public_thread_is_returned_and_can_be_reused(live_service, endpoint):
    client = live_service.client
    response = client.post(f"/first/{endpoint}", headers=ALICE, json={"input": {"message": "new conversation"}})
    assert response.status_code == 200, response.text
    if endpoint == "invoke":
        message = response.json()
    else:
        lines = response.text.splitlines()
        events = [json.loads(line.removeprefix("data: ")) for line in lines if line and line != "data: [DONE]"]
        message = next(event["content"] for event in events if event["type"] == "message")
    public_id = message["thread_id"]
    assert str(UUID(public_id)) == public_id
    assert not public_id.startswith("lat:v1:")
    assert invoke(client, "follow-up", thread_id=public_id).status_code == 200
    assert contents(history(client, thread_id=public_id)) == [
        "new conversation",
        "reply to new conversation",
        "follow-up",
        "reply to follow-up",
    ]


def test_tool_messages_round_trip_and_clear_removes_all_checkpoints(live_service):
    client = live_service.client
    assert invoke(client, "question").status_code == 200
    response = client.post(
        "/first/history/add_messages",
        headers=ALICE,
        json={
            "thread_id": "shared",
            "messages": [
                {"type": "ai", "content": "", "tool_calls": [{"name": "lookup", "args": {"key": "a"}, "id": "call-a"}]},
                {"type": "tool", "content": "real tool result", "tool_call_id": "call-a"},
            ],
        },
    )
    assert response.status_code == 201, response.text
    messages = history(client).json()["messages"]
    assert messages[-1]["type"] == "tool"
    assert messages[-1]["tool_call_id"] == "call-a"
    assert messages[-1]["content"] == "real tool result"
    key = storage_thread_id("alice", "first", "shared")
    with sqlite3.connect(live_service.database) as connection:
        assert connection.execute("SELECT count(*) FROM checkpoints WHERE thread_id = ?", (key,)).fetchone()[0] > 1
        assert connection.execute("SELECT count(*) FROM writes WHERE thread_id = ?", (key,)).fetchone()[0] > 0
    cleared = client.request("DELETE", "/first/history/clear", headers=ALICE, json={"thread_id": "shared"})
    assert cleared.status_code == 200, cleared.text
    assert contents(history(client)) == []
    with sqlite3.connect(live_service.database) as connection:
        assert connection.execute("SELECT count(*) FROM checkpoints WHERE thread_id = ?", (key,)).fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM writes WHERE thread_id = ?", (key,)).fetchone()[0] == 0


def test_custom_and_multimodal_history_round_trip(live_service):
    client = live_service.client
    image_content = [
        {"type": "text", "text": "A local test image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,dGVzdA=="}},
    ]
    custom_data = {"task": "local-job", "state": "complete", "result": {"count": 3}}
    response = client.post(
        "/first/history/add_messages",
        headers=ALICE,
        json={
            "thread_id": "rich-history",
            "messages": [
                {"type": "human", "content": image_content},
                {"type": "custom", "content": "", "custom_data": custom_data},
            ],
        },
    )
    assert response.status_code == 201, response.text
    result = history(client, thread_id="rich-history")
    assert result.status_code == 200, result.text
    messages = result.json()["messages"]
    assert messages[0]["type"] == "human"
    assert messages[0]["content"] == image_content
    assert messages[1]["type"] == "custom"
    assert messages[1]["custom_data"] == custom_data


async def assert_connection_closed(connection):
    with pytest.raises(ValueError, match="no active connection"):
        await connection.execute("SELECT 1")


def test_repeated_lifespan_clones_imported_graphs_and_preserves_sqlite_history(configured_service):
    app = configured_service.app
    original = configured_service.module.first.graph
    with TestClient(app) as client:
        first_graph = app.state.agent_executor.get_agent("first").graph
        first_connection = first_graph.checkpointer.conn
        assert first_graph is not original
        assert original.checkpointer is None
        assert invoke(client, "before lifespan restart").status_code == 200
    asyncio.run(assert_connection_closed(first_connection))
    assert app.state.ready is False
    with TestClient(app) as client:
        second_graph = app.state.agent_executor.get_agent("first").graph
        second_connection = second_graph.checkpointer.conn
        assert second_graph is not first_graph
        assert second_connection is not first_connection
        assert invoke(client, "after lifespan restart").status_code == 200
        assert contents(history(client)) == [
            "before lifespan restart",
            "reply to before lifespan restart",
            "after lifespan restart",
            "reply to after lifespan restart",
        ]
    asyncio.run(assert_connection_closed(second_connection))
    assert original.checkpointer is None
    assert configured_service.module.first.observability is None


def test_required_agent_import_failure_aborts_startup_and_closes_database(configured_service, monkeypatch):
    connections = []
    original = SQLiteMemoryBackend.get_checkpoint_saver

    @asynccontextmanager
    async def track_saver(self):
        async with original(self) as saver:
            connections.append(saver.conn)
            yield saver

    monkeypatch.setattr(SQLiteMemoryBackend, "get_checkpoint_saver", track_saver)
    monkeypatch.setattr(settings, "AGENT_PATHS", [*settings.AGENT_PATHS, "lat_missing_required_agent:missing"])
    with pytest.raises(ValueError, match="Required agents failed to load"):
        with TestClient(configured_service.app):
            pass
    assert connections
    asyncio.run(assert_connection_closed(connections[0]))
    assert configured_service.app.state.ready is False
    assert not hasattr(configured_service.app.state, "agent_executor")


def test_database_open_failure_aborts_startup(configured_service, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "missing-directory" / "history.sqlite"))
    with pytest.raises(sqlite3.OperationalError):
        with TestClient(configured_service.app):
            pass
    assert configured_service.app.state.ready is False
    assert not hasattr(configured_service.app.state, "agent_executor")


def test_unauthenticated_production_configuration_aborts_before_database(configured_service, monkeypatch):
    monkeypatch.setattr(settings, "AUTH_USERS", {})
    with pytest.raises(ValueError, match="Production requires"):
        with TestClient(configured_service.app):
            pass
    assert not configured_service.database.exists()
    assert configured_service.app.state.ready is False


def test_unauthenticated_development_configuration_keeps_optional_user_id(configured_service, monkeypatch):
    monkeypatch.setattr(settings, "ENV_MODE", EnvironmentMode.DEVELOPMENT)
    monkeypatch.setattr(settings, "AUTH_USERS", {})
    with TestClient(configured_service.app) as client:
        response = invoke(client, "local request", headers={})
        assert response.status_code == 200, response.text
        assert contents(history(client, headers={})) == ["local request", "reply to local request"]


@pytest.mark.parametrize("endpoint", ["invoke", "stream", "stream/jsonl"])
def test_one_client_token_can_supply_distinct_user_ids(configured_service, monkeypatch, endpoint):
    """The default accepts the existing bearer token and optional user ID."""
    monkeypatch.setattr(settings, "AUTH_SECRET", SecretStr("client-deployment-token"))
    monkeypatch.setattr(settings, "AUTH_USERS", {})
    headers = {"Authorization": "Bearer client-deployment-token"}
    with TestClient(configured_service.app) as client:
        for user_id in (None, "user-1", "user-2"):
            identity = {} if user_id is None else {"user_id": user_id}
            message = user_id or "shared service"
            response = client.post(
                f"/first/{endpoint}",
                headers=headers,
                json={"input": {"message": message}, "thread_id": "same-public-id", **identity},
            )
            assert response.status_code == 200, response.text
        for user_id in (None, "user-1", "user-2"):
            identity = {} if user_id is None else {"user_id": user_id}
            message = user_id or "shared service"
            response = history(client, headers=headers, thread_id="same-public-id", **identity)
            assert contents(response) == [message, "reply to " + message]


def test_shared_token_defaults_to_one_service_user(configured_service, monkeypatch):
    """Omitted user IDs use one stable identity for a single client deployment."""
    monkeypatch.setattr(settings, "AUTH_SECRET", SecretStr("client-deployment-token"))
    monkeypatch.setattr(settings, "AUTH_USERS", {})
    headers = {"Authorization": "Bearer client-deployment-token"}
    with TestClient(configured_service.app) as client:
        response = invoke(client, "hello", headers=headers)
        assert response.status_code == 200, response.text
        implicit = history(client, headers=headers)
        explicit = history(client, headers=headers, user_id=settings.AUTH_SERVICE_USER_ID)
        assert implicit.json() == explicit.json()
        assert contents(implicit) == ["hello", "reply to hello"]
        added = client.post(
            "/first/history/add_messages",
            headers=headers,
            json={"thread_id": "shared", "messages": [{"type": "human", "content": "service note"}]},
        )
        assert added.status_code == 201, added.text
        assert contents(history(client, headers=headers)) == ["hello", "reply to hello", "service note"]
        assert contents(history(client, headers=headers, user_id="user-1")) == []
        cleared = client.request("DELETE", "/first/history/clear", headers=headers, json={"thread_id": "shared"})
        assert cleared.status_code == 200, cleared.text
        assert contents(history(client, headers=headers)) == []


@pytest.mark.parametrize("endpoint", ["invoke", "stream", "stream/jsonl", "history"])
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer another-client-token"}])
def test_default_shared_token_auth_rejects_missing_or_invalid_credentials(
    configured_service, monkeypatch, endpoint, headers
):
    monkeypatch.setattr(settings, "AUTH_SECRET", SecretStr("client-deployment-token"))
    monkeypatch.setattr(settings, "AUTH_USERS", {})
    with TestClient(configured_service.app) as client:
        if endpoint == "history":
            response = history(client, headers=headers)
        else:
            response = client.post(f"/first/{endpoint}", headers=headers, json={"input": {"message": "denied"}})
        assert response.status_code == 401, response.text
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.headers["content-type"].startswith("application/json")
        assert configured_service.calls == []


@pytest.mark.parametrize("endpoint", ["invoke", "stream", "stream/jsonl"])
def test_token_mode_remains_an_explicit_identity_bound_option(configured_service, monkeypatch, endpoint):
    monkeypatch.setattr(settings, "AUTH_MODE", "token")
    monkeypatch.setattr(settings, "AUTH_SECRET", SecretStr("client-deployment-token"))
    monkeypatch.setattr(settings, "AUTH_USERS", {})
    headers = {"Authorization": "Bearer client-deployment-token"}
    with TestClient(configured_service.app) as client:
        denied = client.post(
            f"/first/{endpoint}",
            headers=headers,
            json={"input": {"message": "denied"}, "user_id": "other-user"},
        )
        assert denied.status_code == 403, denied.text
        assert configured_service.calls == []
        allowed = client.post(f"/first/{endpoint}", headers=headers, json={"input": {"message": "allowed"}})
        assert allowed.status_code == 200, allowed.text


@pytest.mark.parametrize("endpoint", ["invoke", "stream", "stream/jsonl"])
def test_user_memory_spans_threads_and_survives_history_clear(configured_service, monkeypatch, endpoint):
    """Keep shared user memories separate from each conversation's checkpoints."""
    user_store = InMemoryStore()

    async def remember(state: MessagesState, config: RunnableConfig, runtime: Runtime):
        assert runtime.store is user_store
        user_id = config["configurable"]["user_id"]
        namespace = ("users", user_id, "memories")
        if state["messages"][-1].content == "Remember tea":
            await runtime.store.aput(namespace, "drink", {"value": "tea"})
        item = await runtime.store.aget(namespace, "drink")
        return {"messages": [AIMessage(item.value["value"] if item else "No saved preference")]}

    builder = StateGraph(MessagesState)
    builder.add_node("remember", remember)
    builder.set_entry_point("remember")
    builder.add_edge("remember", END)
    graph = builder.compile(store=user_store)
    configured_service.module.first = Agent("first", "User memory agent", graph)
    configured_service.module.second = Agent("second", "Another user memory agent", graph)
    monkeypatch.setattr(settings, "AUTH_SECRET", SecretStr("client-deployment-token"))
    monkeypatch.setattr(settings, "AUTH_USERS", {})
    headers = {"Authorization": "Bearer client-deployment-token"}

    with TestClient(configured_service.app) as client:

        def send(message, thread_id, *, user_id="user-1", agent="first"):
            response = client.post(
                f"/{agent}/{endpoint}",
                headers=headers,
                json={"input": {"message": message}, "thread_id": thread_id, "user_id": user_id},
            )
            assert response.status_code == 200, response.text

        def read(thread_id, *, user_id="user-1", agent="first"):
            return contents(history(client, headers=headers, thread_id=thread_id, user_id=user_id, agent=agent))

        send("Remember tea", "conversation-1")
        send("What drink?", "conversation-2")
        assert read("conversation-1") == ["Remember tea", "tea"]
        assert read("conversation-2") == ["What drink?", "tea"]

        send("What drink?", "conversation-2", user_id="user-2")
        assert read("conversation-2", user_id="user-2") == ["What drink?", "No saved preference"]
        send("What drink?", "conversation-2", agent="second")
        assert read("conversation-2", agent="second") == ["What drink?", "tea"]

        cleared = client.request(
            "DELETE",
            "/first/history/clear",
            headers=headers,
            json={"thread_id": "conversation-1", "user_id": "user-1"},
        )
        assert cleared.status_code == 200, cleared.text
        assert read("conversation-1") == []
        assert read("conversation-2") == ["What drink?", "tea"]
        assert user_store.get(("users", "user-1", "memories"), "drink").value == {"value": "tea"}
        assert user_store.get(("users", "user-2", "memories"), "drink") is None
