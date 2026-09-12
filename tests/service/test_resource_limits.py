"""Verify request limits, dependency health, and cancellation cleanup."""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import aiosqlite
import httpx
import pytest
from fastapi import FastAPI, Request
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from psycopg import OperationalError
from pydantic import SecretStr

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.memory.concurrency import ConversationCoordinator
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper import constants
from langgraph_agent_toolkit.schema import StreamInput
from langgraph_agent_toolkit.service import routes
from langgraph_agent_toolkit.service.handler import create_app
from langgraph_agent_toolkit.service.middleware import RequestSizeLimitMiddleware
from langgraph_agent_toolkit.service.routes import public_router
from langgraph_agent_toolkit.service.utils import jsonl_message_generator, message_generator


def body_scope(headers=()):
    return {"type": "http", "method": "POST", "path": "/", "headers": list(headers), "http_version": "1.1"}


@pytest.mark.asyncio
async def test_declared_oversized_body_is_rejected_before_reading(monkeypatch):
    monkeypatch.setattr(settings, "REQUEST_MAX_BYTES", 8)
    downstream = AsyncMock()
    receive = AsyncMock(side_effect=AssertionError("The body must not be read."))
    sent = []

    async def send(message):
        sent.append(message)

    await RequestSizeLimitMiddleware(downstream)(body_scope([(b"content-length", b"9")]), receive, send)
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["detail"] == "Request body is too large"
    receive.assert_not_awaited()
    downstream.assert_not_awaited()


@pytest.mark.asyncio
async def test_chunked_oversized_body_stops_at_the_limit(monkeypatch):
    monkeypatch.setattr(settings, "REQUEST_MAX_BYTES", 8)
    downstream = AsyncMock()
    receive = AsyncMock(
        side_effect=[
            {"type": "http.request", "body": b"12345", "more_body": True},
            {"type": "http.request", "body": b"6789", "more_body": True},
            AssertionError("The remaining body must not be read."),
        ]
    )
    sent = []

    async def send(message):
        sent.append(message)

    await RequestSizeLimitMiddleware(downstream)(body_scope([(b"transfer-encoding", b"chunked")]), receive, send)
    assert sent[0]["status"] == 413
    assert receive.await_count == 2
    downstream.assert_not_awaited()


@pytest.mark.asyncio
async def test_body_at_limit_is_replayed_without_losing_chunks(monkeypatch):
    monkeypatch.setattr(settings, "REQUEST_MAX_BYTES", 8)
    body = [
        {"type": "http.request", "body": b"1234", "more_body": True},
        {"type": "http.request", "body": b"5678", "more_body": False},
    ]
    receive = AsyncMock(side_effect=[*body, {"type": "http.disconnect"}])
    received = []

    async def downstream(scope, replay, send):
        received.extend([await replay(), await replay()])

    await RequestSizeLimitMiddleware(downstream)(body_scope(), receive, AsyncMock())
    assert received == [
        {"type": "http.request", "body": b"12345678", "more_body": False},
        {"type": "http.disconnect"},
    ]


@pytest.mark.asyncio
async def test_incomplete_body_times_out_before_execution(monkeypatch):
    monkeypatch.setattr(settings, "REQUEST_TIMEOUT", 0.05)
    before = asyncio.all_tasks()
    downstream = AsyncMock()
    interrupted = asyncio.Event()
    sent = []

    async def receive():
        try:
            await asyncio.Event().wait()
        finally:
            interrupted.set()

    async def send(message):
        sent.append(message)

    await asyncio.wait_for(RequestSizeLimitMiddleware(downstream)(body_scope(), receive, send), 1)
    assert sent[0]["status"] == 408
    assert interrupted.is_set()
    downstream.assert_not_awaited()
    await asyncio.sleep(0)
    assert not (asyncio.all_tasks() - before)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [b"-1", b"invalid"])
async def test_invalid_content_length_is_rejected_before_reading(value):
    downstream = AsyncMock()
    receive = AsyncMock(side_effect=AssertionError("The body must not be read."))
    sent = []

    async def send(message):
        sent.append(message)

    await RequestSizeLimitMiddleware(downstream)(body_scope([(b"content-length", value)]), receive, send)
    assert sent[0]["status"] == 400
    receive.assert_not_awaited()
    downstream.assert_not_awaited()


@pytest.mark.asyncio
async def test_disconnect_during_body_does_not_start_work_or_leave_tasks():
    before = asyncio.all_tasks()
    downstream = AsyncMock()
    receive = AsyncMock(
        side_effect=[
            {"type": "http.request", "body": b"partial", "more_body": True},
            {"type": "http.disconnect"},
        ]
    )
    send = AsyncMock()
    await RequestSizeLimitMiddleware(downstream)(body_scope(), receive, send)
    await asyncio.sleep(0)
    downstream.assert_not_awaited()
    send.assert_not_awaited()
    assert not (asyncio.all_tasks() - before)


def probe_app():
    app = FastAPI()
    app.include_router(public_router)
    app.state.ready = True
    app.state.initialized_agents = ["local"]
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("pool_name", ["db_pool", "lock_pool"])
async def test_readiness_tracks_database_outage_and_recovery(pool_name):
    app = probe_app()
    state = {"outage": False, "open": 0}

    @asynccontextmanager
    async def connection(timeout):
        assert timeout == 1
        if state["outage"]:
            raise OperationalError("Test database is unavailable")
        state["open"] += 1
        try:
            yield SimpleNamespace(execute=AsyncMock())
        finally:
            state["open"] -= 1

    setattr(app.state, pool_name, SimpleNamespace(connection=connection))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        assert (await client.get("/health/ready")).status_code == 200
        state["outage"] = True
        assert (await client.get("/health/ready")).status_code == 503
        assert app.state.ready is True
        state["outage"] = False
        assert (await client.get("/health/ready")).status_code == 200
    assert state["open"] == 0


@pytest.mark.asyncio
async def test_readiness_detects_a_closed_sqlite_connection():
    app = probe_app()
    connection = await aiosqlite.connect(":memory:")
    app.state.sqlite_connection = connection
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            assert (await client.get("/health/ready")).status_code == 200
            await connection.close()
            assert (await client.get("/health/ready")).status_code == 503
            assert app.state.ready is True
            async with aiosqlite.connect(":memory:") as replacement:
                app.state.sqlite_connection = replacement
                assert (await client.get("/health/ready")).status_code == 200
    finally:
        await connection.close()


@pytest.fixture
def delayed_service(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    stopped = asyncio.Event()

    async def reply(state):
        if state["messages"][-1].content == "wait":
            started.set()
            try:
                await release.wait()
            finally:
                stopped.set()
        return {"messages": [AIMessage(content="done")]}

    graph = StateGraph(MessagesState)
    graph.add_node("reply", reply)
    graph.set_entry_point("reply")
    graph.add_edge("reply", END)
    module = ModuleType("resource_limit_agent_" + uuid4().hex)
    module.agent = Agent("local", "Local test graph", graph.compile(checkpointer=MemorySaver()), EmptyObservability())
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(settings, "DEFAULT_AGENT", "local")
    monkeypatch.setattr(constants, "_runtime_default_agent", "local")
    monkeypatch.setattr(settings, "AUTH_USERS", {})
    monkeypatch.setattr(settings, "AUTH_SECRET", SecretStr("resource-test-token"))
    monkeypatch.setattr(settings, "AUTH_MODE", "token")
    monkeypatch.setattr(settings, "REQUEST_TIMEOUT", 5)
    executor = AgentExecutor(f"{module.__name__}:agent")
    executor.concurrency = ConversationCoordinator(timeout=0.1, max_waiters=0)
    app = create_app()
    app.state.ready = True
    app.state.initialized_agents = ["local"]
    app.state.agent_executor = executor
    return SimpleNamespace(app=app, executor=executor, started=started, release=release, stopped=stopped)


@pytest.mark.asyncio
async def test_busy_conversation_returns_409_and_recovers(delayed_service):
    service = delayed_service
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(service.app),
        base_url="http://test",
        headers={"Authorization": "Bearer resource-test-token"},
    ) as client:
        payload = {"input": {"message": "wait"}, "thread_id": "one"}
        running = asyncio.create_task(client.post("/local/invoke", json=payload))
        try:
            await asyncio.wait_for(service.started.wait(), 2)
            rejected = await client.post("/local/invoke", json=payload)
            assert rejected.status_code == 409, rejected.text
            assert rejected.headers["retry-after"] == "1"
        finally:
            service.release.set()
            completed = await asyncio.wait_for(running, 2)
        assert completed.status_code == 200, completed.text
        assert not service.executor.concurrency._entries
        followup = await client.post("/local/invoke", json={"input": {"message": "fast"}, "thread_id": "one"})
        assert followup.status_code == 200, followup.text


@pytest.mark.asyncio
async def test_invoke_timeout_returns_504_and_releases_execution(delayed_service, monkeypatch):
    service = delayed_service
    monkeypatch.setattr(settings, "REQUEST_TIMEOUT", 0.1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(service.app),
        base_url="http://test",
        headers={"Authorization": "Bearer resource-test-token"},
    ) as client:
        response = await client.post("/local/invoke", json={"input": {"message": "wait"}, "thread_id": "one"})
        assert response.status_code == 504, response.text
        assert service.stopped.is_set()
        assert not service.executor.concurrency._entries
        monkeypatch.setattr(settings, "REQUEST_TIMEOUT", 2)
        followup = await client.post("/local/invoke", json={"input": {"message": "fast"}, "thread_id": "one"})
        assert followup.status_code == 200, followup.text


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [jsonl_message_generator, message_generator, routes.stream_jsonl])
@pytest.mark.parametrize("cancel_pending", [False, True])
async def test_stream_closure_cancels_execution_without_yielding_on_exit(factory, cancel_pending, monkeypatch):
    waiting = asyncio.Event()
    closed = asyncio.Event()

    async def execution(**kwargs):
        try:
            yield "first"
            waiting.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    app = FastAPI()
    app.state.agent_executor = SimpleNamespace(stream=execution)
    request = Request({"type": "http", "app": app})
    before = asyncio.all_tasks()
    stream_input = StreamInput(input={"message": "hello"})
    if factory is routes.stream_jsonl:
        monkeypatch.setattr(routes, "get_agent", lambda *args: None)
        monkeypatch.setattr(routes, "execution_input", lambda request, agent_id, value: ("public", value))
        response = await factory(stream_input, agent_id="local", request=request)
        generator = response.body_iterator
    else:
        generator = factory(stream_input, request, "local")
    await anext(generator)
    if cancel_pending:
        pending = asyncio.create_task(anext(generator))
        await asyncio.wait_for(waiting.wait(), 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    await generator.aclose()
    assert closed.is_set()
    with pytest.raises(StopAsyncIteration):
        await anext(generator)
    await asyncio.sleep(0)
    assert not (asyncio.all_tasks() - before)


def test_nested_operation_error_does_not_advertise_retry():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from langgraph_agent_toolkit.core.memory.concurrency import NestedConversationError
    from langgraph_agent_toolkit.service.exception_handlers import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/nested")
    async def nested():
        raise NestedConversationError("Compose a subgraph instead")

    response = TestClient(app).get("/nested")
    assert response.status_code == 409
    assert response.json()["error_code"] == "nested_conversation"
    assert "retry-after" not in response.headers
