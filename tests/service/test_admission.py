"""Exercise overload, abandoned requests, and capacity recovery over ASGI."""

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI, Request

from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.service.admission import ResponseSendTimeout
from langgraph_agent_toolkit.service.handler import create_app


@pytest.fixture
def limited_app(monkeypatch):
    monkeypatch.setattr(settings, "REQUEST_MAX_CONCURRENT", 2)
    monkeypatch.setattr(settings, "REQUEST_QUEUE_MAX_WAITERS", 0)
    monkeypatch.setattr(settings, "REQUEST_QUEUE_TIMEOUT", 0.05)
    monkeypatch.setattr(settings, "REQUEST_TIMEOUT", 5)
    monkeypatch.setattr(settings, "CORS_ENABLED", True)
    app = create_app()
    app.state.ready = True
    app.state.initialized_agents = ["test"]
    app.state.started = asyncio.Queue()
    app.state.release = asyncio.Event()
    app.state.stopped = asyncio.Queue()

    @app.post("/work")
    async def work(request: Request):
        payload = await request.json()
        app.state.started.put_nowait(payload["id"])
        try:
            await app.state.release.wait()
            return {"id": payload["id"]}
        finally:
            app.state.stopped.put_nowait(payload["id"])

    return app


def raw_scope(app, path="/work"):
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
        "app": app,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [False, True])
async def test_distinct_requests_are_bounded_before_body_read_and_cors_applies(limited_app, mounted):
    app = limited_app
    serving_app = app
    if mounted:
        serving_app = FastAPI()
        serving_app.mount("/api", app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(serving_app), base_url="http://test/api/" if mounted else "http://test"
    ) as client:
        admitted = [asyncio.create_task(client.post("/work", json={"id": i})) for i in range(2)]
        try:
            assert {await asyncio.wait_for(app.state.started.get(), 1) for _ in range(2)} == {0, 1}
            paths = ["/work", "/invoke", "/stream", "/stream/jsonl"] * 8
            statuses = await asyncio.gather(*[client.post(path, json={"id": i}) for i, path in enumerate(paths)])
            assert all(response.status_code == 503 for response in statuses)
            assert all(response.json()["error_code"] == "service_busy" for response in statuses)
            assert all(response.headers["retry-after"] == "1" for response in statuses)
            assert app.state.request_admission.active == 2
            assert app.state.request_admission.waiting == 0
            assert app.state.started.empty()
            assert (await client.get("/health/live")).status_code == 200
            assert (await client.request("GET", "/health/live", content=b"unused probe body")).status_code == 413
            sent = []

            async def receive():
                pytest.fail("Rejected requests must not read the body")

            async def send(message):
                sent.append(message)

            scope = raw_scope(app)
            scope["headers"].append((b"origin", b"https://client.example"))
            await app(scope, receive, send)
            assert sent[0]["status"] == 503
            assert dict(sent[0]["headers"])[b"access-control-allow-origin"] == b"*"
        finally:
            app.state.release.set()
            responses = await asyncio.gather(*admitted)
        assert all(response.status_code == 200 for response in responses)
        assert app.state.request_admission.active == 0
        assert (await client.post("/work", json={"id": "recovered"})).status_code == 200


@pytest.mark.asyncio
async def test_bounded_queue_timeout_and_cancel_do_not_leak_waiters(limited_app, monkeypatch):
    app = limited_app
    monkeypatch.setattr(settings, "REQUEST_MAX_CONCURRENT", 1)
    monkeypatch.setattr(settings, "REQUEST_QUEUE_MAX_WAITERS", 1)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        active = asyncio.create_task(client.post("/work", json={"id": "active"}))
        await asyncio.wait_for(app.state.started.get(), 1)
        queued = asyncio.create_task(client.post("/work", json={"id": "queued"}))
        try:
            async with asyncio.timeout(1):
                while not app.state.request_admission.waiting:
                    await asyncio.sleep(0)
            assert (await client.post("/work", json={"id": "overflow"})).status_code == 503
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            assert app.state.request_admission.waiting == 0
            assert (await client.post("/work", json={"id": "timeout"})).status_code == 503
            assert app.state.request_admission.waiting == 0
        finally:
            app.state.release.set()
            assert (await active).status_code == 200
        assert (await client.post("/work", json={"id": "recovered"})).status_code == 200


@pytest.mark.asyncio
async def test_disconnect_cancels_invoke_work_before_releasing_capacity(limited_app):
    app = limited_app
    incoming = asyncio.Queue()
    incoming.put_nowait({"type": "http.request", "body": b'{"id":"abandoned"}', "more_body": False})
    sent = []

    async def send(message):
        sent.append(message)

    before = asyncio.all_tasks()
    request = asyncio.create_task(app(raw_scope(app), incoming.get, send))
    await asyncio.wait_for(app.state.started.get(), 1)
    incoming.put_nowait({"type": "http.disconnect"})
    await asyncio.wait_for(request, 1)
    assert app.state.stopped.get_nowait() == "abandoned"
    assert app.state.request_admission.active == 0
    assert not sent
    await asyncio.sleep(0)
    assert not (asyncio.all_tasks() - before)


@pytest.mark.asyncio
async def test_deadline_covers_non_agent_routes_and_recovers(limited_app, monkeypatch):
    monkeypatch.setattr(settings, "REQUEST_TIMEOUT", 0.05)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(limited_app), base_url="http://test") as client:
        response = await client.post("/work", json={"id": "timeout"})
        assert response.status_code == 504
        assert response.json()["error_code"] == "request_timeout"
        assert limited_app.state.stopped.get_nowait() == "timeout"
        assert limited_app.state.request_admission.active == 0


@pytest.mark.asyncio
async def test_nested_service_call_fails_before_waiting_for_its_own_capacity(limited_app, monkeypatch):
    app = limited_app
    monkeypatch.setattr(settings, "REQUEST_MAX_CONCURRENT", 1)
    monkeypatch.setattr(settings, "REQUEST_QUEUE_MAX_WAITERS", 1)

    @app.post("/nested")
    async def nested():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            response = await client.post("/work", json={"id": "nested"})
            assert response.status_code == 409
            assert "retry-after" not in response.headers
            return response.json()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await asyncio.wait_for(client.post("/nested", json={}), 1)
        assert response.status_code == 200
        assert response.json()["error_code"] == "nested_request"
        assert app.state.started.empty()


@pytest.mark.asyncio
async def test_slow_response_send_has_a_deadline(limited_app, monkeypatch):
    app = limited_app
    monkeypatch.setattr(settings, "RESPONSE_SEND_TIMEOUT", 0.05)
    app.state.release.set()
    incoming = asyncio.Queue()
    incoming.put_nowait({"type": "http.request", "body": json.dumps({"id": "slow"}).encode()})
    sent = []

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body":
            await asyncio.Event().wait()

    with pytest.raises(ResponseSendTimeout):
        await asyncio.wait_for(app(raw_scope(app), incoming.get, send), 1)
    assert app.state.request_admission.active == 0
    assert [message["status"] for message in sent if message["type"] == "http.response.start"] == [200]


@pytest.mark.asyncio
async def test_stalled_cleanup_fails_probes_without_releasing_live_work(limited_app, monkeypatch):
    app = limited_app
    monkeypatch.setattr(settings, "REQUEST_MAX_CONCURRENT", 1)
    monkeypatch.setattr(settings, "REQUEST_TIMEOUT", 0.05)
    monkeypatch.setattr(settings, "REQUEST_CLEANUP_TIMEOUT", 0.02)
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()

    @app.post("/stalled")
    async def stalled():
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await finish_cleanup.wait()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        request = asyncio.create_task(client.post("/stalled", json={}))
        try:
            await asyncio.wait_for(cleanup_started.wait(), 1)
            async with asyncio.timeout(1):
                while not getattr(app.state, "stalled_request_cleanups", 0):
                    await asyncio.sleep(0)
            assert (await client.get("/health/live")).status_code == 503
            assert (await client.get("/health/ready")).status_code == 503
            assert (await client.post("/work", json={"id": "must wait"})).status_code == 503
            assert app.state.request_admission.active == 1
            assert not request.done()
        finally:
            finish_cleanup.set()
            assert (await asyncio.wait_for(request, 1)).status_code == 504
        assert app.state.request_admission.active == 0
        assert (await client.get("/health/live")).status_code == 200
        assert (await client.get("/health/ready")).status_code == 200
