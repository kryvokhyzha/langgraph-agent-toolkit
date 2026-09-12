"""Check the local model simulator before process load tests."""

import asyncio
import json
import os

import httpx
import pytest
from starlette.responses import JSONResponse
from support import MODEL_API_KEY, ModelSimulator, WorkerMetrics, local_model_url


def completion(request_id="request-one", *, stream=False):
    return {
        "model": "local-load-model",
        "messages": [{"role": "user", "content": json.dumps({"request_id": request_id})}],
        "stream": stream,
        "stream_options": {"include_usage": True},
    }


def client(app, *, address="127.0.0.1"):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=(address, 12345)),
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {MODEL_API_KEY}"},
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "http://localhost:9000/v1",
        "http://127.0.0.1/v1",
        "http://127.0.0.1:9000/wrong",
        "http://user:secret@127.0.0.1:9000/v1",
        "http://127.0.0.1:9000/v1?remote=1",
        "http://192.0.2.1:9000/v1",
    ],
)
def test_reject_external_model_addresses(monkeypatch, url):
    monkeypatch.setenv("LAT_LOAD_MODEL_URL", url)
    with pytest.raises(ValueError, match="literal-loopback"):
        local_model_url()


def test_accept_literal_ipv6_loopback(monkeypatch):
    monkeypatch.setenv("LAT_LOAD_MODEL_URL", "http://[::1]:9000/v1/")
    assert local_model_url() == "http://[::1]:9000/v1"


async def test_stream_echoes_first_id_and_complete_wire_protocol():
    app = ModelSimulator()
    async with client(app) as http:
        control = await http.post(
            "/control",
            json={"first_token_ms": 0, "chunk_interval_ms": 0, "stream_chunk_count": 3, "stream_chunk_bytes": 5},
        )
        assert control.status_code == 200
        result = await http.post("/v1/chat/completions", json=completion(stream=True))
        lines = [line.removeprefix("data: ") for line in result.text.splitlines() if line.startswith("data: ")]
        assert lines[-1] == "[DONE]"
        events = [json.loads(line) for line in lines[:-1]]
        assert events[0]["choices"][0]["delta"]["content"] == "request-one"
        assert events[1]["choices"][0]["delta"]["content"] == " " * 5
        assert events[-2]["choices"][0]["finish_reason"] == "stop"
        assert events[-1]["usage"]["total_tokens"] == 6
        metrics = (await http.get("/metrics")).json()
        assert metrics["requests"] == metrics["completed"] == metrics["stream_requests"] == 1
        assert metrics["active"] == 0
        assert metrics["distinct_connections"] == 1


async def test_rate_limit_budget_then_success_and_invalid_control():
    app = ModelSimulator()
    async with client(app) as http:
        assert (await http.post("/control", json={"mode": "429", "remaining": 1, "delay_ms": 0})).status_code == 200
        failed = await http.post("/v1/chat/completions", json=completion())
        assert failed.status_code == 429
        assert float(failed.headers["retry-after"]) == 0.05
        success = await http.post("/v1/chat/completions", json=completion("request-two"))
        assert success.json()["choices"][0]["message"]["content"] == "request-two"
        assert (await http.post("/control", json={"mode": "external"})).status_code == 422
        assert (await http.post("/control", json={"stream_chunk_bytes": 1048576})).status_code == 200
        assert (await http.post("/control", json={"stream_chunk_count": 1000})).status_code == 422
        metrics = (await http.get("/metrics")).json()
        assert metrics["requests"] == 2
        assert metrics["rate_limited"] == 1
        assert metrics["active"] == 0


async def test_stalled_request_cancellation_releases_simulator_capacity():
    app = ModelSimulator()
    async with client(app) as http:
        await http.post("/control", json={"mode": "stall", "remaining": 1, "stall_seconds": 30, "delay_ms": 0})
        task = asyncio.create_task(http.post("/v1/chat/completions", json=completion()))
        async with asyncio.timeout(1):
            while app.counts["stalled"] != 1:
                await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert app.active == 0
        assert app.counts["cancelled"] == 1
        assert (await http.post("/v1/chat/completions", json=completion())).status_code == 200


async def test_metrics_bypass_inner_service_and_reject_remote_access():
    async def forbidden(scope, receive, send):
        raise AssertionError("Metrics must not enter the production admission queue")

    app = WorkerMetrics(forbidden, metrics=lambda: {"active": 8, "waiting": 16})
    async with client(app) as http:
        result = await http.get("/load/metrics")
        assert result.json() == {"active": 8, "waiting": 16}
    async with client(app, address="192.0.2.2") as http:
        assert (await http.get("/load/metrics")).status_code == 403
    async with client(ModelSimulator(), address="192.0.2.2") as http:
        assert (await http.post("/control", json={"mode": "stall"})).status_code == 403


async def test_agent_responses_identify_the_actual_worker_only_in_the_test_wrapper():
    async def application(scope, receive, send):
        headers = {"x-load-worker": "99999999"} if scope["path"].startswith("/load-agent/") else {}
        await JSONResponse({"ok": True}, headers=headers)(scope, receive, send)

    async with client(WorkerMetrics(application, metrics=lambda: {"pid": os.getpid()})) as http:
        response = await http.post("/load-agent/invoke", json={"input": {"message": "test"}})
        assert response.status_code == 200
        assert response.headers.get_list("x-load-worker") == [str(os.getpid())]
        assert "x-load-worker" not in (await http.get("/health/ready")).headers
        assert "x-load-worker" not in (await http.get("/load/metrics")).headers
