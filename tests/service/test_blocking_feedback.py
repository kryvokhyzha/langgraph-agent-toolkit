"""Check actual thread capacity after feedback callers cancel."""

import asyncio
import threading
from contextvars import ContextVar
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.service.admission import ServiceBusyError
from langgraph_agent_toolkit.service.blocking import BoundedBlockingExecutor
from langgraph_agent_toolkit.service.handler import create_app


async def test_cancelled_callers_do_not_release_running_thread_capacity():
    executor = BoundedBlockingExecutor(capacity=2)
    loop = asyncio.get_running_loop()
    entered = [asyncio.Event(), asyncio.Event()]
    finished = [asyncio.Event(), asyncio.Event()]
    release = [threading.Event(), threading.Event()]
    identity = ContextVar("feedback_test_identity")
    observed = []

    def feedback(index):
        observed.append((index, identity.get(), threading.current_thread().daemon))
        loop.call_soon_threadsafe(entered[index].set)
        release[index].wait()
        loop.call_soon_threadsafe(finished[index].set)

    async def call(index):
        identity.set(f"owner-{index}")
        await executor.run(feedback, index)

    callers = [asyncio.create_task(call(index)) for index in range(2)]
    try:
        async with asyncio.timeout(2):
            await asyncio.gather(*(event.wait() for event in entered))
            for caller in callers:
                caller.cancel()
            results = await asyncio.gather(*callers, return_exceptions=True)
            assert all(isinstance(result, asyncio.CancelledError) for result in results)
            for _ in range(20):
                with pytest.raises(ServiceBusyError):
                    await executor.run(feedback, 0)
            assert sorted(observed) == [(0, "owner-0", True), (1, "owner-1", True)]

            release[0].set()
            await finished[0].wait()
            await asyncio.sleep(0)
            assert await executor.run(lambda left, *, right: left + right, 6, right=7) == 13
            assert not finished[1].is_set()
    finally:
        for event in release:
            event.set()
        await asyncio.gather(*callers, return_exceptions=True)
        await executor.aclose(timeout=1)


async def test_failed_feedback_releases_capacity_and_propagates_the_error():
    executor = BoundedBlockingExecutor(capacity=1)

    def fail():
        raise ValueError("Invalid feedback score")

    with pytest.raises(ValueError, match="Invalid feedback score"):
        await executor.run(fail)
    assert await executor.run(lambda: "recovered") == "recovered"
    await executor.aclose(timeout=1)


async def test_shutdown_does_not_wait_forever_for_a_stuck_feedback_thread():
    executor = BoundedBlockingExecutor(capacity=1)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    finished = asyncio.Event()
    workers = []

    def feedback():
        workers.append(threading.current_thread())
        loop.call_soon_threadsafe(entered.set)
        release.wait()
        loop.call_soon_threadsafe(finished.set)

    caller = asyncio.create_task(executor.run(feedback))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.wait_for(executor.aclose(timeout=0.02), timeout=1)
        assert workers[0].is_alive()
        assert workers[0].daemon
        with pytest.raises(ServiceBusyError):
            await executor.run(lambda: None)
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=1)
        await caller


async def test_feedback_api_rejects_work_while_a_timed_out_call_still_runs(monkeypatch):
    monkeypatch.setattr(settings, "REQUEST_MAX_CONCURRENT", 1)
    monkeypatch.setattr(settings, "REQUEST_TIMEOUT", 1)
    monkeypatch.setattr(settings, "AUTH_USERS", {})
    monkeypatch.setattr(settings, "AUTH_SECRET", SecretStr("feedback-test-token"))
    monkeypatch.setattr(settings, "AUTH_MODE", "trusted")
    entered = threading.Event()
    release = threading.Event()
    finished = asyncio.Event()
    loop = asyncio.get_running_loop()

    def feedback(**kwargs):
        entered.set()
        release.wait()
        loop.call_soon_threadsafe(finished.set)

    app = create_app()
    app.state.agent_executor = SimpleNamespace(
        get_agent=lambda name: SimpleNamespace(observability=SimpleNamespace(record_feedback=feedback))
    )
    payload = {"run_id": "7d638a81-7d4a-4e2d-b16f-633354efbe22", "key": "correctness", "score": 1}
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app),
            base_url="http://test",
            headers={"Authorization": "Bearer feedback-test-token"},
        ) as client:
            # Build lazy route handlers before testing a timeout during feedback.
            release.set()
            assert (await client.post("/local/feedback", json=payload)).status_code == 201
            release.clear()
            entered.clear()
            finished.clear()
            monkeypatch.setattr(settings, "REQUEST_TIMEOUT", 0.05)
            timed_out = await client.post("/local/feedback", json=payload)
            assert timed_out.status_code == 504
            assert entered.is_set()
            rejected = await client.post("/local/feedback", json=payload)
            assert rejected.status_code == 503
            assert rejected.json()["error_code"] == "service_busy"
            assert rejected.headers["retry-after"] == "1"
            assert (await client.get("/health/live")).status_code == 200

            release.set()
            await asyncio.wait_for(finished.wait(), timeout=1)
            await asyncio.sleep(0)
            monkeypatch.setattr(settings, "REQUEST_TIMEOUT", 1)
            assert (await client.post("/local/feedback", json=payload)).status_code == 201
    finally:
        release.set()
        await app.state.blocking_executor.aclose(timeout=1)
