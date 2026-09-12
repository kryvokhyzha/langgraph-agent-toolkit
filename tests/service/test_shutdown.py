"""Verify shutdown remains bounded when a synchronous telemetry SDK stalls."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.service.handler import shutdown_observability


@pytest.mark.asyncio
async def test_stalled_flush_does_not_hold_worker_shutdown(monkeypatch):
    monkeypatch.setattr(settings, "OBSERVABILITY_SHUTDOWN_TIMEOUT", 0.02)
    release = threading.Event()
    stopped = threading.Event()
    threads = []

    def flush():
        threads.append(threading.current_thread())
        try:
            release.wait(2)
        finally:
            stopped.set()

    try:
        async with asyncio.timeout(0.5):
            await shutdown_observability(SimpleNamespace(before_shutdown=flush))
        assert threads and threads[0].daemon
        assert not stopped.is_set()
    finally:
        release.set()
        threads[0].join(timeout=0.5)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_successful_flush_finishes_before_shutdown_returns():
    flushed = threading.Event()
    await shutdown_observability(SimpleNamespace(before_shutdown=flushed.set))
    assert flushed.is_set()


@pytest.mark.asyncio
async def test_flush_error_does_not_replace_application_shutdown():
    def fail():
        raise RuntimeError("telemetry unavailable")

    await shutdown_observability(SimpleNamespace(before_shutdown=fail))
