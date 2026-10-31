"""Check worker CLI supervision without a model or database."""

from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace

import pytest


pytest.importorskip("procrastinate")


def test_worker_cli_fails_when_library_worker_stops_unexpectedly(monkeypatch):
    from langgraph_agent_toolkit import run_worker
    from langgraph_agent_toolkit.core import background, runtime
    from langgraph_agent_toolkit.core.settings import settings

    closed = []

    async def run(*args, **kwargs):
        assert kwargs["install_signal_handlers"] is False
        # Procrastinate can return normally after its listener fails.

    @asynccontextmanager
    async def resources(value):
        try:
            yield value
        finally:
            closed.append(value)

    monkeypatch.setattr(settings, "BACKGROUND_ENABLED", True)
    monkeypatch.setattr(runtime, "agent_runtime", lambda: resources("runtime"))
    monkeypatch.setattr(background, "queue_from_settings", lambda: resources(SimpleNamespace(run_worker=run)))
    monkeypatch.setattr("procrastinate.signals.on_stop", lambda callback: nullcontext())
    with pytest.raises(background.BackgroundUnavailable, match="stopped unexpectedly"):
        run_worker.run()
    assert len(closed) == 2


def test_worker_cli_requires_explicit_enable(monkeypatch):
    from langgraph_agent_toolkit import run_worker
    from langgraph_agent_toolkit.core.background import BackgroundUnavailable
    from langgraph_agent_toolkit.core.settings import settings

    monkeypatch.setattr(settings, "BACKGROUND_ENABLED", False)
    with pytest.raises(BackgroundUnavailable, match="BACKGROUND_ENABLED"):
        run_worker.run()
