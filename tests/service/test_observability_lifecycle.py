"""Close each service observability provider once, including failed startup."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI

from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.service import handler


@pytest.mark.parametrize("fail_startup", [False, True])
async def test_distinct_agent_providers_close_once_after_request_executor(monkeypatch, fail_startup):
    events = []
    default = Mock()
    shared = Mock()
    separate = Mock()
    providers = {"default": default, "shared": shared, "separate": separate}
    for name, provider in providers.items():
        provider.before_shutdown.side_effect = lambda key=name: events.append(key)
    agents = {
        name: SimpleNamespace(graph=SimpleNamespace(checkpointer=object()), observability=provider)
        for name, provider in {
            "implicit": None,
            "explicit-default": default,
            "shared-first": shared,
            "shared-second": shared,
            "separate": separate,
        }.items()
    }
    executor = Mock()
    executor.get_all_agent_info.return_value = [SimpleNamespace(key=name) for name in agents]
    executor.get_agent.side_effect = agents.__getitem__
    configure = AsyncMock(side_effect=RuntimeError("Graph startup failed") if fail_startup else None)
    monkeypatch.setattr(settings, "MEMORY_BACKEND", None)
    monkeypatch.setattr(handler, "validate_auth_configuration", lambda: None)
    monkeypatch.setattr(handler, "AgentExecutor", lambda *args, **kwargs: executor)
    monkeypatch.setattr(handler.ObservabilityFactory, "create", lambda *args, **kwargs: default)
    monkeypatch.setattr(handler, "configure_mcp_agents", configure)
    close_requests = handler.BoundedBlockingExecutor.aclose

    async def close_and_record(self, *args, **kwargs):
        await close_requests(self, *args, **kwargs)
        events.append("requests")

    monkeypatch.setattr(handler.BoundedBlockingExecutor, "aclose", close_and_record)
    app = FastAPI()
    if fail_startup:
        with pytest.raises(RuntimeError, match="Graph startup failed"):
            async with handler.lifespan(app):
                pytest.fail("Startup must fail before the service accepts requests")
    else:
        async with handler.lifespan(app):
            assert app.state.ready
            assert agents["implicit"].observability is default
            assert events == []
    assert events[0] == "requests"
    assert set(events[1:]) == set(providers)
    for provider in providers.values():
        provider.before_shutdown.assert_called_once_with()
    assert not app.state.ready
    assert not hasattr(app.state, "agent_executor")
