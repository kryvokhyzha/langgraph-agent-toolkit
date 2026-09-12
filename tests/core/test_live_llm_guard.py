"""Check live-test budget controls without creating a provider client."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


@pytest.fixture
def live_guard(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.delenv("LAT_TEST_LLM_CHILD", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-only")
    monkeypatch.setenv("LAT_TEST_OPENAI_MODEL", "offline-model")
    monkeypatch.setenv("LAT_TEST_OPENAI_MODEL_KWARGS", "{}")
    from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory

    factory = Mock(side_effect=AssertionError("An offline test tried to construct a provider client"))
    monkeypatch.setattr(CompletionModelFactory, "create", factory)
    source = Path(__file__).resolve().parents[1] / "e2e" / "live_support_agent.py"
    spec = importlib.util.spec_from_file_location("offline_live_guard_support", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, factory


async def test_live_tool_graph_preserves_real_tool_history_and_fixed_request_limits(live_guard, monkeypatch):
    support, factory = live_guard
    monkeypatch.setenv("LAT_TEST_LLM_CHILD", "yes")
    monkeypatch.setenv("LAT_TEST_OPENAI_MODEL_KWARGS", '{"reasoning_effort":"none","temperature":0}')
    requested = AIMessage(
        content="",
        tool_calls=[{"name": "add", "args": {"a": 2, "b": 5}, "id": "offline-tool-call"}],
        usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
    )
    final = AIMessage(
        content="The sum is seven.",
        usage_metadata={"input_tokens": 23, "output_tokens": 5, "total_tokens": 28},
    )
    tool_model = Mock()
    tool_model.bind_tools.return_value = tool_model
    tool_model.ainvoke = AsyncMock(return_value=requested)
    final_model = Mock()
    final_model.ainvoke = AsyncMock(return_value=final)
    factory.side_effect = [tool_model, final_model]

    result = await support.live_agent.graph.ainvoke({"messages": [HumanMessage("live-tools: Add 2 and 5.")]})

    history = result["messages"]
    assert [message.type for message in history] == ["human", "ai", "tool", "ai"]
    assert history[1] is requested
    assert isinstance(history[2], ToolMessage)
    assert history[2].tool_call_id == "offline-tool-call"
    assert history[2].content == "7"
    assert history[3] is final
    assert final_model.ainvoke.await_args.args[0][-1] is history[2]
    final_model.bind_tools.assert_not_called()
    tool_model.bind_tools.assert_called_once_with([support.add], tool_choice="add", parallel_tool_calls=False)
    assert support.calls.attempted == support.calls.completed == 2
    assert support.calls.tools_completed == 1
    for invocation in factory.call_args_list:
        assert invocation.args == ("openai", "offline-model")
        parameters = invocation.kwargs
        assert parameters["reasoning_effort"] == "none" and parameters["temperature"] == 0
        assert parameters["base_url"] == "https://api.openai.com/v1"
        assert parameters["api_key"] == "offline-test-only"
        assert parameters["model_parameter_values"] == ()
        assert parameters["configurable_fields"] == ()
        assert parameters["max_completion_tokens"] == 256
        assert parameters["max_retries"] == 0 and parameters["n"] == 1
        assert parameters["store"] is False and parameters["timeout"] == 60
        assert parameters["stream_usage"] is True
        assert parameters["streaming"] is False and parameters["disable_streaming"] is True


async def test_live_budget_rejects_seventh_concurrent_call_and_does_not_refund_cancellation(live_guard, monkeypatch):
    support, factory = live_guard
    monkeypatch.setenv("LAT_TEST_LLM_CHILD", "yes")
    entered = 0
    all_entered = asyncio.Event()
    release = asyncio.Event()

    async def reply(*args, **kwargs):
        nonlocal entered
        entered += 1
        if entered == 6:
            all_entered.set()
        await release.wait()
        return AIMessage(content="Offline reply.")

    factory.side_effect = None
    factory.return_value = Mock(ainvoke=AsyncMock(side_effect=reply))
    tasks = [asyncio.create_task(support.call_model([], {}, streaming=True)) for _ in range(6)]
    try:
        await asyncio.wait_for(all_entered.wait(), 1)
        with pytest.raises(RuntimeError, match="budget"):
            await support.call_model([], {}, streaming=True)
        assert factory.call_count == 6
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
        with pytest.raises(RuntimeError, match="budget"):
            await support.call_model([], {}, streaming=True)
        assert factory.call_count == 6
        release.set()
        await asyncio.gather(*tasks[1:])
        assert support.calls.attempted == 6
        assert support.calls.completed == 5 and support.calls.cancelled == 1
        assert support.calls.active == 0
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_live_calls_require_explicit_child_activation(live_guard):
    support, factory = live_guard
    with pytest.raises(RuntimeError, match="not enabled"):
        await support.call_model([], {}, streaming=True)
    factory.assert_not_called()
    assert support.calls.attempted == support.calls.active == 0


@pytest.mark.parametrize(
    "parameters",
    [
        "not-json",
        "[]",
        json.dumps(
            {
                "reasoning_effort": "none",
                "base_url": "https://example.invalid/v1",
                "api_key": "injected-test-only",
                "max_completion_tokens": 9999,
                "max_retries": 4,
                "n": 8,
                "store": True,
            }
        ),
    ],
    ids=["invalid-json", "not-an-object", "override-protected-options"],
)
async def test_live_model_kwargs_fail_before_provider_construction(live_guard, monkeypatch, parameters):
    support, factory = live_guard
    monkeypatch.setenv("LAT_TEST_LLM_CHILD", "yes")
    monkeypatch.setenv("LAT_TEST_OPENAI_MODEL_KWARGS", parameters)
    with pytest.raises(ValueError):
        await support.call_model([], {}, streaming=True)
    factory.assert_not_called()
    assert support.calls.active == 0
