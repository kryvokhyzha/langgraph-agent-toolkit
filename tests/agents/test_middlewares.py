"""Tests for the create_agent middleware that ports custom create_react_agent features."""

from unittest.mock import MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from langgraph_agent_toolkit.agents.components.middlewares import (
    ClearIntermediateToolCallsMiddleware,
    ImmediateGenerationMiddleware,
    SanitizeHistoryMiddleware,
    TrimMessagesMiddleware,
)
from langgraph_agent_toolkit.core.models.fake import FakeToolModel
from langgraph_agent_toolkit.core.settings import settings


def _tool_call(call_id: str = "1") -> dict:
    return {"name": "add", "args": {"a": 1, "b": 2}, "id": call_id, "type": "tool_call"}


def test_immediate_generation_counts_model_calls_this_run():
    """Only AI messages since the latest human message count as this-run model calls."""
    mw = ImmediateGenerationMiddleware()
    messages = [
        HumanMessage("old turn"),
        AIMessage("old answer"),  # previous turn — must NOT count
        HumanMessage("this turn"),
        AIMessage("", tool_calls=[_tool_call("1")]),
        ToolMessage("3", tool_call_id="1"),
        AIMessage("", tool_calls=[_tool_call("2")]),
    ]
    assert mw._calls_made_this_run(messages) == 2


def test_immediate_generation_forces_tool_free_answer_at_budget():
    """At the budget, tools are stripped and the synthesize-now instruction is appended."""
    mw = ImmediateGenerationMiddleware(model_call_limit=2)
    request = MagicMock()
    # 1 model call already made this run -> next call is the 2nd (== limit) -> force
    request.messages = [HumanMessage("q"), AIMessage("", tool_calls=[_tool_call()])]
    request.system_message = SystemMessage("You are helpful.")
    request.override.return_value = "OVERRIDDEN"
    handler = MagicMock(return_value="RESPONSE")

    result = mw.wrap_model_call(request, handler)

    kwargs = request.override.call_args.kwargs
    assert kwargs["tools"] == []
    assert "best final answer" in kwargs["system_message"].content  # answer-now wording, not a prohibition
    assert "You are helpful." in kwargs["system_message"].content  # original preserved
    handler.assert_called_once_with("OVERRIDDEN")
    assert result == "RESPONSE"


def test_immediate_generation_default_budget_from_recursion_limit():
    """With no explicit limit, the budget derives from the recursion limit (~half), not a flat 5."""
    mw = ImmediateGenerationMiddleware()
    assert mw.model_call_limit == max(1, settings.DEFAULT_RECURSION_LIMIT // 2)
    assert mw.model_call_limit > 5  # not the old over-aggressive default


def test_immediate_generation_passes_through_under_budget():
    """Below the budget, the request is forwarded unchanged."""
    mw = ImmediateGenerationMiddleware(model_call_limit=5)
    request = MagicMock()
    request.messages = [HumanMessage("q")]  # 0 calls made -> well under budget
    handler = MagicMock(return_value="R")

    mw.wrap_model_call(request, handler)

    request.override.assert_not_called()
    handler.assert_called_once_with(request)


def test_immediate_generation_rejects_bad_limit():
    with pytest.raises(ValueError, match="model_call_limit"):
        ImmediateGenerationMiddleware(model_call_limit=0)


def test_sanitize_history_strips_incomplete_tool_calls():
    """The messages sent to the model have incomplete tool calls removed."""
    mw = SanitizeHistoryMiddleware()
    request = MagicMock()
    # AIMessage requests a tool but there is no matching ToolMessage (interrupted run)
    request.messages = [HumanMessage("q"), AIMessage("", tool_calls=[_tool_call("missing")])]
    request.override.return_value = "SANITIZED"
    handler = MagicMock(return_value="R")

    mw.wrap_model_call(request, handler)

    sanitized = request.override.call_args.kwargs["messages"]
    assert sanitized[-1].tool_calls == []  # incomplete tool call removed
    handler.assert_called_once_with("SANITIZED")


def _ai_tool(name: str, call_id: str, args: dict | None = None) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": call_id, "type": "tool_call"}])


def _kept_tool_ids(mw: ClearIntermediateToolCallsMiddleware, messages: list) -> list[str]:
    out = mw._process(messages)
    kept = [m.tool_call_id for m in out if isinstance(m, ToolMessage)]
    # invariant: pairing is always consistent (no orphan in either direction)
    ai_ids = {c["id"] for m in out if isinstance(m, AIMessage) and m.tool_calls for c in m.tool_calls}
    assert set(kept) == ai_ids
    return kept


def test_clear_intermediate_keeps_current_turn_dedups_previous():
    """Earlier turns keep only the most recent result per key; the current turn is untouched."""
    mw = ClearIntermediateToolCallsMiddleware(by="name")
    messages = [
        HumanMessage("turn 1"),
        _ai_tool("search", "s1"),
        ToolMessage("old", tool_call_id="s1", name="search"),
        _ai_tool("search", "s2"),
        ToolMessage("newer", tool_call_id="s2", name="search"),
        AIMessage("answer 1"),
        HumanMessage("turn 2"),  # current turn below — preserved in full
        _ai_tool("search", "s3"),
        ToolMessage("current", tool_call_id="s3", name="search"),
    ]
    kept = _kept_tool_ids(mw, messages)
    assert kept == ["s2", "s3"]  # s1 (intermediate) dropped; last-of-prev + current kept


def test_name_args_preserves_distinct_argument_calls():
    """by='name_args' (default) keeps repeat calls that have different arguments."""
    mw = ClearIntermediateToolCallsMiddleware(by="name_args")
    messages = [
        HumanMessage("turn 1"),
        _ai_tool("search", "a", {"q": "tokyo"}),
        ToolMessage("tokyo", tool_call_id="a", name="search"),
        _ai_tool("search", "b", {"q": "paris"}),
        ToolMessage("paris", tool_call_id="b", name="search"),
        HumanMessage("turn 2"),
    ]
    assert sorted(_kept_tool_ids(mw, messages)) == ["a", "b"]  # distinct args -> both kept


def test_name_collapses_distinct_argument_calls():
    """by='name' collapses repeats of a tool regardless of arguments."""
    mw = ClearIntermediateToolCallsMiddleware(by="name")
    messages = [
        HumanMessage("turn 1"),
        _ai_tool("search", "a", {"q": "tokyo"}),
        ToolMessage("tokyo", tool_call_id="a", name="search"),
        _ai_tool("search", "b", {"q": "paris"}),
        ToolMessage("paris", tool_call_id="b", name="search"),
        HumanMessage("turn 2"),
    ]
    assert _kept_tool_ids(mw, messages) == ["b"]  # only the last survives


def test_keep_last_n_and_exclude_tools():
    mw = ClearIntermediateToolCallsMiddleware(by="name", keep_last_n=2, exclude_tools={"writer"})
    messages = [
        HumanMessage("turn 1"),
        _ai_tool("search", "s1"),
        ToolMessage("r1", tool_call_id="s1", name="search"),
        _ai_tool("search", "s2"),
        ToolMessage("r2", tool_call_id="s2", name="search"),
        _ai_tool("search", "s3"),
        ToolMessage("r3", tool_call_id="s3", name="search"),
        _ai_tool("writer", "w1"),
        ToolMessage("wrote", tool_call_id="w1", name="writer"),
        HumanMessage("turn 2"),
    ]
    kept = _kept_tool_ids(mw, messages)
    assert "s1" not in kept  # keep_last_n=2 -> oldest search dropped
    assert {"s2", "s3"}.issubset(set(kept))  # last 2 search kept
    assert "w1" in kept  # excluded tool never deduped


def test_disabled_passes_through_unchanged():
    mw = ClearIntermediateToolCallsMiddleware(by="name", enabled=False)
    messages = [
        HumanMessage("turn 1"),
        _ai_tool("search", "s1"),
        ToolMessage("r1", tool_call_id="s1", name="search"),
        _ai_tool("search", "s2"),
        ToolMessage("r2", tool_call_id="s2", name="search"),
        HumanMessage("turn 2"),
    ]
    assert mw._process(messages) is messages


def test_uses_settings_defaults(monkeypatch):
    """With no constructor args the middleware reads its config from Settings."""
    monkeypatch.setattr(settings, "CLEAR_INTERMEDIATE_TOOL_CALLS_BY", "name")
    monkeypatch.setattr(settings, "CLEAR_INTERMEDIATE_TOOL_CALLS_KEEP_LAST_N", 1)
    monkeypatch.setattr(settings, "CLEAR_INTERMEDIATE_TOOL_CALLS", True)
    mw = ClearIntermediateToolCallsMiddleware()
    assert (mw.by, mw.keep_last_n, mw.enabled) == ("name", 1, True)


def test_clear_intermediate_rejects_bad_by():
    with pytest.raises(ValueError, match="name_args"):
        ClearIntermediateToolCallsMiddleware(by="bogus")


def test_clear_intermediate_no_human_returns_unchanged():
    mw = ClearIntermediateToolCallsMiddleware()
    msgs = [_ai_tool("search", "s1"), ToolMessage("r", tool_call_id="s1", name="search")]
    assert mw._process(msgs) is msgs


def test_trim_messages_bounds_to_max_and_starts_on_human():
    """The model's view is trimmed to the last max_messages, starting on a human turn."""
    mw = TrimMessagesMiddleware(max_messages=4)
    request = MagicMock()
    request.messages = [SystemMessage("sys")] + [
        HumanMessage(f"h{i}") if i % 2 == 0 else AIMessage(f"a{i}") for i in range(8)
    ]
    request.override.return_value = "TRIMMED"
    handler = MagicMock(return_value="R")

    mw.wrap_model_call(request, handler)

    trimmed = request.override.call_args.kwargs["messages"]
    assert len(trimmed) <= mw.max_messages
    handler.assert_called_once_with("TRIMMED")


def test_trim_messages_keeps_short_history_unchanged():
    mw = TrimMessagesMiddleware(max_messages=10)
    request = MagicMock()
    # realistic model input ends on a human turn (the model is about to answer it)
    request.messages = [HumanMessage("q1"), AIMessage("a1"), HumanMessage("q2")]
    handler = MagicMock()

    mw.wrap_model_call(request, handler)

    assert request.override.call_args.kwargs["messages"] == request.messages


def test_trim_messages_default_from_settings():
    assert TrimMessagesMiddleware().max_messages == settings.DEFAULT_MAX_MESSAGE_HISTORY_LENGTH


def test_trim_messages_rejects_bad_max():
    with pytest.raises(ValueError, match="max_messages"):
        TrimMessagesMiddleware(max_messages=0)


def test_create_agent_with_middleware_runs_end_to_end():
    """create_agent composes both middleware and runs without error (FAKE model, no tool loop)."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent = create_agent(
        model=FakeToolModel(responses=["final answer"]),
        tools=[add],
        middleware=[SanitizeHistoryMiddleware(), ImmediateGenerationMiddleware(model_call_limit=3)],
    )
    out = agent.invoke({"messages": [HumanMessage("hi")]})
    assert out["messages"][-1].content == "final answer"
