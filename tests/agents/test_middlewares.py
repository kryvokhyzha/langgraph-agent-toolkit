"""Test ``create_agent`` middleware."""

from unittest.mock import MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from langgraph_agent_toolkit.agents.components.middlewares import (
    ClearIntermediateToolCallsMiddleware,
    ImmediateGenerationMiddleware,
    SanitizeHistoryMiddleware,
    TokenTrimMiddleware,
    TrimMessagesMiddleware,
)
from langgraph_agent_toolkit.core.models.fake import FakeToolModel
from langgraph_agent_toolkit.core.settings import settings


def _tool_call(call_id: str = "1") -> dict:
    return {"name": "add", "args": {"a": 1, "b": 2}, "id": call_id, "type": "tool_call"}


def test_immediate_generation_counts_model_calls_this_run():
    """Count AI messages after the latest human message."""
    mw = ImmediateGenerationMiddleware()
    messages = [
        HumanMessage("old turn"),
        AIMessage("old answer"),
        HumanMessage("this turn"),
        AIMessage("", tool_calls=[_tool_call("1")]),
        ToolMessage("3", tool_call_id="1"),
        AIMessage("", tool_calls=[_tool_call("2")]),
    ]
    assert mw._calls_made_this_run(messages) == 2


def test_immediate_generation_forces_tool_free_answer_at_budget():
    """Remove tools and add the instruction at the call limit."""
    mw = ImmediateGenerationMiddleware(model_call_limit=2)
    request = MagicMock()
    request.messages = [HumanMessage("q"), AIMessage("", tool_calls=[_tool_call()])]
    request.system_message = SystemMessage("You are helpful.")
    request.override.return_value = "OVERRIDDEN"
    handler = MagicMock(return_value="RESPONSE")

    result = mw.wrap_model_call(request, handler)

    kwargs = request.override.call_args.kwargs
    assert kwargs["tools"] == []
    assert "best final answer" in kwargs["system_message"].content
    assert "You are helpful." in kwargs["system_message"].content
    handler.assert_called_once_with("OVERRIDDEN")
    assert result == "RESPONSE"


def test_immediate_generation_default_budget_from_recursion_limit():
    """Set the default budget from the recursion limit."""
    mw = ImmediateGenerationMiddleware()
    assert mw.model_call_limit == max(1, settings.DEFAULT_RECURSION_LIMIT // 2)
    assert mw.model_call_limit > 5


def test_immediate_generation_passes_through_under_budget():
    """Forward the request unchanged below the call limit."""
    mw = ImmediateGenerationMiddleware(model_call_limit=5)
    request = MagicMock()
    request.messages = [HumanMessage("q")]
    handler = MagicMock(return_value="R")

    mw.wrap_model_call(request, handler)

    request.override.assert_not_called()
    handler.assert_called_once_with(request)


def test_immediate_generation_rejects_bad_limit():
    with pytest.raises(ValueError, match="model_call_limit"):
        ImmediateGenerationMiddleware(model_call_limit=0)


def test_sanitize_history_strips_incomplete_tool_calls():
    """Remove incomplete tool calls from model messages."""
    mw = SanitizeHistoryMiddleware()
    request = MagicMock()
    request.messages = [HumanMessage("q"), AIMessage("", tool_calls=[_tool_call("missing")])]
    request.override.return_value = "SANITIZED"
    handler = MagicMock(return_value="R")

    mw.wrap_model_call(request, handler)

    sanitized = request.override.call_args.kwargs["messages"]
    assert sanitized[-1].tool_calls == []
    handler.assert_called_once_with("SANITIZED")


def _ai_tool(name: str, call_id: str, args: dict | None = None) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": call_id, "type": "tool_call"}])


def _kept_tool_ids(mw: ClearIntermediateToolCallsMiddleware, messages: list) -> list[str]:
    out = mw._process(messages)
    kept = [m.tool_call_id for m in out if isinstance(m, ToolMessage)]
    ai_ids = {c["id"] for m in out if isinstance(m, AIMessage) and m.tool_calls for c in m.tool_calls}
    assert set(kept) == ai_ids
    return kept


def test_clear_intermediate_keeps_current_turn_dedups_previous():
    """Keep recent prior results and preserve the current turn."""
    mw = ClearIntermediateToolCallsMiddleware(by="name")
    messages = [
        HumanMessage("turn 1"),
        _ai_tool("search", "s1"),
        ToolMessage("old", tool_call_id="s1", name="search"),
        _ai_tool("search", "s2"),
        ToolMessage("newer", tool_call_id="s2", name="search"),
        AIMessage("answer 1"),
        HumanMessage("turn 2"),
        _ai_tool("search", "s3"),
        ToolMessage("current", tool_call_id="s3", name="search"),
    ]
    kept = _kept_tool_ids(mw, messages)
    assert kept == ["s2", "s3"]


def test_name_args_preserves_distinct_argument_calls():
    """Keep repeated calls with distinct arguments for `by="name_args"`."""
    mw = ClearIntermediateToolCallsMiddleware(by="name_args")
    messages = [
        HumanMessage("turn 1"),
        _ai_tool("search", "a", {"q": "tokyo"}),
        ToolMessage("tokyo", tool_call_id="a", name="search"),
        _ai_tool("search", "b", {"q": "paris"}),
        ToolMessage("paris", tool_call_id="b", name="search"),
        HumanMessage("turn 2"),
    ]
    assert sorted(_kept_tool_ids(mw, messages)) == ["a", "b"]


def test_name_collapses_distinct_argument_calls():
    """Collapse repeated calls for `by="name"`."""
    mw = ClearIntermediateToolCallsMiddleware(by="name")
    messages = [
        HumanMessage("turn 1"),
        _ai_tool("search", "a", {"q": "tokyo"}),
        ToolMessage("tokyo", tool_call_id="a", name="search"),
        _ai_tool("search", "b", {"q": "paris"}),
        ToolMessage("paris", tool_call_id="b", name="search"),
        HumanMessage("turn 2"),
    ]
    assert _kept_tool_ids(mw, messages) == ["b"]


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
    assert "s1" not in kept
    assert {"s2", "s3"}.issubset(set(kept))
    assert "w1" in kept


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
    """Read middleware settings when no arguments are set."""
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
    """Trim the model view to `max_messages` from a human turn."""
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
    request.messages = [HumanMessage("q1"), AIMessage("a1"), HumanMessage("q2")]
    handler = MagicMock()

    mw.wrap_model_call(request, handler)

    assert request.override.call_args.kwargs["messages"] == request.messages


def test_trim_messages_default_from_settings():
    assert TrimMessagesMiddleware().max_messages == settings.DEFAULT_MAX_MESSAGE_HISTORY_LENGTH


def test_trim_messages_rejects_bad_max():
    with pytest.raises(ValueError, match="max_messages"):
        TrimMessagesMiddleware(max_messages=0)


def _content_len(messages: list) -> int:
    """Count characters in message content."""
    return sum(len(m.content) for m in messages)


def test_token_trim_bounds_to_token_budget():
    """Trim the model view to the token budget from a human turn."""
    mw = TokenTrimMiddleware(max_tokens=12, token_counter=_content_len)
    request = MagicMock()
    request.messages = [
        HumanMessage("aaaa"),
        AIMessage("bbbb"),
        HumanMessage("cccc"),
        AIMessage("dddd"),
        HumanMessage("eeee"),
    ]
    request.override.return_value = "TRIMMED"
    handler = MagicMock(return_value="R")

    mw.wrap_model_call(request, handler)

    trimmed = request.override.call_args.kwargs["messages"]
    assert _content_len(trimmed) <= 12
    assert trimmed[0].type == "human"
    handler.assert_called_once_with("TRIMMED")


def test_token_trim_keeps_system_message():
    """Keep the system message when other messages are trimmed."""
    mw = TokenTrimMiddleware(max_tokens=8, token_counter=_content_len)
    request = MagicMock()
    request.messages = [SystemMessage("sys"), HumanMessage("aaaa"), AIMessage("bbbb"), HumanMessage("cccc")]
    handler = MagicMock()

    mw.wrap_model_call(request, handler)

    trimmed = request.override.call_args.kwargs["messages"]
    assert any(m.type == "system" for m in trimmed)


def test_token_trim_disabled_passes_through():
    """Keep messages unchanged when the budget is `None`."""
    mw = TokenTrimMiddleware(max_tokens=None)
    assert mw.max_tokens is None
    request = MagicMock()
    request.messages = [HumanMessage("q1"), AIMessage("a1"), HumanMessage("q2")]
    handler = MagicMock()

    mw.wrap_model_call(request, handler)

    assert request.override.call_args.kwargs["messages"] == request.messages


def test_token_trim_default_from_settings():
    assert TokenTrimMiddleware().max_tokens == settings.DEFAULT_MAX_TOKENS_HISTORY_LENGTH


def test_token_trim_rejects_bad_max():
    with pytest.raises(ValueError, match="max_tokens"):
        TokenTrimMiddleware(max_tokens=0)


def _latest_human_present(view: list, text: str) -> bool:
    return any(m.type == "human" and text in str(m.content) for m in view)


def test_token_trim_preserves_oversize_latest_human():
    """Keep an oversized user message."""
    mw = TokenTrimMiddleware(max_tokens=5, token_counter=_content_len)
    request = MagicMock()
    request.messages = [HumanMessage("this question is far longer than the tiny budget UNIQUE")]
    handler = MagicMock()

    mw.wrap_model_call(request, handler)

    kept = request.override.call_args.kwargs["messages"]
    assert _latest_human_present(kept, "UNIQUE")


def test_token_trim_preserves_oversize_tool_turn():
    """Keep an oversized current turn."""
    mw = TokenTrimMiddleware(max_tokens=5, token_counter=_content_len)
    request = MagicMock()
    request.messages = [
        HumanMessage("q UNIQUE"),
        _ai_tool("search", "s1"),
        ToolMessage("x" * 200, tool_call_id="s1", name="search"),
    ]
    handler = MagicMock()

    mw.wrap_model_call(request, handler)

    kept = request.override.call_args.kwargs["messages"]
    assert _latest_human_present(kept, "UNIQUE")
    ai_ids = {c["id"] for m in kept if m.type == "ai" and m.tool_calls for c in m.tool_calls}
    assert all(m.tool_call_id in ai_ids for m in kept if m.type == "tool")


def test_trim_messages_preserves_long_tool_burst_turn():
    """Keep a current turn that exceeds `max_messages`."""
    mw = TrimMessagesMiddleware(max_messages=12)
    burst = [HumanMessage("BURST_Q")]
    for i in range(6):
        burst += [_ai_tool("search", f"s{i}"), ToolMessage("r", tool_call_id=f"s{i}", name="search")]
    request = MagicMock()
    request.messages = burst
    handler = MagicMock()

    mw.wrap_model_call(request, handler)

    kept = request.override.call_args.kwargs["messages"]
    assert _latest_human_present(kept, "BURST_Q")
    ai_ids = {c["id"] for m in kept if m.type == "ai" and m.tool_calls for c in m.tool_calls}
    assert all(m.tool_call_id in ai_ids for m in kept if m.type == "tool")


def test_keep_latest_turn_floor_keeps_only_latest_turn_and_leading_system():
    """Keep leading system messages and the latest turn."""
    from langgraph_agent_toolkit.agents.components.middlewares._history import keep_latest_turn_if_emptied

    original = [SystemMessage("sys"), HumanMessage("old"), AIMessage("a"), HumanMessage("latest")]
    floored = keep_latest_turn_if_emptied(original, [])

    assert floored[0].type == "system"
    assert floored[-1].content == "latest"
    assert [m.content for m in floored if m.type == "human"] == ["latest"]


def test_keep_latest_turn_floor_passes_through_healthy_trim():
    """Return the trimmed messages when they contain user content."""
    from langgraph_agent_toolkit.agents.components.middlewares._history import keep_latest_turn_if_emptied

    original = [HumanMessage("a"), AIMessage("b"), HumanMessage("c")]
    trimmed = [HumanMessage("c")]
    assert keep_latest_turn_if_emptied(original, trimmed) is trimmed


def test_create_agent_with_middleware_runs_end_to_end():
    """Verify that ``create_agent`` composes the middleware."""

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
