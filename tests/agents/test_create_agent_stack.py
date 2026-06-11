"""Integration / regression tests for the create_agent middleware stack.

Uses a scripted fake model (deterministic, no real LLM) so the full middleware stack runs the actual
tool-calling loop in CI — the seam where manual real-model testing kept finding interaction bugs.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware, ToolRetryMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langgraph_agent_toolkit.agents.components.middlewares import (
    ClearIntermediateToolCallsMiddleware,
    ImmediateGenerationMiddleware,
    SanitizeHistoryMiddleware,
)


class ToolScriptModel(FakeMessagesListChatModel):
    """Return scripted AIMessages in order; no-op bind_tools so create_agent can drive the loop."""

    def bind_tools(self, tools, **kwargs):
        return self


def _flagship_stack():
    """Build the exact create_agent flagship stack (fast ToolRetry: no real sleeping)."""
    return [
        ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=100_000, keep=3)]),
        ImmediateGenerationMiddleware(),
        ClearIntermediateToolCallsMiddleware(),
        SanitizeHistoryMiddleware(),
        ToolRetryMiddleware(max_retries=2, initial_delay=0.0, backoff_factor=0.0),
    ]


@tool
def search(query: str) -> str:
    """Search the web."""
    return f"result for {query}"


def test_full_stack_runs_tool_loop():
    """The flagship 5-middleware stack composes and runs a tool-call -> result -> answer loop."""
    model = ToolScriptModel(
        responses=[
            AIMessage(
                content="", tool_calls=[{"name": "search", "args": {"query": "x"}, "id": "c1", "type": "tool_call"}]
            ),
            AIMessage(content="here is the answer"),
        ]
    )
    agent = create_agent(model=model, tools=[search], middleware=_flagship_stack())
    out = agent.invoke({"messages": [HumanMessage("go")]})

    assert any(isinstance(m, ToolMessage) for m in out["messages"])  # the tool actually executed
    assert out["messages"][-1].content == "here is the answer"


def test_tool_failure_does_not_abort_run():
    """Regression (session b94a3dab): a tool that always raises is retried, then the run continues."""
    calls = {"n": 0}

    @tool
    def flaky(query: str) -> str:
        """Flaky search."""
        calls["n"] += 1
        raise TimeoutError("boom")

    model = ToolScriptModel(
        responses=[
            AIMessage(
                content="", tool_calls=[{"name": "flaky", "args": {"query": "x"}, "id": "c1", "type": "tool_call"}]
            ),
            AIMessage(content="answered despite the failure"),
        ]
    )
    agent = create_agent(model=model, tools=[flaky], middleware=_flagship_stack())

    out = agent.invoke({"messages": [HumanMessage("go")]})  # must NOT raise

    assert calls["n"] == 3  # initial call + 2 retries
    assert any(isinstance(m, ToolMessage) for m in out["messages"])  # error surfaced as a ToolMessage
    assert out["messages"][-1].content == "answered despite the failure"  # run completed, not aborted


def test_stack_preserves_a_clean_multi_tool_turn():
    """Regression: the stack does not drop current-turn tool results (the WC-2026 failure mode)."""
    model = ToolScriptModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search", "args": {"query": "a"}, "id": "c1", "type": "tool_call"},
                    {"name": "search", "args": {"query": "b"}, "id": "c2", "type": "tool_call"},
                ],
            ),
            AIMessage(content="synthesized from both results"),
        ]
    )
    agent = create_agent(model=model, tools=[search], middleware=_flagship_stack())
    out = agent.invoke({"messages": [HumanMessage("search a and b")]})

    tool_results = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_results) == 2  # both current-turn results survived to the answer
    assert out["messages"][-1].content == "synthesized from both results"
