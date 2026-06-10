"""Example agent built with LangChain's native ``create_agent`` (the recommended modern pattern).

A ``create_agent`` tool-calling loop composed with middleware. It reproduces the toolkit's custom
``create_react_agent`` behavior using only public LangChain middleware:

- ``ContextEditingMiddleware`` — bound the context window non-destructively: once tokens pass the
  trigger, clear OLD tool outputs (keeping the most recent), reclaiming tokens while keeping the
  conversation text. Chosen over ``SummarizationMiddleware`` (which destructively replaces history
  with a summary).
- ``ImmediateGenerationMiddleware`` — the graceful loop bound: near the recursion limit, drop tools
  and ask the model to answer from what it already gathered, instead of stalling or erroring (ports
  the custom router's ``immediate_generation``). This is the loop bound — a separate hard tool-call
  cap is intentionally omitted, since it would end abruptly and preempt this graceful synthesis.
- ``ClearIntermediateToolCallsMiddleware`` — token reduction: keep only the latest result per tool
  from earlier turns (configurable via the ``CLEAR_INTERMEDIATE_TOOL_CALLS*`` settings).
- ``SanitizeHistoryMiddleware`` — repair broken tool-call/result pairing (interrupted runs, post
  trim/clear orphans).
- ``ToolRetryMiddleware`` — resilience: retry a failed tool call with backoff, and after retries are
  exhausted return the error as a ToolMessage so a single flaky tool (e.g. a search timeout) doesn't
  abort the whole run.

Compare with ``blueprints/react``: the same agent built with the toolkit's ``create_react_agent``.
"""

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware, ToolRetryMiddleware
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.checkpoint.memory import MemorySaver

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.components.middlewares import (
    ClearIntermediateToolCallsMiddleware,
    ImmediateGenerationMiddleware,
    SanitizeHistoryMiddleware,
)
from langgraph_agent_toolkit.agents.components.tools import add, multiply
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory
from langgraph_agent_toolkit.schema.models import ModelProvider


model = CompletionModelFactory.create(
    model_provider=ModelProvider.OPENAI,
    model_name=settings.OPENAI_MODEL_NAME,
    config_prefix="",
    configurable_fields=(),
    model_parameter_values=(("temperature", 0.2), ("top_p", 0.95), ("streaming", False)),
    openai_api_base=settings.OPENAI_API_BASE_URL,
    openai_api_key=settings.OPENAI_API_KEY,
)

react_agent = Agent(
    name="create-agent",
    description="A tool-calling ReAct agent built with LangChain's native create_agent + middleware.",
    graph=create_agent(
        model=model,
        tools=[add, multiply, DuckDuckGoSearchResults()],
        # Middleware compose in list order. The wrap_model_call-based ones (Immediate /
        # ClearIntermediate / Sanitize) wrap the model node rather than adding graph nodes;
        # SanitizeHistory is placed last so it cleans the final request sent to the model.
        middleware=[
            # Bound the context window without summarizing: once the conversation passes the token
            # trigger, clear OLD tool outputs (keeping the most recent `keep`), reclaiming tokens
            # while preserving conversation text. Tune `trigger` to the model's context window.
            ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=100_000, keep=3)]),
            # Graceful loop bound near the recursion limit: on the last allowed model call, strip
            # tools and ask the model to answer from what it already gathered (ports the custom
            # create_react_agent's immediate_generation router). Default budget ~= recursion_limit/2.
            # No separate hard tool-call cap — it would end abruptly and preempt this synthesis.
            ImmediateGenerationMiddleware(),
            # Token reduction: keep only the latest result per tool from earlier turns (current turn
            # untouched). Defaults from CLEAR_INTERMEDIATE_TOOL_CALLS* settings (safe name_args key).
            ClearIntermediateToolCallsMiddleware(),
            # Repair broken tool-call/result pairing before the model call (interrupted runs, post
            # trim/summarize orphans).
            SanitizeHistoryMiddleware(),
            # Resilience: retry a failed tool call (backoff), then on_failure="continue" (the default)
            # returns the error as a ToolMessage so one flaky search timeout doesn't abort the run.
            ToolRetryMiddleware(max_retries=2),
        ],
        system_prompt=(
            "You are a team support agent that can perform calculations and search the web. "
            "You can use the tools provided to help you with your tasks. "
            "You can also ask clarifying questions to the user. "
        ),
        state_schema=AgentState,
        checkpointer=MemorySaver(),
    ),
)

__all__ = ["react_agent"]
