"""Example agent built with LangChain's native ``create_agent`` that returns structured output.

Same builder as ``blueprints/create_agent``, plus a ``response_format`` (a Pydantic schema): the
agent runs its tool-calling loop and then produces a validated ``structured_response``. Compare with
``blueprints/react`` for the toolkit's custom ``create_react_agent`` builder.
"""

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware, ToolRetryMiddleware
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

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


class ResponseSchema(BaseModel):
    response: str = Field(
        description="The response on user query.",
    )
    alternative_response: str = Field(
        description="The alternative response on user query.",
    )


model = CompletionModelFactory.create(
    model_provider=ModelProvider.OPENAI,
    model_name=settings.OPENAI_MODEL_NAME,
    config_prefix="",
    configurable_fields=(),
    model_parameter_values=(("temperature", 0.2), ("top_p", 0.95), ("streaming", False)),
    openai_api_base=settings.OPENAI_API_BASE_URL,
    openai_api_key=settings.OPENAI_API_KEY,
)

react_agent_so = Agent(
    name="create-agent-structured",
    description="A create_agent ReAct agent that returns structured output (response_format).",
    graph=create_agent(
        model=model,
        tools=[add, multiply, DuckDuckGoSearchResults()],
        # Same stack as blueprints/create_agent (see its docstring for the rationale of each).
        middleware=[
            # Bound the context window non-destructively: clear old tool outputs once the
            # conversation passes the token trigger (keeps conversation text, unlike summarization).
            ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=100_000, keep=3)]),
            # Graceful loop bound near the recursion limit (no separate hard tool-call cap).
            ImmediateGenerationMiddleware(),
            # Token reduction: keep only the latest result per tool from earlier turns.
            ClearIntermediateToolCallsMiddleware(),
            # Repair broken tool-call/result pairing before the model call.
            SanitizeHistoryMiddleware(),
            # Resilience: retry failed tool calls; on exhaustion return the error as a ToolMessage
            # so one flaky tool doesn't abort the run.
            ToolRetryMiddleware(max_retries=2),
        ],
        system_prompt=(
            "You are a team support agent that can perform calculations and search the web. "
            "You can use the tools provided to help you with your tasks. "
            "You can also ask clarifying questions to the user. "
        ),
        # pre_model_hook=pre_model_hook_standard,
        response_format=ResponseSchema,
        state_schema=AgentState,
        checkpointer=MemorySaver(),
    ),
)
