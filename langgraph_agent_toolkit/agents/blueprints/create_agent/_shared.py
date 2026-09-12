"""Build the common model and middleware for native tool agents."""

from collections.abc import Sequence

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware, ToolRetryMiddleware
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel
from pydantic import BaseModel

from langgraph_agent_toolkit.agents.components.middlewares import (
    ClearIntermediateToolCallsMiddleware,
    ImmediateGenerationMiddleware,
    SanitizeHistoryMiddleware,
)
from langgraph_agent_toolkit.agents.components.tools import add, multiply
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.core.mcp import merge_tools
from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory, ModelT
from langgraph_agent_toolkit.schema.models import ModelProvider


def create_model() -> ModelT:
    """Create a separate model for each blueprint."""
    return CompletionModelFactory.create(
        model_provider=ModelProvider.FAKE if settings.USE_FAKE_MODEL else ModelProvider.OPENAI,
        model_name=settings.OPENAI_MODEL_NAME,
        config_prefix="",
        configurable_fields=(),
        model_parameter_values=(("temperature", 0.2), ("top_p", 0.95), ("streaming", False)),
        openai_api_base=settings.OPENAI_API_BASE_URL,
        openai_api_key=settings.OPENAI_API_KEY,
    )


def build_tool_graph(
    model: ModelT,
    extra_tools: Sequence[BaseTool] = (),
    *,
    response_format: type[BaseModel] | None = None,
) -> Pregel:
    """Build a fresh graph and restrict automatic retries to local tools."""
    local_tools = [add, multiply, DuckDuckGoSearchResults()]
    return create_agent(
        model=model,
        tools=merge_tools(local_tools, extra_tools),
        middleware=[
            ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=100_000, keep=3)]),
            ImmediateGenerationMiddleware(),
            ClearIntermediateToolCallsMiddleware(),
            SanitizeHistoryMiddleware(),
            ToolRetryMiddleware(max_retries=2, tools=local_tools),
        ],
        system_prompt=(
            "You are a team support agent that can perform calculations and search the web. "
            "You can use the tools provided to help you with your tasks. "
            "You can also ask clarifying questions to the user. "
        ),
        response_format=response_format,
        state_schema=AgentState,
        checkpointer=None,
    )
