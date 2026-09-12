"""Example ``create_agent`` agent with tool approval."""

from collections.abc import Sequence

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import BaseTool, tool
from langgraph.pregel import Pregel

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.components.middlewares import (
    SanitizeHistoryMiddleware,
    TrimMessagesMiddleware,
)
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.core.mcp import merge_tools
from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory
from langgraph_agent_toolkit.schema.models import ModelProvider


@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email after human approval."""
    return f"Email sent to {recipient} (subject: {subject!r})."


model = CompletionModelFactory.create(
    model_provider=ModelProvider.FAKE if settings.USE_FAKE_MODEL else ModelProvider.OPENAI,
    model_name=settings.OPENAI_MODEL_NAME,
    openai_api_base=settings.OPENAI_API_BASE_URL,
    openai_api_key=settings.OPENAI_API_KEY,
)


def build_graph(extra_tools: Sequence[BaseTool] = ()) -> Pregel:
    """Build the graph with email approval and supplied tools."""
    return create_agent(
        model=model,
        tools=merge_tools([send_email], extra_tools),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"send_email": True},
                description_prefix="The assistant wants to send an email and needs your approval",
            ),
            TrimMessagesMiddleware(),
            SanitizeHistoryMiddleware(),
        ],
        system_prompt=(
            "You are a helpful assistant that can send emails on the user's behalf using the "
            "send_email tool. Call the tool whenever the user asks to send an email."
        ),
        state_schema=AgentState,
        checkpointer=None,
    )


hitl_agent = Agent(
    name="hitl-agent",
    description="A create_agent assistant that requires human approval before sending email.",
    graph=build_graph(),
    graph_factory=build_graph,
)

__all__ = ["build_graph", "hitl_agent"]
