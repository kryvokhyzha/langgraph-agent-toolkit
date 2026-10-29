"""Example ``create_agent`` agent that returns structured output."""

from collections.abc import Sequence

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel
from pydantic import BaseModel, Field

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.blueprints.create_agent._shared import build_tool_graph, create_model
from langgraph_agent_toolkit.core.tool_selection import ToolSelectionConfig


class ResponseSchema(BaseModel):
    response: str = Field(
        description="The response on user query.",
    )
    alternative_response: str = Field(
        description="The alternative response on user query.",
    )


model = create_model()


def build_graph(
    extra_tools: Sequence[BaseTool] = (),
    *,
    tool_selection: ToolSelectionConfig | None = None,
    selector_model: BaseChatModel | None = None,
    extra_middleware: Sequence[AgentMiddleware] = (),
) -> Pregel:
    """Build the graph with local tools and supplied tools."""
    return build_tool_graph(
        model,
        extra_tools,
        response_format=ResponseSchema,
        tool_selection=tool_selection,
        selector_model=selector_model,
        extra_middleware=extra_middleware,
    )


react_agent_so = Agent(
    name="create-agent-structured",
    description="A create_agent ReAct agent that returns structured output (response_format).",
    graph=build_graph(),
    graph_factory=build_graph,
)

__all__ = ["build_graph", "react_agent_so"]
