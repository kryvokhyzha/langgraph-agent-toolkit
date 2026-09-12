"""Example ``create_agent`` agent that returns structured output."""

from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel
from pydantic import BaseModel, Field

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.blueprints.create_agent._shared import build_tool_graph, create_model


class ResponseSchema(BaseModel):
    response: str = Field(
        description="The response on user query.",
    )
    alternative_response: str = Field(
        description="The alternative response on user query.",
    )


model = create_model()


def build_graph(extra_tools: Sequence[BaseTool] = ()) -> Pregel:
    """Build the graph with local tools and supplied tools."""
    return build_tool_graph(model, extra_tools, response_format=ResponseSchema)


react_agent_so = Agent(
    name="create-agent-structured",
    description="A create_agent ReAct agent that returns structured output (response_format).",
    graph=build_graph(),
    graph_factory=build_graph,
)

__all__ = ["build_graph", "react_agent_so"]
