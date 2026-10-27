"""Run parallel MCP tools through the configured service connection manager."""

from collections.abc import Sequence

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph_agent_toolkit.agents.agent import Agent


def build_graph(extra_tools: Sequence[BaseTool] = ()):
    builder = StateGraph(MessagesState)
    for label in ("first", "second"):

        async def node(state, label=label):
            tool = extra_tools[0]
            call = {
                "name": tool.name,
                "args": {"label": label, "tenant": "test-client"},
                "id": f"call-{label}",
                "type": "tool_call",
            }
            result = await tool.ainvoke(call)
            return {"messages": [AIMessage("", tool_calls=[call]), result]}

        builder.add_node(label, node)
        builder.add_edge(START, label)
    builder.add_node("finish", lambda state: {"messages": [AIMessage("Complete.")]})
    builder.add_edge(["first", "second"], "finish")
    builder.add_edge("finish", END)
    return builder.compile()


mcp_agent = Agent("mcp-journey", "A deterministic agent for MCP recovery.", build_graph(), graph_factory=build_graph)
