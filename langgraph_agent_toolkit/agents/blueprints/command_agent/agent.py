import random
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.types import Command

from langgraph_agent_toolkit.agents.agent import Agent


class AgentState(MessagesState, total=False):
    pass


def node_a(state: AgentState) -> Command[Literal["node_b", "node_c"]]:
    print("Called A")
    value = random.choice(["a", "b"])
    if value == "a":
        goto = "node_b"
    else:
        goto = "node_c"

    return Command(
        update={"messages": [AIMessage(content=f"Hello {value}")]},
        goto=goto,
    )


def node_b(state: AgentState):
    print("Called B")
    return {"messages": [AIMessage(content="Hello B")]}


def node_c(state: AgentState):
    print("Called C")
    return {"messages": [AIMessage(content="Hello C")]}


builder = StateGraph(AgentState)
builder.add_edge(START, "node_a")
builder.add_node(node_a)
builder.add_node(node_b)
builder.add_node(node_c)

command_agent = Agent(
    name="command-agent",
    description="A command agent.",
    graph=builder.compile(),
)
