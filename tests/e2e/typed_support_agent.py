"""Run typed approvals and record completed effects for process tests."""

import json
import os
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from langgraph_agent_toolkit.agents.agent import Agent


class Quantity(BaseModel):
    quantity: int = Field(ge=1, le=5, strict=True)


def record(event: str, **values) -> None:
    """Append one completed effect to the test log."""
    with Path(os.environ["LAT_TEST_TYPED_EVENTS"]).open("a") as output:
        output.write(json.dumps({"event": event, **values}) + "\n")


def committed(state):
    record("committed")
    return {"messages": [AIMessage("The initial write is complete.")]}


def choose(state, label):
    answer = interrupt({"label": label, "question": "Choose a quantity."}, response_schema=Quantity)
    record("answered", label=label, quantity=answer.quantity)
    return {"messages": [AIMessage(f"Quantity for {label}: {answer.quantity}.")]}


def first(state):
    return choose(state, "first")


def second(state):
    return choose(state, "second")


builder = StateGraph(MessagesState)
for name, node in (("committed", committed), ("first", first), ("second", second)):
    builder.add_node(name, node)
    builder.add_edge(START, name)
    builder.add_edge(name, END)
typed_agent = Agent("typed-journey", "Typed approvals for process tests.", builder.compile())
