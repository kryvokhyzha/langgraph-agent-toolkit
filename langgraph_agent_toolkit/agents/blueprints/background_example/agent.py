"""Prepare a value, ask for approval, and save a structured result."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from langgraph_agent_toolkit.agents.agent import Agent


class ExampleState(TypedDict, total=False):
    value: int
    prepared: int
    approved: bool
    result: int | None


async def prepare(state: ExampleState) -> dict:
    return {"prepared": state["value"] * 2}


async def review(state: ExampleState) -> dict:
    approved = interrupt({"question": "Accept this result?", "value": state["prepared"]})
    if not isinstance(approved, bool):
        raise ValueError("The approval answer must be a boolean")
    return {"approved": approved, "result": state["prepared"] if approved else None}


builder = StateGraph(ExampleState)
builder.add_node("prepare", prepare)
builder.add_node("review", review)
builder.add_edge(START, "prepare")
builder.add_edge("prepare", "review")
builder.add_edge("review", END)
agent = Agent("background-example", "Prepare and review a structured result.", builder.compile())
