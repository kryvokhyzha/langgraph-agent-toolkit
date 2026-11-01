"""Use deterministic graph nodes to test durable background execution."""

import asyncio
import os
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from langgraph_agent_toolkit.agents.agent import Agent


class WorkState(TypedDict, total=False):
    mode: str
    marker: str
    value: str
    prepared: str
    output: str


def record(marker: str, event: str) -> None:
    """Flush node events before a test terminates the worker process."""
    directory = Path(os.environ["LAT_TEST_BACKGROUND_DIRECTORY"])
    with (directory / f"{marker}.events").open("a") as events:
        events.write(event + "\n")


async def prepare(state: WorkState) -> dict[str, str]:
    record(state["marker"], "prepared")
    return {"prepared": f"prepared:{state['value']}"}


async def perform(state: WorkState) -> dict[str, str]:
    if state["mode"] == "approval":
        answer = interrupt({"question": "Approve this operation?", "value": state["value"]})
        return {"output": f"{state['prepared']}; answer={answer}"}
    if state["mode"] == "blocked":
        directory = Path(os.environ["LAT_TEST_BACKGROUND_DIRECTORY"])
        record(state["marker"], "waiting")
        try:
            while not (directory / f"{state['marker']}.release").exists():
                await asyncio.sleep(0.03)
        except asyncio.CancelledError:
            record(state["marker"], "cancelled")
            raise
    record(state["marker"], "completed")
    return {"output": state["prepared"]}


builder = StateGraph(WorkState)
builder.add_node("prepare", prepare)
builder.add_node("perform", perform)
builder.add_edge(START, "prepare")
builder.add_edge("prepare", "perform")
builder.add_edge("perform", END)
background_agent = Agent(
    name="background-agent",
    description="A deterministic graph for background worker process tests.",
    graph=builder.compile(),
)
