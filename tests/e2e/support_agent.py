"""Run deterministic model calls through the real API.

The user store is in memory. The service stores thread checkpoints in SQLite.
"""

import asyncio

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt

from langgraph_agent_toolkit.agents.agent import Agent


async def reply(state: MessagesState, config: RunnableConfig, runtime: Runtime):
    """Read the user's preference and return a deterministic model response."""
    messages = [message for message in state["messages"] if isinstance(message, HumanMessage)]
    text = messages[-1].content
    if text == "fail":
        raise RuntimeError("E2E internal failure detail")
    if text == "pause":
        response = interrupt("Choose the next action.")
        text = f"resumed:{response['message']}"
    if text.startswith("slow:"):
        await asyncio.sleep(0.05)
    namespace = ("users", config["configurable"]["user_id"], "preferences")
    if text.startswith("remember:"):
        await runtime.store.aput(namespace, "drink", {"value": text.removeprefix("remember:")})
    preference = await runtime.store.aget(namespace, "drink")
    drink = preference.value["value"] if preference else "none"
    content = f"turn={len(messages)}; preference={drink}; message={text}"
    model = FakeListChatModel(responses=[content])
    response = await model.ainvoke([HumanMessage(text)], config=config)
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("reply", reply)
builder.add_edge(START, "reply")
builder.add_edge("reply", END)
journey_agent = Agent(
    name="journey-agent",
    description="A deterministic agent for API journeys.",
    graph=builder.compile(store=InMemoryStore()),
)
