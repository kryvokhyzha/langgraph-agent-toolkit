from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint
from langgraph.graph import add_messages

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.core.models.factory import ModelFactory


@entrypoint(
    # checkpointer=MemorySaver(),  # Uncomment if you want to save the state of the agent
)
async def chatbot(
    inputs: dict[str, list[BaseMessage]],
    *,
    previous: dict[str, list[BaseMessage]],
    config: RunnableConfig,
):
    messages = inputs["messages"]
    if previous:
        messages = previous["messages"] + messages

    model = ModelFactory.create(config["configurable"].get("model", settings.DEFAULT_MODEL_TYPE))
    response = await model.ainvoke(messages)
    return entrypoint.final(value={"messages": [response]}, save={"messages": messages + [response]})


chatbot_agent = Agent(
    name="chatbot-agent",
    description="A simple chatbot.",
    graph=chatbot,
)
