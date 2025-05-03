import sys

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.func import Pregel


load_dotenv(find_dotenv(filename=".local.env"), override=True)
sys.path.append(".")

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.blueprints.react.agent import react_agent
from langgraph_agent_toolkit.core.memory.factory import MemoryFactory


async def main():
    # thread_id = "847c6285-8fc9-4560-a83f-4e6285809254"
    thread_id = "07afd798-8d98-4250-a730-6b609abded9d"
    # user_id = "521c0a60-ea75-43fa-a793-a4cf11e013ae"

    agent: Agent = react_agent
    agent_graph: Pregel = agent.graph

    memory_backend = MemoryFactory.create("postgres")

    checkpoint = memory_backend.get_checkpoint_saver()
    async with checkpoint as saver:
        await saver.setup()

        agent.graph.checkpointer = saver

        await agent_graph.aupdate_state(
            values={
                "messages": [
                    {
                        "type": "human",
                        "content": "Hello, how are you?",
                    },
                    {
                        "type": "ai",
                        "content": "I'm doing well, thank you! How can I assist you today?",
                    },
                ]
            },
            config=RunnableConfig(
                configurable={
                    "thread_id": thread_id,
                    # "user_id": user_id,
                }
            ),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
