import asyncio
from uuid import UUID
import random

from dotenv import load_dotenv, find_dotenv
from langchain_core.runnables import RunnableConfig
from langfuse.callback import CallbackHandler

load_dotenv(find_dotenv())

from langgraph_agent_toolkit.helper.constants import DEFAULT_AGENT
from langgraph_agent_toolkit.agents.agents import get_agent
from langgraph_agent_toolkit.helper.logging import logger

# agent = get_agent("react-agent").graph
# agent = get_agent("react-agent-so").graph
agent = get_agent(DEFAULT_AGENT).graph


async def main() -> None:
    inputs = {"messages": [("user", "Find me a recipe for chocolate chip cookies")]}

    rd = random.Random()
    rd.seed(0)
    thread_id = UUID(int=rd.getrandbits(128), version=4)

    logger.info(f"Starting agent with thread {thread_id}")
    handler = CallbackHandler(session_id=str(thread_id))

    result = await agent.ainvoke(
        inputs,
        config=RunnableConfig(configurable={"thread_id": thread_id}, callbacks=[handler]),
    )
    logger.info(result.keys())
    logger.info(result)

    try:
        result["messages"][-1].pretty_print()
    except Exception as _:
        logger.warning("Can't print pretty following message.")

    # Draw the agent graph as png
    # requires:
    # brew install graphviz
    # export CFLAGS="-I $(brew --prefix graphviz)/include"
    # export LDFLAGS="-L $(brew --prefix graphviz)/lib"
    # pip install pygraphviz
    #
    # agent.get_graph().draw_png("agent_diagram.png")


if __name__ == "__main__":
    asyncio.run(main())
