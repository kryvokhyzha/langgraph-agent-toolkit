import asyncio
import random
from uuid import UUID

import rootutils
from dotenv import find_dotenv, load_dotenv
from langchain_core.runnables import RunnableConfig


try:
    from langfuse.callback import CallbackHandler
except (ModuleNotFoundError, ImportError):
    from langfuse.langchain import CallbackHandler


_ = rootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    pythonpath=True,
    dotenv=False,
)
load_dotenv(find_dotenv(".local.env"), override=True)

from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.constants import DEFAULT_AGENT
from langgraph_agent_toolkit.helper.logging import logger


# agent = AgentExecutor(*settings.AGENT_PATHS).get_agent("react-agent").graph
# agent = AgentExecutor(*settings.AGENT_PATHS).get_agent("react-agent-so").graph
agent = AgentExecutor(*settings.AGENT_PATHS).get_agent(DEFAULT_AGENT).graph


async def main() -> None:
    inputs = {
        "messages": [("user", "Find me a recipe for chocolate chip cookies and how Dynamo Kyiv played last game")]
    }

    rd = random.Random()
    rd.seed(0)
    thread_id = UUID(int=rd.getrandbits(128), version=4)

    logger.info(f"Starting agent with thread {thread_id}")
    handler = CallbackHandler(session_id=str(thread_id))

    result = await agent.ainvoke(
        inputs,
        config=RunnableConfig(
            configurable={
                "thread_id": thread_id,
                "agent_temperature": 0.9,
                "agent_top_p": 0.75,
                "agent_max_tokens": 512,
                "checkpointer_params": {"k": 3},
            },
            callbacks=[handler],
            recursion_limit=15,
        ),
    )
    logger.info(result.keys())
    # logger.info(result)

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
