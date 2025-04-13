from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.blueprints.tools import add, multiply
from langgraph_agent_toolkit.agents.blueprints.utils import pre_model_hook_standard
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.core.models.factory import ModelFactory


react_agent = Agent(
    name="react-agent",
    description="A react agent.",
    graph=create_react_agent(
        model=ModelFactory.create(settings.DEFAULT_MODEL),
        tools=[add, multiply, DuckDuckGoSearchResults()],
        prompt=(
            "You are a team support agent that can perform calculations and search the web. "
            "You can use the tools provided to help you with your tasks. "
            "You can also ask clarifying questions to the user. "
        ),
        pre_model_hook=pre_model_hook_standard,
        checkpointer=MemorySaver(),
    ),
)
