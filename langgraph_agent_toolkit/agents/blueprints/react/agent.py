from collections.abc import Sequence

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.components.creators.create_react_agent import create_react_agent
from langgraph_agent_toolkit.agents.components.tools import add, multiply
from langgraph_agent_toolkit.agents.components.utils import AgentStateWithRemainingSteps, pre_model_hook_standard
from langgraph_agent_toolkit.core.mcp import merge_tools
from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.schema.models import ModelProvider


PROMPT_NAME = "react-assistant"


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a team support agent that can perform calculations and search the web. "
                "You can use the tools provided to help you with your tasks. "
                "You can also ask clarifying questions to the user. "
            ),
        ),
        ("placeholder", "{messages}"),
    ]
)


def build_graph(extra_tools: Sequence[BaseTool] = ()) -> Pregel:
    """Build the custom ReAct graph with supplied tools."""
    return create_react_agent(
        model=CompletionModelFactory.create(
            model_provider=ModelProvider.FAKE if settings.USE_FAKE_MODEL else ModelProvider.OPENAI,
            model_name=settings.OPENAI_MODEL_NAME,
            openai_api_base=settings.OPENAI_API_BASE_URL,
            openai_api_key=settings.OPENAI_API_KEY,
        ),
        tools=merge_tools([add, multiply, DuckDuckGoSearchResults()], extra_tools),
        prompt=prompt,
        pre_model_hook=pre_model_hook_standard,
        state_schema=AgentStateWithRemainingSteps,
        checkpointer=None,
        immediate_step_threshold=5,
    )


react_agent = Agent(
    name="react-agent",
    description="A react agent.",
    graph=build_graph(),
    graph_factory=build_graph,
)

__all__ = ["build_graph", "react_agent"]
