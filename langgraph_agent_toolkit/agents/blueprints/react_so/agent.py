from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.components.creators.create_react_agent import create_react_agent
from langgraph_agent_toolkit.agents.components.tools import add, multiply
from langgraph_agent_toolkit.agents.components.utils import AgentStateWithRemainingSteps, pre_model_hook_standard
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.core.models.factory import ModelFactory


class ResponseSchema(BaseModel):
    response: list[str] = Field(
        description="The response on user query.",
    )
    alternative_response: str = Field(
        description="The alternative response on user query.",
    )


react_agent_so = Agent(
    name="react-agent-so",
    description="A react agent with structured output.",
    graph=create_react_agent(
        model=ModelFactory.create(settings.DEFAULT_MODEL_TYPE),
        tools=[add, multiply, DuckDuckGoSearchResults()],
        prompt=(
            "You are a team support agent that can perform calculations and search the web. "
            "You can use the tools provided to help you with your tasks. "
            "You can also ask clarifying questions to the user. "
        ),
        pre_model_hook=pre_model_hook_standard,
        response_format=ResponseSchema,
        state_schema=AgentStateWithRemainingSteps,
        checkpointer=MemorySaver(),
        immediate_step_threshold=5,
    ),
)
