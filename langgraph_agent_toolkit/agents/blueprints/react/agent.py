from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.checkpoint.memory import MemorySaver

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.components.creators.create_react_agent import create_react_agent
from langgraph_agent_toolkit.agents.components.tools import add, multiply
from langgraph_agent_toolkit.agents.components.utils import AgentStateWithRemainingSteps, pre_model_hook_standard
from langgraph_agent_toolkit.core.models.factory import ModelFactory
from langgraph_agent_toolkit.core.observability.factory import ObservabilityFactory
from langgraph_agent_toolkit.core.observability.types import ChatMessageDict, MessageRole, ObservabilityBackend
from langgraph_agent_toolkit.core.prompts.chat_prompt_template import ObservabilityChatPromptTemplate
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.schema.models import ModelProvider


PROMPT_NAME = "react-assistant"


# observability = ObservabilityFactory.create(settings.OBSERVABILITY_BACKEND)
observability = ObservabilityFactory.create(ObservabilityBackend.LANGFUSE)

# Push the prompt template to the observability platform first
observability.push_prompt(
    name=PROMPT_NAME,
    prompt_template=[
        ChatMessageDict(
            role=MessageRole.SYSTEM,
            content=(
                "You are a team support agent that can perform calculations and search the web. "
                "You can use the tools provided to help you with your tasks. "
                "You can also ask clarifying questions to the user. "
            ),
        ),
        ChatMessageDict(
            role=MessageRole.PLACEHOLDER,
            content="messages",
        ),
    ],
    create_new_version=False,
)

prompt = ObservabilityChatPromptTemplate.from_observability_platform(
    prompt_name=PROMPT_NAME,
    observability_platform=observability,
    load_at_runtime=True,
    template_format="jinja2",
    input_variables=["messages"],
)

react_agent = Agent(
    name="react-agent",
    description="A react agent.",
    graph=create_react_agent(
        model=ModelFactory.create(
            model_provider=ModelProvider.OPENAI,
            model_name=settings.OPENAI_MODEL_NAME,
            openai_api_base=settings.OPENAI_API_BASE_URL,
            openai_api_key=settings.OPENAI_API_KEY,
        ),
        tools=[add, multiply, DuckDuckGoSearchResults()],
        prompt=prompt,
        pre_model_hook=pre_model_hook_standard,
        state_schema=AgentStateWithRemainingSteps,
        checkpointer=MemorySaver(),
        immediate_step_threshold=5,
    ),
    observability=observability,
)

__all__ = ["react_agent"]
