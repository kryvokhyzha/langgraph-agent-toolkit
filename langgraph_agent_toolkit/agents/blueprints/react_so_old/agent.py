from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.prebuilt.chat_agent_executor import AgentStateWithStructuredResponse
from pydantic import BaseModel, Field

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.components.creators.create_react_agent import create_react_agent
from langgraph_agent_toolkit.agents.components.tools import add, multiply
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory
from langgraph_agent_toolkit.core.observability.factory import ObservabilityFactory
from langgraph_agent_toolkit.core.observability.types import ChatMessageDict, MessageRole, ObservabilityBackend
from langgraph_agent_toolkit.core.prompts.chat_prompt_template import ObservabilityChatPromptTemplate
from langgraph_agent_toolkit.helper.constants import DEFAULT_MAX_MESSAGE_HISTORY_LENGTH
from langgraph_agent_toolkit.schema.models import ModelProvider


class AgentStateWithStructuredResponseAndRemainingSteps(AgentStateWithStructuredResponse):
    remaining_steps: RemainingSteps


class ResponseSchema(BaseModel):
    response: str = Field(
        description="The response on user query.",
    )
    alternative_response: str = Field(
        description="The alternative response on user query.",
    )


PROMPT_NAME = "react-assistant-so-old"

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
    force_create_new_version=False,
)

prompt = ObservabilityChatPromptTemplate.from_observability_platform(
    prompt_name=PROMPT_NAME,
    observability_platform=observability,
    load_at_runtime=True,
    template_format="jinja2",
    input_variables=["messages"],
)

model = CompletionModelFactory.create(
    model_provider=ModelProvider.OPENAI,
    model_name=settings.OPENAI_MODEL_NAME,
    config_prefix="",
    configurable_fields=(),
    model_parameter_values=(("temperature", 0.2), ("top_p", 0.95), ("streaming", False)),
    openai_api_base=settings.OPENAI_API_BASE_URL,
    openai_api_key=settings.OPENAI_API_KEY,
)

react_agent_so = Agent(
    name="react-agent-so-old",
    description="A react agent with structured output.",
    graph=create_react_agent(
        model=model,
        tools=[add, multiply, DuckDuckGoSearchResults()],
        prompt=prompt,
        # pre_model_hook=pre_model_hook_standard,
        response_format=ResponseSchema,
        state_schema=AgentStateWithStructuredResponseAndRemainingSteps,
        checkpointer=MemorySaver(),
    ),
)
