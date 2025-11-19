from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.checkpoint.memory import MemorySaver

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.components.tools import add, multiply
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory
from langgraph_agent_toolkit.helper.constants import DEFAULT_MAX_MESSAGE_HISTORY_LENGTH
from langgraph_agent_toolkit.schema.models import ModelProvider


model = CompletionModelFactory.create(
    model_provider=ModelProvider.OPENAI,
    model_name=settings.OPENAI_MODEL_NAME,
    config_prefix="",
    configurable_fields=(),
    model_parameter_values=(("temperature", 0.2), ("top_p", 0.95), ("streaming", False)),
    openai_api_base=settings.OPENAI_API_BASE_URL,
    openai_api_key=settings.OPENAI_API_KEY,
)

react_agent = Agent(
    name="react-agent-new",
    description="A new react agent.",
    graph=create_agent(
        model=model,
        tools=[add, multiply, DuckDuckGoSearchResults()],
        middleware=[
            SummarizationMiddleware(
                model=model,
                max_tokens_before_summary=25_000,  # Trigger summarization at 25,000 tokens
                messages_to_keep=DEFAULT_MAX_MESSAGE_HISTORY_LENGTH,  # Keep last N messages after summary
            ),
            ModelCallLimitMiddleware(
                run_limit=5,
                exit_behavior="end",
            ),
            ToolCallLimitMiddleware(
                run_limit=10,
                exit_behavior="end",
            ),
            # ContextEditingMiddleware(
            #     edits=[
            #         ClearToolUsesEdit(trigger=25_000),
            #     ],
            # ),
        ],
        system_prompt=(
            "You are a team support agent that can perform calculations and search the web. "
            "You can use the tools provided to help you with your tasks. "
            "You can also ask clarifying questions to the user. "
        ),
        state_schema=AgentState,
        checkpointer=MemorySaver(),
    ),
)
