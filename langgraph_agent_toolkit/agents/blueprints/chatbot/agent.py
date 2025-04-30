from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.func import entrypoint

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.core.models.factory import ModelFactory
from langgraph_agent_toolkit.schema.models import ModelProvider


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

    # Check if a model_config is specified in agent_config
    model_config_key = config["configurable"].get("agent_config", {}).get("model_config")

    if model_config_key and model_config_key in settings.MODEL_CONFIGS:
        # Use the model configuration from settings
        model_config = settings.MODEL_CONFIGS[model_config_key]
        model = ModelFactory.get_model_from_config(model_config)
    else:
        # Fall back to the traditional approach
        model = ModelFactory.create(
            model_provider=config["configurable"].get("model_provider", ModelProvider.OPENAI),
            model_name=config["configurable"].get("model_name", settings.OPENAI_MODEL_NAME),
            openai_api_base=settings.OPENAI_API_BASE_URL,
            openai_api_key=settings.OPENAI_API_KEY,
        )

    response = await model.ainvoke(messages)
    return entrypoint.final(value={"messages": [response]}, save={"messages": messages + [response]})


chatbot_agent = Agent(
    name="chatbot-agent",
    description="A simple chatbot.",
    graph=chatbot,
)
