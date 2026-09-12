import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import StreamWriter

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.blueprints.bg_task_agent.task import Task
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory
from langgraph_agent_toolkit.schema.models import ModelProvider


class AgentState(MessagesState, total=False):
    """State that permits optional keys."""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    preprocessor = RunnableLambda(
        lambda state: state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    model_config_key = config["configurable"].get("model_config_key")

    if model_config_key and model_config_key in settings.MODEL_CONFIGS:
        model_config = settings.MODEL_CONFIGS[model_config_key]
        m = CompletionModelFactory.get_model_from_config(model_config)
    else:
        m = CompletionModelFactory.create(
            model_provider=config["configurable"].get(
                "model_provider", ModelProvider.FAKE if settings.USE_FAKE_MODEL else ModelProvider.OPENAI
            ),
            model_name=config["configurable"].get("model_name", settings.OPENAI_MODEL_NAME),
            openai_api_base=settings.OPENAI_API_BASE_URL,
            openai_api_key=settings.OPENAI_API_KEY,
        )

    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [response]}


async def bg_task(state: AgentState, writer: StreamWriter) -> AgentState:
    task1 = Task("Simple task 1...", writer)
    task2 = Task("Simple task 2...", writer)

    task1.start()
    await asyncio.sleep(2)
    task2.start()
    await asyncio.sleep(2)
    task1.write_data(data={"status": "Still running..."})
    await asyncio.sleep(2)
    task2.finish(result="error", data={"output": 42})
    await asyncio.sleep(2)
    task1.finish(result="success", data={"output": 42})
    return {"messages": []}


agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("bg_task", bg_task)
agent.set_entry_point("bg_task")

agent.add_edge("bg_task", "model")
agent.add_edge("model", END)

bg_task_agent = Agent(
    name="bg-task-agent",
    description="A background task agent.",
    graph=agent.compile(checkpointer=None),
)
