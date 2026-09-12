from datetime import datetime
from typing import Any

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory
from langgraph_agent_toolkit.schema.models import ModelProvider


class AgentState(MessagesState, total=False):
    """State that permits optional keys."""

    birthdate: datetime | None


def wrap_model(
    model: BaseChatModel | Runnable[LanguageModelInput, Any], system_prompt: BaseMessage
) -> RunnableSerializable[Any, BaseMessage]:
    preprocessor = RunnableLambda(
        lambda state: [system_prompt] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


background_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful assistant that tells users there zodiac sign.
Provide a one paragraph summary of the origin of zodiac signs.
Don't tell the user what their sign is, you are just demonstrating your knowledge on the topic.
""")


async def background(state: AgentState, config: RunnableConfig) -> AgentState:
    """Run a model call before the interrupt."""
    m = CompletionModelFactory.create(
        model_provider=config["configurable"].get(
            "model_provider", ModelProvider.FAKE if settings.USE_FAKE_MODEL else ModelProvider.OPENAI
        ),
        model_name=config["configurable"].get("model_name", settings.OPENAI_MODEL_NAME),
        openai_api_base=settings.OPENAI_API_BASE_URL,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    model_runnable = wrap_model(m, background_prompt.format())
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [AIMessage(content=response.content)]}


birthdate_extraction_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert at extracting birthdates from conversational text.

Rules for extraction:
- Look for user messages that mention birthdates
- Consider various date formats (MM/DD/YYYY, YYYY-MM-DD, Month Day, Year)
- Validate that the date is reasonable (not in the future)
- If no clear birthdate was provided by the user, return None
""")


class BirthdateExtraction(BaseModel):
    birthdate: str | None = Field(
        description="The extracted birthdate in YYYY-MM-DD format. If no birthdate is found, this should be None."
    )
    reasoning: str = Field(description="Explanation of how the birthdate was extracted or why no birthdate was found")


async def determine_birthdate(state: AgentState, config: RunnableConfig) -> AgentState:
    """Find the user's birthdate in the conversation history.

    Interrupt when no birthdate is found.
    """
    m = CompletionModelFactory.create(
        model_provider=config["configurable"].get(
            "model_provider", ModelProvider.FAKE if settings.USE_FAKE_MODEL else ModelProvider.OPENAI
        ),
        model_name=config["configurable"].get("model_name", settings.OPENAI_MODEL_NAME),
        config_prefix="",
        configurable_fields=(),
        model_parameter_values=(("temperature", 0.0), ("top_p", 0.7), ("streaming", False)),
        openai_api_base=settings.OPENAI_API_BASE_URL,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    model_runnable = wrap_model(
        m.with_structured_output(BirthdateExtraction, strict=True, include_raw=True),
        birthdate_extraction_prompt.format(),
    ).with_config(tags=["skip_stream"])
    raw_response: BirthdateExtraction = await model_runnable.ainvoke(state, config)

    if raw_response["parsed"] is not None:
        response: BirthdateExtraction = raw_response["parsed"]
    elif raw_response["raw"].tool_calls is not None:
        raw_result = raw_response["raw"].tool_calls[-1]["args"]
        response = BirthdateExtraction(**raw_result)
    else:
        raise ValueError("No valid response from the model")

    if response.birthdate is None:
        birthdate_input = interrupt(f"{response.reasoning}\nPlease tell me your birthdate?")
        state["messages"].append(HumanMessage(birthdate_input["message"]))
        return await determine_birthdate(state, config)

    try:
        birthdate = datetime.fromisoformat(response.birthdate)
    except ValueError:
        birthdate_input = interrupt(
            "I couldn't understand the date format. Please provide your birthdate in YYYY-MM-DD format."
        )
        state["messages"].append(HumanMessage(birthdate_input["message"]))
        return await determine_birthdate(state, config)

    return {
        "messages": [],
        "birthdate": birthdate,
    }


sign_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful assistant that tells users there zodiac sign.
What is the sign of somebody born on {birthdate}?
""")


async def determine_sign(state: AgentState, config: RunnableConfig) -> AgentState:
    """Determine the user's zodiac sign from their birthdate."""
    if not state.get("birthdate"):
        raise ValueError("No birthdate found in state")

    m = CompletionModelFactory.create(
        model_provider=config["configurable"].get(
            "model_provider", ModelProvider.FAKE if settings.USE_FAKE_MODEL else ModelProvider.OPENAI
        ),
        model_name=config["configurable"].get("model_name", settings.OPENAI_MODEL_NAME),
        openai_api_base=settings.OPENAI_API_BASE_URL,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    model_runnable = wrap_model(m, sign_prompt.format(birthdate=state["birthdate"].strftime("%Y-%m-%d")))
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [AIMessage(content=response.content)]}


agent = StateGraph(AgentState)
agent.add_node("background", background)
agent.add_node("determine_birthdate", determine_birthdate)
agent.add_node("determine_sign", determine_sign)

agent.set_entry_point("background")
agent.add_edge("background", "determine_birthdate")
agent.add_edge("determine_birthdate", "determine_sign")
agent.add_edge("determine_sign", END)

interrupt_agent = Agent(
    name="interrupt-agent",
    description="An agent the uses interrupts.",
    graph=agent.compile(checkpointer=None),
)
interrupt_agent.graph.name = "interrupt-agent"
