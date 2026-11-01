"""Use a bounded real provider call in immediate and queued execution tests."""

import json
import os
from pathlib import Path

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory


attempted = 0


class LiveState(MessagesState):
    review_required: bool


def record(value: str) -> None:
    path = Path(os.environ["LAT_TEST_LIVE_EXECUTION_EVENTS"])
    with path.open("a") as stream:
        stream.write(value + "\n")


async def review(state: LiveState) -> dict:
    if state.get("review_required"):
        answer = interrupt({"question": "Allow the synthetic model request?"})
        if answer is not True:
            raise ValueError("The synthetic model request requires approval")
    return {}


async def reply(state: LiveState, config: RunnableConfig) -> dict:
    global attempted
    if os.environ.get("LAT_TEST_LIVE_EXECUTION") != "yes":
        raise RuntimeError("Live execution is not enabled")
    if attempted >= 2:
        raise RuntimeError("The live execution call budget is exhausted")
    attempted += 1
    record("attempted")
    parameters = json.loads(os.environ.get("LAT_TEST_OPENAI_MODEL_KWARGS", "{}"))
    if not isinstance(parameters, dict) or set(parameters) - {"reasoning_effort", "temperature"}:
        raise ValueError("Unsupported live model parameters")
    provider = os.environ.get("LAT_TEST_OPENAI_PROVIDER", "openai")
    if provider == "azure_openai":
        connection = {
            "azure_endpoint": os.environ["LAT_TEST_AZURE_OPENAI_ENDPOINT"],
            "api_version": os.environ["LAT_TEST_AZURE_OPENAI_API_VERSION"],
            "azure_deployment": os.environ["LAT_TEST_AZURE_OPENAI_DEPLOYMENT"],
        }
    elif provider == "openai":
        connection = {"base_url": os.environ.get("LAT_TEST_OPENAI_BASE_URL", "https://api.openai.com/v1")}
    else:
        raise ValueError("Only OpenAI and Azure OpenAI are supported in this live check")
    try:
        model = CompletionModelFactory.create(
            provider,
            os.environ["LAT_TEST_OPENAI_MODEL"],
            configurable_fields=(),
            config_prefix="",
            model_parameter_values=(),
            api_key=os.environ["LAT_TEST_OPENAI_API_KEY"],
            max_completion_tokens=512,
            max_retries=0,
            timeout=20,
            n=1,
            streaming=False,
            disable_streaming=True,
            use_responses_api=False,
            **connection,
            **parameters,
        )
        response = await model.ainvoke(
            [SystemMessage("Answer the synthetic arithmetic question in one short sentence."), *state["messages"]],
            config=config,
        )
    except Exception as error:
        record("failed:" + type(error).__name__)
        raise RuntimeError(f"Live provider call failed ({type(error).__name__})") from None
    if not isinstance(response, AIMessage) or not response.content:
        raise ValueError("The real model returned no message content")
    record("completed")
    return {"messages": [response]}


builder = StateGraph(LiveState)
builder.add_node("review", review)
builder.add_node("reply", reply)
builder.add_edge(START, "review")
builder.add_edge("review", "reply")
builder.add_edge("reply", END)
live_execution_agent = Agent("live-execution", "A bounded real-model execution check.", builder.compile())
