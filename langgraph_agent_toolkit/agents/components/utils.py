from typing import TypeVar

from langchain_core.messages.utils import trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.prebuilt.chat_agent_executor import AgentState, AgentStateWithStructuredResponse

from langgraph_agent_toolkit.helper.constants import DEFAULT_MAX_MESSAGE_HISTORY_LENGTH


T = TypeVar("T")


class AgentStateWithRemainingSteps(AgentState):
    remaining_steps: RemainingSteps


class AgentStateWithStructuredResponseAndRemainingSteps(AgentStateWithStructuredResponse):
    remaining_steps: RemainingSteps


def pre_model_hook_standard(state: T, config: RunnableConfig):
    _max_messages = config.get("configurable", {}).get("checkpointer_params", {}).get("k", None)

    updated_messages = trim_messages(
        state["messages"],
        token_counter=len,
        max_tokens=_max_messages or DEFAULT_MAX_MESSAGE_HISTORY_LENGTH,
        strategy="last",
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
        allow_partial=False,
    )

    return {"llm_input_messages": updated_messages}


def default_pre_model_hook(state: T, config: RunnableConfig) -> T:
    return state
