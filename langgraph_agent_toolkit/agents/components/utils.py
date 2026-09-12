from typing import Sequence, TypeVar

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.managed.is_last_step import RemainingSteps

from langgraph_agent_toolkit.core.settings import settings


try:
    from langchain.agents import AgentState
except ImportError:
    from langgraph.prebuilt.chat_agent_executor import AgentState


T = TypeVar("T")


class AgentStateWithRemainingSteps(AgentState):
    remaining_steps: RemainingSteps


def pre_model_hook_standard(state: T, config: RunnableConfig):
    _max_messages = config.get("configurable", {}).get("checkpointer_params", {}).get("k", None)

    updated_messages = trim_messages(
        state["messages"],
        token_counter=len,
        max_tokens=int(_max_messages or settings.DEFAULT_MAX_MESSAGE_HISTORY_LENGTH),
        strategy="last",
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
        allow_partial=False,
    )

    return {"llm_input_messages": updated_messages}


def default_pre_model_hook(state: T, config: RunnableConfig) -> T:
    return state


def trim_messages_wrapper(messages: Sequence[BaseMessage], config: RunnableConfig, **kwargs):
    """Trim messages to the configured limit.

    Args:
        messages: Messages to trim.
        config: Configuration with trimming parameters.
        **kwargs: Additional arguments for `trim_messages`.

    Returns:
        Trimmed messages.

    """
    _max_messages = config.get("configurable", {}).get("checkpointer_params", {}).get("k", None)

    default_kwargs = dict(
        token_counter=len,
        max_tokens=int(_max_messages or settings.DEFAULT_MAX_MESSAGE_HISTORY_LENGTH),
        strategy="last",
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
        allow_partial=False,
    )
    default_kwargs.update(kwargs)

    return trim_messages(
        messages=messages,
        **default_kwargs,
    )
