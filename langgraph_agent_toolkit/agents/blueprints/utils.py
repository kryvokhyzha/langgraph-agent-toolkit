from langchain_core.messages.utils import trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState

from langgraph_agent_toolkit.helper.constants import DEFAULT_MAX_MESSAGE_HISTORY_LENGTH


def pre_model_hook_standard(state: AgentState, config: RunnableConfig):
    # https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-manage-message-history/
    # if last message is a human message, trim the messages to only include human messages
    updated_messages = state["messages"]
    if updated_messages[-1].type == "human" or config["metadata"]["langgraph_step"] == 1:
        updated_messages = [
            message
            for message in updated_messages
            if message.type not in {"tool", "tool_call", "function"} and message.content
        ]

        MAX_MESSAGES = config.get("configurable", {}).get("memory_saver_params", {}).get("k", None)

        updated_messages = trim_messages(
            updated_messages,
            token_counter=len,  # <-- len will simply count the number of messages rather than tokens
            max_tokens=MAX_MESSAGES
            or DEFAULT_MAX_MESSAGE_HISTORY_LENGTH,  # <-- allow up to `memory_max_tokens` messages.
            strategy="last",
            start_on="human",
            end_on=("human", "tool"),
            include_system=True,
            allow_partial=False,
        )

    return {"llm_input_messages": updated_messages}
