"""Render chat history, tokens, tool calls, and task data."""

from collections.abc import AsyncGenerator

import streamlit as st
from pydantic import ValidationError

from langgraph_agent_toolkit.schema import ChatMessage
from langgraph_agent_toolkit.schema.task_data import TaskData, TaskDataStatus
from langgraph_agent_toolkit.ui.utils.message import render_human_message


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """Draw chat messages from history or a stream.

    Use a placeholder to render incoming tokens.
    Use status containers to render tool calls and results.
    Store the last message container in session state for later messages and feedback.

    Args:
        messages_agen: Asynchronous message iterator.
        is_new: Whether the messages are new.

    """
    # Store the last message container.
    last_message_type = None
    st.session_state.last_message = None

    # Store intermediate streaming tokens.
    streaming_content = ""
    streaming_placeholder = None
    pending_tool_calls = set()
    tool_statuses = {}

    # Use `None` to terminate the stream. An empty token is valid.
    while True:
        msg = await anext(messages_agen, None)
        if msg is None:
            break
        # A `str` message is an incoming token.
        if isinstance(msg, str):
            # Create a placeholder for the first token of a message.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("assistant")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue

        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            # Render a user message.
            case "human":
                last_message_type = "human"
                with st.chat_message("user"):
                    render_human_message(msg.content)

            # Render an agent message, its tokens, and its tool calls.
            case "ai":
                # Store new messages in session state.
                if is_new:
                    st.session_state.messages.append(msg)

                # Create a chat message after a non-AI message.
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("assistant")

                with st.session_state.last_message:
                    # Write message content and reset streaming data.
                    if msg.content:
                        if pending_tool_calls and not msg.tool_calls and not streaming_placeholder:
                            st.warning(msg.content)
                        elif streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            pending_tool_calls.add(tool_call["id"])
                            if st.session_state.display_tools_execution:
                                status = st.status(
                                    f"""Tool Call: {tool_call["name"]}""",
                                    state="running" if is_new else "complete",
                                )
                                tool_statuses[tool_call["id"]] = status
                                status.write("Input:")
                                status.write(tool_call["args"])

            case "tool":
                # Progress events can arrive between a tool call and its result.
                if is_new:
                    st.session_state.messages.append(msg)
                pending_tool_calls.discard(msg.tool_call_id)
                status = tool_statuses.pop(msg.tool_call_id, None)
                last_message_type = "tool"
                if st.session_state.display_tools_execution:
                    if status is None:
                        status = st.status("Tool Result", state="complete")
                    status.write("Output:")
                    status.code(str(msg.content))
                    status.update(state="complete")

            case "custom":
                # The `bg-task-agent` uses `CustomData` for task data.
                if is_new:
                    st.session_state.messages.append(msg)
                try:
                    task_data: TaskData | None = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    task_data = None
                if task_data is None or msg.custom_data.keys() - TaskData.model_fields.keys():
                    # Keep payloads from agents that use a different progress schema.
                    last_message_type = "custom"
                    st.session_state.last_message = st.chat_message(name="task", avatar=":material/manufacturing:")
                    with st.session_state.last_message:
                        st.write(msg.custom_data)
                    continue

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(name="task", avatar=":material/manufacturing:")
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # Report and stop on an unexpected message type.
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()
