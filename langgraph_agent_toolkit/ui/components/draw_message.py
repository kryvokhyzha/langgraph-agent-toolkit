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
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Map each tool call ID to its status container.
                        call_results = {}

                        for tool_call in msg.tool_calls:
                            if st.session_state.display_tools_execution:
                                status = st.status(
                                    f"""Tool Call: {tool_call["name"]}""",
                                    state="running" if is_new else "complete",
                                )
                                call_results[tool_call["id"]] = status
                                status.write("Input:")
                                status.write(tool_call["args"])

                        # Read one `ToolMessage` for each tool call unless the run pauses.
                        for _ in range(len(msg.tool_calls)):
                            tool_result: ChatMessage | str | None = await anext(messages_agen, None)

                            # Stop when the next item is not a `ToolMessage`.
                            # A human-in-the-loop run can emit an AI interrupt prompt before a tool runs.
                            # A token at this point is invalid.
                            if not isinstance(tool_result, ChatMessage) or tool_result.type != "tool":
                                if isinstance(tool_result, ChatMessage):
                                    if is_new:
                                        st.session_state.messages.append(tool_result)
                                    if tool_result.content:
                                        st.warning(tool_result.content)
                                elif tool_result is not None:
                                    st.error(
                                        f"Unexpected stream chunk while waiting for a tool result: {type(tool_result)}"
                                    )
                                    st.stop()
                                break

                            # Store new results and update the matching status container.
                            if is_new:
                                st.session_state.messages.append(tool_result)

                            if st.session_state.display_tools_execution:
                                # Use a standalone status when `tool_call_id` is missing.
                                status = call_results.get(tool_result.tool_call_id)
                                if status is None:
                                    status = st.status("Tool Result", state="complete")
                                status.write("Output:")
                                # Render plain tool output in a code block.
                                status.code(str(tool_result.content))
                                status.update(state="complete")

            case "custom":
                # The `bg-task-agent` uses `CustomData` for task data.
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Unexpected CustomData message received from agent")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

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
