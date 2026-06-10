"""Render chat messages — replay existing history or stream new tokens, tool calls, and task data."""

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
    """Draw a set of chat messages - either replaying existing messages or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_agen: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.

    """
    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
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
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                with st.chat_message("user"):
                    render_human_message(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("assistant")

                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
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

                        # Expect one ToolMessage for each tool call.
                        for _ in range(len(msg.tool_calls)):
                            tool_result: ChatMessage = await anext(messages_agen)

                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)

                            if st.session_state.display_tools_execution:
                                if tool_result.tool_call_id:
                                    status = call_results[tool_result.tool_call_id]
                                status.write("Output:")
                                # Write content as plain text, disabling markdown rendering
                                status.write(f"<pre>{tool_result.content}</pre>", unsafe_allow_html=True)
                                status.update(state="complete")

            case "custom":
                # CustomData example used by the bg-task-agent
                # See:
                # - langgraph_agent_toolkit/agents/components/utils.py CustomData
                # - langgraph_agent_toolkit/agents/blueprints/bg_task_agent/task.py
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

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()
