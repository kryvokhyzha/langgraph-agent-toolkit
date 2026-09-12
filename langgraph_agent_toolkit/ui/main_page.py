"""Render the Streamlit chat page."""

import asyncio
import os
import uuid
from collections.abc import AsyncGenerator

import streamlit as st
from dotenv import find_dotenv, load_dotenv

from langgraph_agent_toolkit.client import AgentClient, AgentClientError
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.schema import ChatMessage
from langgraph_agent_toolkit.ui.components.draw_message import draw_messages
from langgraph_agent_toolkit.ui.components.side_panel import side_panel_component
from langgraph_agent_toolkit.ui.utils import constants
from langgraph_agent_toolkit.ui.utils.message import (
    build_chat_message,
    create_welcome_message,
    handle_feedback,
    render_human_message,
)


load_dotenv(find_dotenv(), override=False)


async def main_page() -> None:
    """Close HTTP connections before Streamlit replaces this event loop."""
    try:
        await _main_page()
    finally:
        client = st.session_state.get("agent_client")
        if client is not None:
            await client.aclose()


async def _main_page() -> None:
    """Set up the chat page, render history, handle input, and record feedback."""
    st.set_page_config(
        page_title=constants.APP_TITLE,
        page_icon=constants.APP_ICON,
        menu_items={},
    )

    # Hide the Streamlit upper-right controls.
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    if "agent_client" not in st.session_state:
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = await asyncio.to_thread(AgentClient, base_url=agent_url)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client
    user_id = settings.DEFAULT_STREAMLIT_USER_ID if settings.AUTH_MODE == "trusted" else None

    selected_agent = st.query_params.get("agent") or agent_client.info.default_agent
    if selected_agent not in {agent.key for agent in agent_client.info.agents}:
        st.error("The URL selects an agent that is not available.")
        st.stop()
    agent_client.agent = selected_agent

    requested_thread = st.query_params.get("thread_id")
    thread_id = requested_thread or str(uuid.uuid4())
    st.session_state.thread_id = thread_id
    st.query_params["thread_id"] = thread_id
    st.query_params["agent"] = selected_agent

    # Select the agent before loading or displaying its conversation.
    use_streaming, stream_protocol = side_panel_component(agent_client)

    conversation = (user_id, selected_agent, thread_id)
    if st.session_state.get("conversation") != conversation:
        if not requested_thread:
            # Add a welcome message for a new thread.
            messages = [create_welcome_message(agent_client.agent)]
        else:
            try:
                messages = []
                offset = 0
                while True:
                    history = await asyncio.to_thread(
                        agent_client.get_history, thread_id=thread_id, user_id=user_id, offset=offset
                    )
                    messages.extend(history.messages)
                    if history.next_offset is None:
                        break
                    offset = history.next_offset
            except AgentClientError:
                st.error("Could not load this conversation. Check the URL and try again.")
                st.stop()
        st.session_state.messages = messages
        st.session_state.conversation = conversation
        st.session_state.last_feedback = (None, None)

    # Draw existing messages.
    messages: list[ChatMessage] = st.session_state.messages

    # `draw_messages()` requires an asynchronous message iterator.
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate a message when the user submits text or attachments.
    if user_input := st.chat_input(accept_file="multiple", file_type=constants.MULTIMODAL_FILE_TYPES):
        # `accept_file` adds `.text` and `.files` to the submission.
        text = getattr(user_input, "text", "") or ""
        files = list(getattr(user_input, "files", None) or [])
        message = build_chat_message(text, files)

        max_attachments = settings.MULTIMODAL_MAX_ATTACHMENTS
        if isinstance(message, list) and max_attachments is not None:
            n_media = sum(1 for b in message if b.get("type") != "text")
            if n_media > max_attachments:
                st.error(f"Too many attachments ({n_media}); the limit is {max_attachments}.")
                st.stop()

        messages.append(ChatMessage(type="human", content=message))
        with st.chat_message("user"):
            render_human_message(message)
        try:
            if use_streaming:
                # `astream` and `astream_jsonl` yield the same `ChatMessage | str` values.
                astream_fn = agent_client.astream_jsonl if stream_protocol == "JSON Lines" else agent_client.astream
                stream = astream_fn(
                    input=dict(message=message),
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                await draw_messages(stream, is_new=True)
            else:
                response = await agent_client.ainvoke(
                    input=dict(message=message),
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                messages.append(response)
                st.chat_message("assistant").write(response.content)
            st.rerun()  # Clear stale containers
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # Show feedback only after messages are generated.
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()
