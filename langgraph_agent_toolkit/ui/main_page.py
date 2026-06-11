"""Top-level Streamlit page: wires the agent client, sidebar, message stream, and feedback widget."""

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


load_dotenv(find_dotenv(), override=True)


async def main_page() -> None:
    """Render the chat page: set up the client/thread, draw history, handle input, and feedback."""
    st.set_page_config(
        page_title=constants.APP_TITLE,
        page_icon=constants.APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
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
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            # Add welcome message to messages when creating a new thread
            messages = [create_welcome_message(agent_client.agent)]
        else:
            try:
                messages: list[ChatMessage] = agent_client.get_history(
                    thread_id=thread_id,
                    user_id=settings.DEFAULT_STREAMLIT_USER_ID,
                ).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Sidebar / config options
    use_streaming, stream_protocol = side_panel_component(agent_client)

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input (text and/or file attachments)
    if user_input := st.chat_input(accept_file="multiple", file_type=constants.MULTIMODAL_FILE_TYPES):
        # With accept_file set, the submission carries `.text` and `.files`.
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
                # astream (SSE) and astream_jsonl (NDJSON) share a signature and yield the same
                # ChatMessage | str, so draw_messages handles either protocol unchanged.
                astream_fn = agent_client.astream_jsonl if stream_protocol == "JSON Lines" else agent_client.astream
                stream = astream_fn(
                    input=dict(message=message),
                    thread_id=st.session_state.thread_id,
                    user_id=settings.DEFAULT_STREAMLIT_USER_ID,
                )
                await draw_messages(stream, is_new=True)
            else:
                response = await agent_client.ainvoke(
                    input=dict(message=message),
                    thread_id=st.session_state.thread_id,
                    user_id=settings.DEFAULT_STREAMLIT_USER_ID,
                )
                messages.append(response)
                st.chat_message("assistant").write(response.content)
            st.rerun()  # Clear stale containers
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # If messages have been generated, show feedback widget only for AI messages
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()
