"""Sidebar: chat controls (new chat, architecture, privacy, share) and the settings popover."""

import urllib.parse
import uuid

import streamlit as st

from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.ui.utils import constants
from langgraph_agent_toolkit.ui.utils.message import create_welcome_message


def side_panel_component(agent_client: AgentClient) -> tuple[bool, str]:
    """Render the sidebar and return the chosen (use_streaming, stream_protocol) settings."""
    with st.sidebar:
        st.header(f"{constants.APP_ICON} {constants.APP_TITLE}")

        ""
        "Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit"
        ""

        if st.button(":material/chat: New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            # Add welcome message to new chat
            st.session_state.messages.append(create_welcome_message(agent_client.agent))
            st.rerun()

        with st.popover(":material/settings: Settings", use_container_width=True):
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=agent_list,
                index=agent_idx,
            )
            use_streaming = st.toggle("Stream results", value=True)
            stream_protocol = st.radio(
                "Streaming protocol",
                options=["SSE", "JSON Lines"],
                index=0,
                horizontal=True,
                disabled=not use_streaming,
                help="Transport for streamed responses: Server-Sent Events or JSON Lines (NDJSON). "
                "Both deliver identical messages.",
            )
            st.session_state.display_tools_execution = st.toggle("Display tools execution", value=False)

        @st.dialog("Architecture")
        def architecture_dialog() -> None:
            st.image(
                "https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/media/agent_architecture.png?raw=true"
            )
            "[View full size on GitHub](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/media/agent_architecture.png)"
            st.caption(
                "App hosted on [Streamlit Cloud](https://share.streamlit.io/) with FastAPI service running in "
                "[Azure](https://learn.microsoft.com/en-us/azure/app-service/)"
            )

        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to selected "
                "observability service for product evaluation and improvement purposes only."
            )

        @st.dialog("Share/resume chat")
        def share_chat_dialog() -> None:
            session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
            st_base_url = urllib.parse.urlunparse(
                [session.client.request.protocol, session.client.request.host, "", "", "", ""]
            )

            if isinstance(st_base_url, bytes):
                # Decode the base URL if it's in bytes
                st_base_url = st_base_url.decode()

            # if it's not localhost, switch to https by default
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            chat_url = f"{st_base_url}?thread_id={st.session_state.thread_id}"
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or revisit this chat")

        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()

        "[View the source code](https://github.com/kryvokhyzha/langgraph-agent-toolkit)"

    return use_streaming, stream_protocol
