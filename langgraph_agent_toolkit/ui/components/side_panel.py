"""Render the chat controls and settings sidebar."""

import urllib.parse

import streamlit as st

from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.ui.utils import constants


def build_chat_url(base_url: str | None, agent: str, thread_id: str) -> str:
    """Keep the app path and encode the public conversation identifiers."""
    if not base_url:
        raise ValueError("The app URL is not available.")
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("The app URL must use HTTP or HTTPS.")
    query = urllib.parse.urlencode({"agent": agent, "thread_id": thread_id})
    return urllib.parse.urlunsplit(parsed._replace(query=query, fragment=""))


def side_panel_component(agent_client: AgentClient) -> tuple[bool, str]:
    """Render the sidebar and return `use_streaming` and `stream_protocol`."""
    with st.sidebar:
        st.header(f"{constants.APP_ICON} {constants.APP_TITLE}")

        ""
        "Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit"
        ""

        if st.button(":material/chat: New Chat", use_container_width=True):
            st.query_params.pop("thread_id", None)
            st.rerun()

        with st.popover(":material/settings: Settings", use_container_width=True):
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.agent)
            selected_agent = st.selectbox(
                "Agent to use",
                options=agent_list,
                index=agent_idx,
            )
            if selected_agent != agent_client.agent:
                st.query_params["agent"] = selected_agent
                st.query_params.pop("thread_id", None)
                st.rerun()
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
            try:
                chat_url = build_chat_url(st.context.url, agent_client.agent, st.session_state.thread_id)
            except ValueError:
                st.error("Could not determine the app URL. Copy the URL from your browser.")
                return
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Use this URL to resume the chat with the same authenticated user.")

        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()

        "[View the source code](https://github.com/kryvokhyzha/langgraph-agent-toolkit)"

    return use_streaming, stream_protocol
