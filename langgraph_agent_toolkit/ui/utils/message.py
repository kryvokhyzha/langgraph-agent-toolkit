"""Provide Streamlit message, feedback, and multimodal content helpers."""

import base64

import streamlit as st

from langgraph_agent_toolkit.client import AgentClient, AgentClientError
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.schema import ChatMessage


def create_welcome_message(agent: str) -> ChatMessage:
    """Create a welcome message for the current agent."""
    match agent:
        case "chatbot":
            welcome_content = "Hello! I'm a simple chatbot. Ask me anything!"
        case "interrupt-agent":
            welcome_content = (
                "Hello! I'm an interrupt agent. Tell me your birthday and I will predict your personality!"
            )
        case _:
            welcome_content = "Hello! I'm an AI agent. Ask me anything!"
    return ChatMessage(type="ai", content=welcome_content)


def file_to_content_block(file) -> dict:
    """Encode an uploaded Streamlit file as a base64 LangChain content block."""
    mime = file.type or "application/octet-stream"
    data = base64.b64encode(file.getvalue()).decode("utf-8")
    if mime.startswith("image/"):
        block_type = "image"
    elif mime.startswith("audio/"):
        block_type = "audio"
    elif mime.startswith("video/"):
        block_type = "video"
    else:
        block_type = "file"  # e.g. application/pdf
    return {"type": block_type, "base64": data, "mime_type": mime}


def build_chat_message(text: str, files: list) -> str | list[dict]:
    """Return text or content blocks when files are attached."""
    if not files:
        return text
    blocks: list[dict] = []
    if text:
        blocks.append({"type": "text", "text": text})
    blocks.extend(file_to_content_block(f) for f in files)
    return blocks


def render_human_message(content: str | list) -> None:
    """Render text or multimodal content blocks from a user message."""
    if not isinstance(content, list):
        st.write(content)
        return
    for block in content:
        btype = block.get("type")
        if btype == "text":
            st.write(block.get("text", ""))
        elif btype == "image" and block.get("url"):
            st.image(block["url"])
        elif btype == "image" and block.get("base64"):
            # Do not crash when history contains invalid base64 data.
            try:
                st.image(base64.b64decode(block["base64"]))
            except Exception:
                st.caption(f"📎 {block.get('mime_type') or btype} attachment (invalid base64)")
        else:
            st.caption(f"📎 {block.get('mime_type') or btype} attachment")


async def handle_feedback() -> None:
    """Draw the feedback widget and record user feedback."""
    # Store the last feedback to prevent duplicate records.
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id

    if latest_run_id:
        feedback = st.feedback("stars", key=latest_run_id)

        # Send a record when the feedback value or run ID changes.
        if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
            # Convert the feedback index to a score from 0 to 1.
            normalized_score = (feedback + 1) / 5.0

            agent_client: AgentClient = st.session_state.agent_client
            try:
                await agent_client.acreate_feedback(
                    run_id=latest_run_id,
                    key="human-feedback-stars",
                    score=normalized_score,
                    kwargs={"comment": "In-line human feedback"},
                    user_id=settings.DEFAULT_STREAMLIT_USER_ID if settings.AUTH_MODE == "trusted" else None,
                    feedback_token=st.session_state.messages[-1].feedback_token,
                )
            except AgentClientError as e:
                st.error(f"Error recording feedback: {e}")
                st.stop()
            st.session_state.last_feedback = (latest_run_id, feedback)
            st.toast("Feedback recorded", icon=":material/reviews:")
