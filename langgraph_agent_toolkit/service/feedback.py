"""Bind feedback to runs issued to an authenticated user."""

import hashlib
import hmac
import json

from fastapi import HTTPException, Request

from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.schema import ChatMessage, Feedback
from langgraph_agent_toolkit.service.auth import authenticated_user_id


# These provider options can change the feedback target or record identity.
_RESERVED_KWARGS = frozenset(
    {
        "run_id",
        "trace_id",
        "observation_id",
        "session_id",
        "project_id",
        "dataset_run_id",
        "user_id",
        "key",
        "name",
        "score",
        "value",
        "id",
        "score_id",
        "feedback_id",
        "source_run_id",
        "comparative_experiment_id",
        "feedback_group_id",
        "environment",
    }
)


def _signing_key() -> bytes | None:
    """Return a server-only key, or disable signing when it is unavailable."""
    secret = settings.FEEDBACK_SIGNING_SECRET
    if secret is None:
        return None
    value = secret.get_secret_value()
    credentials = [token.get_secret_value() for token in settings.AUTH_USERS.values()]
    if settings.AUTH_SECRET is not None:
        credentials.append(settings.AUTH_SECRET.get_secret_value())
    if len(value) < 32 or value in credentials:
        return None
    return value.encode("utf-8")


def _token(key: bytes, user_id: str, agent_id: str, run_id: str) -> str:
    """Sign one run without storing a per-worker record."""
    payload = json.dumps(
        ["lat-feedback-v1", user_id, agent_id, run_id], ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return "v1." + hmac.new(key, payload, hashlib.sha256).hexdigest()


def sign_feedback_message(request: Request, agent_id: str, message: ChatMessage) -> ChatMessage:
    """Attach proof to a genuine run result. Never call this for saved history."""
    token = None
    key = _signing_key()
    if not request.state.principal.trusted and message.run_id and key is not None:
        owner = authenticated_user_id(request, None)
        token = _token(key, owner, agent_id, message.run_id)
    return message.model_copy(update={"feedback_token": token})


def authorize_feedback(request: Request, agent_id: str, feedback: Feedback) -> str:
    """Check the target and proof before a provider call. Return the user ID."""
    owner = authenticated_user_id(request, feedback.user_id)
    source_info = feedback.kwargs.get("source_info")
    if _RESERVED_KWARGS.intersection(feedback.kwargs) or (isinstance(source_info, dict) and "__run" in source_info):
        raise HTTPException(422, "Feedback kwargs cannot override the feedback target or identity")
    if request.state.principal.trusted:
        return owner
    key = _signing_key()
    if key is None:
        raise HTTPException(503, "Token-user feedback requires a separate server-only FEEDBACK_SIGNING_SECRET")
    expected = _token(key, owner, agent_id, feedback.run_id)
    provided = feedback.feedback_token
    if not provided or not provided.isascii() or not hmac.compare_digest(provided, expected):
        raise HTTPException(403, "Missing or invalid feedback token")
    return owner
