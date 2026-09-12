"""Bind conversation storage to authenticated callers."""

import hashlib
import json
import secrets
from dataclasses import dataclass
from uuid import uuid4

from fastapi import HTTPException, Request

from langgraph_agent_toolkit.core.settings import settings


@dataclass(frozen=True)
class Principal:
    user_id: str
    trusted: bool = False


def validate_auth_configuration() -> None:
    """Reject ambiguous credentials and unauthenticated production startup."""
    tokens = [secret.get_secret_value() for secret in settings.AUTH_USERS.values()]
    if any(not user for user in settings.AUTH_USERS) or any(not token for token in tokens):
        raise ValueError("AUTH_USERS requires nonempty user IDs and tokens")
    if len(tokens) != len(set(tokens)):
        raise ValueError("Each AUTH_USERS token must identify exactly one user")
    shared = settings.AUTH_SECRET.get_secret_value() if settings.AUTH_SECRET else None
    if shared in tokens:
        raise ValueError("AUTH_SECRET must differ from each AUTH_USERS token")
    signing_secret = settings.FEEDBACK_SIGNING_SECRET
    if signing_secret is not None:
        signing_value = signing_secret.get_secret_value()
        if len(signing_value) < 32:
            raise ValueError("FEEDBACK_SIGNING_SECRET must contain at least 32 characters")
        if signing_value == shared or signing_value in tokens:
            raise ValueError("FEEDBACK_SIGNING_SECRET must differ from all client bearer tokens")
    if settings.AUTH_MODE == "trusted" and not shared:
        raise ValueError("AUTH_MODE=trusted requires AUTH_SECRET")
    if not settings.is_dev() and not tokens and not shared:
        raise ValueError("Production requires AUTH_SECRET or AUTH_USERS")


def authenticate(token: str | None) -> Principal:
    """Resolve a bearer token to one principal."""
    matched = None
    for user_id, secret in settings.AUTH_USERS.items():
        if secrets.compare_digest((token or "").encode(), secret.get_secret_value().encode()):
            matched = Principal(user_id)
    if matched is not None:
        return matched
    if settings.AUTH_SECRET and secrets.compare_digest(
        (token or "").encode(), settings.AUTH_SECRET.get_secret_value().encode()
    ):
        return Principal(settings.AUTH_SERVICE_USER_ID, trusted=settings.AUTH_MODE == "trusted")
    if not settings.AUTH_USERS and not settings.AUTH_SECRET and settings.is_dev():
        return Principal("anonymous", trusted=True)
    raise HTTPException(401, "Missing or invalid bearer token", headers={"WWW-Authenticate": "Bearer"})


def storage_thread_id(user_id: str, agent_id: str, thread_id: str) -> str:
    """Scope short-term checkpoints to their owner, agent, and public thread.

    This key does not replace the user ID used by long-term memory stores.
    """
    identity = json.dumps([user_id, agent_id, thread_id], ensure_ascii=True, separators=(",", ":"))
    return "lat:v1:" + hashlib.sha256(identity.encode()).hexdigest()


def authenticated_user_id(request: Request, user_id: str | None) -> str:
    """Resolve the stable user identity for memory access and observability."""
    principal: Principal = request.state.principal
    if principal.trusted:
        owner = user_id or principal.user_id
    else:
        if user_id is not None and user_id != principal.user_id:
            raise HTTPException(403, "user_id must match the authenticated user")
        owner = principal.user_id
    return owner


def conversation_identity(
    request: Request, agent_id: str, thread_id: str | None, user_id: str | None, *, create: bool = False
) -> tuple[str, str, str]:
    """Validate the caller and return public, owner, and storage identifiers."""
    owner = authenticated_user_id(request, user_id)
    public_id = thread_id or (str(uuid4()) if create else None)
    if not public_id:
        raise HTTPException(422, "thread_id is required")
    if len(public_id) > 256 or len(owner) > 256:
        raise HTTPException(422, "Conversation identifiers must not exceed 256 characters")
    return public_id, owner, storage_thread_id(owner, agent_id, public_id)


def execution_input(request: Request, agent_id: str, value):
    """Create execution input with authenticated storage identifiers."""
    public_id, owner, key = conversation_identity(request, agent_id, value.thread_id, value.user_id, create=True)
    forbidden = {"thread_id", "user_id", "checkpoint_id", "checkpoint_ns"}
    if forbidden.intersection(value.agent_config) or any(k.startswith("__") for k in value.agent_config):
        raise HTTPException(422, "agent_config cannot override identity or checkpoint fields")
    if value.recursion_limit is not None and not 1 <= value.recursion_limit <= settings.DEFAULT_RECURSION_LIMIT:
        raise HTTPException(422, f"recursion_limit must be between 1 and {settings.DEFAULT_RECURSION_LIMIT}")
    return public_id, value.model_copy(update={"thread_id": key, "user_id": owner})
