"""Use the same scoped conversation keys in HTTP and worker execution."""

import hashlib
import json


def validate_conversation_identifiers(owner_id: str, agent_id: str, thread_id: str) -> None:
    """Reject invalid identities before worker access to scoped checkpoints."""
    for name, value in (("owner_id", owner_id), ("agent_id", agent_id), ("thread_id", thread_id)):
        if not isinstance(value, str) or not value or len(value) > 256 or "\x00" in value:
            raise ValueError(f"{name} must contain 1 to 256 characters without NUL")


def storage_thread_id(user_id: str, agent_id: str, thread_id: str) -> str:
    """Scope short-term checkpoints to their owner, agent, and public thread.

    This key does not replace the user ID used by long-term memory stores.
    """
    identity = json.dumps([user_id, agent_id, thread_id], ensure_ascii=True, separators=(",", ":"))
    return "lat:v1:" + hashlib.sha256(identity.encode()).hexdigest()
