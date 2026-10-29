"""Check conversation identities without loading saved graph state."""

from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_agent_toolkit.core.memory.threads import (
    InvalidThreadCursor,
    ThreadListingUnsupported,
    list_threads,
)
from langgraph_agent_toolkit.service.auth import storage_thread_id


def write(saver, thread_id, metadata, *, checkpoint_id="0000000001"):
    checkpoint = empty_checkpoint()
    checkpoint["id"] = checkpoint_id
    saver.put(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
        checkpoint,
        {"step": 0, **metadata},
        {},
    )


async def read(saver, **kwargs):
    return await list_threads(
        saver,
        user_id="owner",
        agent_id="agent",
        storage_key=lambda public: storage_thread_id("owner", "agent", public),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_canonical_identity_takes_precedence_and_storage_hash_must_match():
    saver = InMemorySaver()
    entries = [
        ("valid", {"lat_user_id": "owner", "lat_agent_id": "agent", "lat_thread_id": "valid"}),
        ("legacy", {"user_id": "owner", "langfuse_session_id": "legacy"}),
        ("changed-owner", {"lat_user_id": "other", "user_id": "owner", "langfuse_session_id": "changed-owner"}),
        ("null-owner", {"lat_user_id": None, "user_id": "owner", "langfuse_session_id": "null-owner"}),
        ("changed-agent", {"user_id": "owner", "lat_agent_id": "other", "langfuse_session_id": "changed-agent"}),
        ("changed-public", {"user_id": "owner", "lat_thread_id": "forged", "langfuse_session_id": "changed-public"}),
        ("missing-public", {"user_id": "owner"}),
    ]
    for public, metadata in entries:
        write(saver, storage_thread_id("owner", "agent", public), metadata)
    write(
        saver,
        storage_thread_id("owner", "other-agent", "forged-scope"),
        {"user_id": "owner", "lat_agent_id": "agent", "lat_thread_id": "forged-scope"},
    )
    page = await read(saver)
    assert {thread.thread_id for thread in page.threads} == {"valid", "legacy"}
    assert page.next_cursor is None


@pytest.mark.asyncio
async def test_memory_listing_does_not_deserialize_later_checkpoints(monkeypatch):
    saver = InMemorySaver()
    key = storage_thread_id("owner", "agent", "conversation")
    metadata = {"user_id": "owner", "langfuse_session_id": "conversation"}
    for step in range(-1, 1000):
        write(saver, key, {**metadata, "step": step}, checkpoint_id=f"{step + 1:010}")
    original_loads = saver.serde.loads_typed
    loaded = []

    def checked_loads(value):
        loaded.append(value)
        return original_loads(value)

    monkeypatch.setattr(saver.serde, "loads_typed", checked_loads)
    monkeypatch.setattr(saver, "aget_tuple", AsyncMock(side_effect=AssertionError("Checkpoint payload read")))
    page = await read(saver)
    assert [thread.thread_id for thread in page.threads] == ["conversation"]
    assert len(loaded) == 2
    assert all("channel_values" not in original_loads(value) for value in loaded)


@pytest.mark.asyncio
async def test_cursor_tie_breaker_keeps_threads_with_the_same_checkpoint_id():
    saver = InMemorySaver()
    for public in ("first", "second", "third"):
        write(
            saver,
            storage_thread_id("owner", "agent", public),
            {"user_id": "owner", "langfuse_session_id": public},
        )
    ids = []
    cursor = None
    for _ in range(3):
        page = await read(saver, limit=1, cursor=cursor)
        ids.extend(thread.thread_id for thread in page.threads)
        cursor = page.next_cursor
    assert set(ids) == {"first", "second", "third"}
    assert cursor is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", ["", "not a cursor", "x" * 2049, "W10", "bnVsbA", "e30"])
async def test_invalid_cursor_is_rejected(cursor):
    with pytest.raises(InvalidThreadCursor):
        await read(InMemorySaver(), cursor=cursor)


@pytest.mark.asyncio
async def test_unknown_checkpointer_does_not_fall_back_to_checkpoint_scans():
    saver = AsyncMock()
    with pytest.raises(ThreadListingUnsupported):
        await read(saver)
    saver.alist.assert_not_called()
