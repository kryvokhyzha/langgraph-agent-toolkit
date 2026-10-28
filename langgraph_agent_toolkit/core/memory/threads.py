"""List saved conversations without loading checkpoint state."""

import base64
import binascii
import hashlib
import heapq
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langgraph_agent_toolkit.agents.components.checkpoint.empty import NoOpSaver


SQLITE_THREAD_INDEX = """
CREATE INDEX IF NOT EXISTS lat_checkpoint_thread_heads_idx ON checkpoints (
    COALESCE(json_extract(metadata, '$.lat_user_id'), json_extract(metadata, '$.user_id')),
    checkpoint_id DESC, thread_id DESC
)
WHERE checkpoint_ns = '' AND json_extract(metadata, '$.step') = 0
"""

POSTGRES_THREAD_INDEX = """
CREATE INDEX CONCURRENTLY IF NOT EXISTS lat_checkpoint_thread_heads_idx ON checkpoints (
    (COALESCE(metadata->>'lat_user_id', metadata->>'user_id')),
    checkpoint_id DESC, thread_id DESC
)
WHERE checkpoint_ns = '' AND metadata->>'step' = '0'
"""


class InvalidThreadCursor(ValueError):
    """The cursor is invalid or belongs to another conversation scope."""


class ThreadListingUnsupported(NotImplementedError):
    """The checkpointer cannot list saved conversations."""


@dataclass(frozen=True)
class ThreadSummary:
    thread_id: str
    agent_id: str


@dataclass(frozen=True)
class ThreadPage:
    threads: list[ThreadSummary]
    next_cursor: str | None


@dataclass(frozen=True)
class _Candidate:
    checkpoint_id: str
    storage_id: str
    metadata: dict[str, Any]

    @property
    def boundary(self) -> tuple[str, str]:
        return self.checkpoint_id, self.storage_id


def _scope(user_id: str, agent_id: str) -> str:
    value = json.dumps([user_id, agent_id], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def _decode_cursor(cursor: str | None, scope: str) -> tuple[str, str] | None:
    if cursor is None:
        return None
    try:
        if not isinstance(cursor, str) or not 1 <= len(cursor) <= 2048:
            raise ValueError
        data = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        value = json.loads(data)
        if (
            not isinstance(value, list)
            or len(value) != 4
            or type(value[0]) is not int
            or value[0] != 1
            or value[1] != scope
        ):
            raise ValueError
        checkpoint_id, storage_id = value[2:]
        if not isinstance(checkpoint_id, str) or not 1 <= len(checkpoint_id) <= 128:
            raise ValueError
        if not isinstance(storage_id, str) or not 1 <= len(storage_id) <= 256:
            raise ValueError
        if any(ord(char) < 32 for char in checkpoint_id + storage_id):
            raise ValueError
        return checkpoint_id, storage_id
    except (ValueError, TypeError, UnicodeError, binascii.Error, RecursionError) as exc:
        raise InvalidThreadCursor("Invalid thread cursor. Use a cursor from the same user and agent.") from exc


def _encode_cursor(candidate: _Candidate, scope: str) -> str:
    value = json.dumps([1, scope, *candidate.boundary], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _matches_owner(metadata: dict[str, Any], user_id: str, agent_id: str) -> bool:
    owner = metadata.get("lat_user_id", metadata.get("user_id"))
    agent = metadata.get("lat_agent_id", agent_id)
    return owner == user_id and agent == agent_id


def _memory_candidates(
    saver: InMemorySaver, user_id: str, agent_id: str, before: tuple[str, str] | None, limit: int
) -> list[_Candidate]:
    def candidates() -> Iterator[_Candidate]:
        for storage_id, namespaces in saver.storage.items():
            checkpoints = namespaces.get("", {})
            # A new run writes step -1 and step 0. A history import starts at step 0.
            for checkpoint_id in heapq.nsmallest(2, checkpoints):
                metadata = saver.serde.loads_typed(checkpoints[checkpoint_id][1])
                if not isinstance(metadata, dict) or metadata.get("step") != 0:
                    continue
                candidate = _Candidate(checkpoint_id, storage_id, metadata)
                if _matches_owner(metadata, user_id, agent_id) and (before is None or candidate.boundary < before):
                    yield candidate
                break

    return heapq.nlargest(limit, candidates(), key=lambda candidate: candidate.boundary)


async def _sqlite_candidates(
    saver: AsyncSqliteSaver, user_id: str, before: tuple[str, str] | None, limit: int
) -> list[_Candidate]:
    await saver.setup()
    query = """
        SELECT c.checkpoint_id, c.thread_id, c.metadata FROM checkpoints AS c
        WHERE c.checkpoint_ns = '' AND json_extract(c.metadata, '$.step') = 0
          AND COALESCE(json_extract(c.metadata, '$.lat_user_id'), json_extract(c.metadata, '$.user_id')) = ?
          AND NOT EXISTS (
              SELECT 1 FROM checkpoints AS earlier
              WHERE earlier.thread_id = c.thread_id AND earlier.checkpoint_ns = ''
                AND json_extract(earlier.metadata, '$.step') = 0 AND earlier.checkpoint_id < c.checkpoint_id
          )
    """
    params: list[Any] = [user_id]
    if before is not None:
        query += " AND (c.checkpoint_id, c.thread_id) < (?, ?)"
        params.extend(before)
    query += " ORDER BY c.checkpoint_id DESC, c.thread_id DESC LIMIT ?"
    params.append(limit)
    async with saver.lock, saver.conn.execute(query, params) as cursor:
        rows = await cursor.fetchall()
    return [_Candidate(row[0], row[1], json.loads(row[2])) for row in rows]


async def _postgres_candidates(
    saver: AsyncPostgresSaver, user_id: str, before: tuple[str, str] | None, limit: int
) -> list[_Candidate]:
    query = """
        SELECT c.checkpoint_id, c.thread_id, c.metadata FROM checkpoints AS c
        WHERE c.checkpoint_ns = '' AND c.metadata->>'step' = '0'
          AND COALESCE(c.metadata->>'lat_user_id', c.metadata->>'user_id') = %s
          AND NOT EXISTS (
              SELECT 1 FROM checkpoints AS earlier
              WHERE earlier.thread_id = c.thread_id AND earlier.checkpoint_ns = ''
                AND earlier.metadata->>'step' = '0' AND earlier.checkpoint_id < c.checkpoint_id
          )
    """
    params: list[Any] = [user_id]
    if before is not None:
        query += " AND (c.checkpoint_id, c.thread_id) < (%s, %s)"
        params.extend(before)
    query += " ORDER BY c.checkpoint_id DESC, c.thread_id DESC LIMIT %s"
    params.append(limit)
    async with saver._cursor() as cursor:
        await cursor.execute(query, params)
        rows = await cursor.fetchall()
    return [_Candidate(row["checkpoint_id"], row["thread_id"], row["metadata"]) for row in rows]


async def list_threads(
    saver: Any,
    *,
    user_id: str,
    agent_id: str,
    storage_key: Callable[[str], str],
    limit: int = 100,
    cursor: str | None = None,
) -> ThreadPage:
    """Read a bounded page of conversation identities from root checkpoints.

    The cursor orders conversations by their first step 0 checkpoint. It does
    not order them by their latest activity. Other agents in the same user
    scope can require extra pages. Continue until ``next_cursor`` is ``None``,
    including after an empty page. ``storage_key`` must bind public IDs to the
    authenticated owner and selected agent.
    """
    if not 1 <= limit <= 100:
        raise ValueError("Thread page size must be between 1 and 100.")
    scope = _scope(user_id, agent_id)
    before = _decode_cursor(cursor, scope)
    scan_limit = max(limit * 4, 100)
    if saver is None or saver is False or isinstance(saver, NoOpSaver):
        return ThreadPage([], None)
    if isinstance(saver, InMemorySaver):
        candidates = _memory_candidates(saver, user_id, agent_id, before, scan_limit + 1)
    elif isinstance(saver, AsyncSqliteSaver) and getattr(saver, "_lat_thread_listing_indexed", False):
        candidates = await _sqlite_candidates(saver, user_id, before, scan_limit + 1)
    elif isinstance(saver, AsyncPostgresSaver) and getattr(saver, "_lat_thread_listing_indexed", False):
        candidates = await _postgres_candidates(saver, user_id, before, scan_limit + 1)
    else:
        raise ThreadListingUnsupported("This checkpointer does not support conversation listing.")

    threads = []
    for index, candidate in enumerate(candidates[:scan_limit]):
        metadata = candidate.metadata
        if isinstance(metadata, dict) and _matches_owner(metadata, user_id, agent_id):
            public_id = metadata.get("lat_thread_id", metadata.get("langfuse_session_id"))
            if (
                isinstance(public_id, str)
                and 1 <= len(public_id) <= 256
                and storage_key(public_id) == candidate.storage_id
            ):
                threads.append(ThreadSummary(public_id, agent_id))
        if len(threads) == limit or index + 1 == scan_limit:
            next_cursor = _encode_cursor(candidate, scope) if index + 1 < len(candidates) else None
            return ThreadPage(threads, next_cursor)
    return ThreadPage(threads, None)
