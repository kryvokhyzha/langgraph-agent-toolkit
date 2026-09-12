"""Apply bounded pressure to an explicitly selected disposable database."""

import asyncio
import ipaddress
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import monotonic
from urllib.parse import quote

import httpx
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from langgraph_agent_toolkit.core.memory.pool import CancellableAsyncConnection


_ROLES = frozenset({"saver", "store", "locks"})


def _require_loopback(host: str) -> None:
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise ValueError("Load tests require a literal loopback address.")


@dataclass(frozen=True)
class LoadDatabase:
    """Identify one disposable schema and its unique application tags."""

    dsn: str = field(repr=False)
    schema: str
    application_prefix: str

    def validate(self) -> None:
        if os.environ.get("LAT_LOAD_TEST_DATABASE") != "yes":
            raise ValueError("Set LAT_LOAD_TEST_DATABASE=yes for disposable database tests.")
        values = conninfo_to_dict(self.dsn)
        _require_loopback(values.get("host", ""))
        if values.get("hostaddr"):
            _require_loopback(values["hostaddr"])
        if values.get("service") or not values.get("dbname"):
            raise ValueError("Use an explicit database name without a libpq service.")
        if not re.fullmatch(r"lat_load_[a-z0-9_]{1,54}", self.schema):
            raise ValueError("Use a unique lat_load_ schema with at most 63 characters.")
        if not re.fullmatch(r"lat-load-[a-z0-9_-]{1,42}", self.application_prefix):
            raise ValueError("Use a unique lat-load- application prefix.")

    def application_names(self, roles: Sequence[str] = ("saver", "store", "locks")) -> list[str]:
        if not roles or not set(roles) <= _ROLES:
            raise ValueError("Select only saver, store, or locks pool roles.")
        return [f"{self.application_prefix}-{role}" for role in sorted(set(roles))]


@asynccontextmanager
async def _connection(target: LoadDatabase, *, timeout: float = 5, statement_timeout_ms: int = 3000):
    target.validate()
    values = conninfo_to_dict(target.dsn)
    async with asyncio.timeout(timeout):
        connection = await CancellableAsyncConnection.connect(
            target.dsn,
            # Do not let an exported PGHOSTADDR override the validated destination.
            hostaddr=values.get("hostaddr") or values["host"],
            autocommit=True,
            row_factory=dict_row,
            connect_timeout=3,
            application_name=f"{target.application_prefix}-observer",
            options=f"-c statement_timeout={statement_timeout_ms} -c lock_timeout=1000",
        )
    async with connection:
        row = await (await connection.execute("SELECT to_regnamespace(%s) AS schema_oid", (target.schema,))).fetchone()
        if row["schema_oid"] is None:
            raise ValueError("The selected disposable schema does not exist.")
        yield connection


async def gather_db_metrics(target: LoadDatabase) -> dict:
    """Read tagged session counts and database counters without SQL text."""
    async with _connection(target) as connection:
        groups = await (
            await connection.execute(
                "SELECT application_name, state, wait_event_type, wait_event, count(*) AS sessions, "
                "COALESCE(max(extract(epoch FROM clock_timestamp() - xact_start)), 0)::float AS oldest_transaction_s, "
                "COALESCE(max(extract(epoch FROM clock_timestamp() - query_start)) "
                "FILTER (WHERE state = 'active'), 0)::float AS longest_active_query_s "
                "FROM pg_stat_activity WHERE datname = current_database() AND application_name = ANY(%s) "
                "GROUP BY application_name, state, wait_event_type, wait_event ORDER BY application_name, state",
                (target.application_names(),),
            )
        ).fetchall()
        locks = await (
            await connection.execute(
                "SELECT count(*) FILTER (WHERE granted) AS granted, count(*) FILTER (WHERE NOT granted) AS waiting "
                "FROM pg_locks JOIN pg_stat_activity USING (pid) WHERE datname = current_database() "
                "AND application_name = ANY(%s) AND locktype = 'advisory'",
                (target.application_names(),),
            )
        ).fetchone()
        counters = await (
            await connection.execute(
                "SELECT xact_commit, xact_rollback, deadlocks, temp_files, temp_bytes, blks_read, blks_hit "
                "FROM pg_stat_database WHERE datname = current_database()"
            )
        ).fetchone()
    return {
        "sessions": groups,
        "session_count": sum(row["sessions"] for row in groups),
        "advisory_locks": locks,
        "database_counters": counters,
    }


async def hold_checkpoint_table_lock(
    target: LoadDatabase,
    release_event: asyncio.Event,
    acquired_event: asyncio.Event | None = None,
    *,
    max_hold_seconds: float = 30,
) -> None:
    """Block checkpoint reads and writes until release or a bounded timeout."""
    if not 0 < max_hold_seconds <= 120:
        raise ValueError("Lock duration must be greater than zero and at most 120 seconds.")
    async with _connection(target) as connection:
        target.validate()
        async with connection.transaction():
            await connection.execute(
                sql.SQL("LOCK TABLE {}.checkpoints IN ACCESS EXCLUSIVE MODE NOWAIT").format(
                    sql.Identifier(target.schema)
                )
            )
            if acquired_event is not None:
                acquired_event.set()
            await asyncio.wait_for(release_event.wait(), max_hold_seconds)


async def run_sleep_probe(target: LoadDatabase, seconds: float = 0.1, *, timeout: float = 5) -> dict:
    """Measure one disposable database query without running a model or graph."""
    if not 0 <= seconds <= 30 or not 0 < timeout <= 60:
        raise ValueError("Sleep must be 0 to 30 seconds. Timeout must be greater than zero and at most 60 seconds.")
    started = monotonic()
    async with _connection(target, statement_timeout_ms=max(1, int(timeout * 1000))) as connection:
        async with asyncio.timeout(timeout):
            await connection.execute("SELECT pg_sleep(%s)", (seconds,))
    return {"requested_sleep_s": seconds, "elapsed_s": monotonic() - started}


async def terminate_toolkit_sessions(
    target: LoadDatabase, *, roles: Sequence[str] = ("locks",), limit: int | None = None
) -> list[dict]:
    """Drop only exact tagged pool sessions in the selected disposable database."""
    if limit is not None and not 1 <= limit <= 1000:
        raise ValueError("Session termination limit must be 1 to 1000.")
    names = target.application_names(roles)
    async with _connection(target) as connection:
        target.validate()
        return await (
            await connection.execute(
                "WITH selected AS (SELECT pid FROM pg_stat_activity WHERE datname = current_database() "
                "AND application_name = ANY(%s) AND backend_type = 'client backend' AND pid <> pg_backend_pid() "
                "ORDER BY backend_start, pid LIMIT %s) "
                "SELECT pid, pg_terminate_backend(pid) AS terminated FROM selected",
                (names, limit or 1000),
            )
        ).fetchall()


async def checkpoint_counts(target: LoadDatabase, *, storage_thread_id: str | None = None) -> dict:
    """Count stored checkpoints after the measured traffic has stopped."""
    async with _connection(target) as connection:
        query = sql.SQL(
            "SELECT thread_id, count(*) AS checkpoints, count(DISTINCT checkpoint_id) AS checkpoint_ids "
            "FROM {}.checkpoints"
        ).format(sql.Identifier(target.schema))
        params = ()
        if storage_thread_id is not None:
            query += sql.SQL(" WHERE thread_id = %s")
            params = (storage_thread_id,)
        query += sql.SQL(" GROUP BY thread_id ORDER BY thread_id")
        rows = await (await connection.execute(query, params)).fetchall()
    return {"threads": rows, "thread_count": len(rows), "checkpoints": sum(row["checkpoints"] for row in rows)}


def _differences(actual: Sequence[str], expected: Sequence[str]) -> dict:
    actual_counts, expected_counts = Counter(actual), Counter(expected)
    return {
        "missing": list((expected_counts - actual_counts).elements()),
        "unexpected": list((actual_counts - expected_counts).elements()),
        "duplicates": {
            value: count for value, count in actual_counts.items() if count > 1 and count > expected_counts[value]
        },
    }


async def verify_http_history(
    client: httpx.AsyncClient,
    agent_id: str,
    thread_id: str,
    expected_human_messages: Sequence[str],
    *,
    user_id: str | None = None,
    expected_ai_messages: Sequence[str] | None = None,
    expected_pairs: Mapping[str, str] | None = None,
    timeout: float = 10,
    max_messages: int = 10000,
) -> dict:
    """Report missing turns and reply order through the public history contract."""
    _require_loopback(client.base_url.host)
    messages, totals = [], []
    offset = 0
    async with asyncio.timeout(timeout):
        while True:
            params = {"thread_id": thread_id, "offset": offset, "limit": 1000}
            if user_id is not None:
                params["user_id"] = user_id
            response = await client.get(f"/{quote(agent_id, safe='')}/history", params=params)
            response.raise_for_status()
            page = response.json()
            totals.append(page["total"])
            messages.extend(page["messages"])
            if len(messages) > max_messages:
                raise ValueError("History exceeds the configured load-test message limit.")
            next_offset = page.get("next_offset")
            if next_offset is None:
                break
            if next_offset <= offset:
                raise ValueError("History pagination did not advance.")
            offset = next_offset
    human = [message["content"] for message in messages if message["type"] == "human"]
    ai = [message["content"] for message in messages if message["type"] == "ai"]
    human_diff = _differences(human, expected_human_messages)
    ai_diff = _differences(ai, expected_ai_messages) if expected_ai_messages is not None else None
    order_errors = []
    if expected_pairs is not None:
        current_human = current_reply = None

        def check_turn():
            if current_human in expected_pairs and current_reply != expected_pairs[current_human]:
                order_errors.append(current_human)

        for message in messages:
            if message["type"] == "human":
                check_turn()
                current_human, current_reply = message["content"], None
            elif message["type"] == "ai":
                current_reply = message["content"]
        check_turn()
    stable_total = len(set(totals)) == 1 and totals[0] == len(messages)
    return {
        "ok": stable_total
        and not any(human_diff.values())
        and not (ai_diff and any(ai_diff.values()))
        and not order_errors,
        "total": len(messages),
        "human_count": len(human),
        "ai_count": len(ai),
        "human_order": human,
        "human": human_diff,
        "ai": ai_diff,
        "reply_order_errors": order_errors,
        "stable_total": stable_total,
    }
