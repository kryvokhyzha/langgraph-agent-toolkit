"""Plan and apply checkpoint maintenance while service workers are stopped."""

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from langgraph_agent_toolkit.service.auth import storage_thread_id


class MigrationEntry(BaseModel):
    """Map one legacy thread to an explicit owner and agent."""

    model_config = ConfigDict(extra="forbid", strict=True)

    old_thread_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1, max_length=256)
    agent_id: str = Field(min_length=1, max_length=256)
    thread_id: Optional[str] = Field(default=None, min_length=1, max_length=256)

    @property
    def public_thread_id(self) -> str:
        value = self.old_thread_id if self.thread_id is None else self.thread_id
        if len(value) > 256:
            raise ValueError("Set thread_id to a public ID with at most 256 characters")
        return value

    @property
    def destination(self) -> str:
        return storage_thread_id(self.user_id, self.agent_id, self.public_thread_id)


def _validate_manifest(entries: Any) -> list[MigrationEntry]:
    mappings = TypeAdapter(list[MigrationEntry]).validate_python(entries)
    if not mappings:
        raise ValueError("The migration manifest must not be empty")
    sources = [entry.old_thread_id for entry in mappings]
    destinations = [entry.destination for entry in mappings]
    if len(sources) != len(set(sources)):
        raise ValueError("Duplicate old_thread_id in the migration manifest")
    if len(destinations) != len(set(destinations)):
        raise ValueError("Duplicate destination in the migration manifest")
    if set(sources) & set(destinations):
        raise ValueError("Source and destination thread IDs overlap")
    return mappings


def load_manifest(path: str | Path) -> list[MigrationEntry]:
    """Read explicit mappings from a JSON array."""
    return _validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("A timestamp must be an ISO 8601 string with a timezone")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Invalid ISO 8601 timestamp") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("A timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


class _Store:
    def __init__(self, backend: str, database: str | Path, schema: str = "public", apply: bool = False):
        if not isinstance(apply, bool):
            raise ValueError("apply must be a Boolean value")
        self.backend = backend
        self.database = database
        self.schema = schema
        self.apply = apply
        self.connection = None
        self.tables: list[str] = []
        self.parameter = "?" if backend == "sqlite" else "%s"

    def __enter__(self):
        if self.backend == "sqlite":
            path = Path(self.database).resolve()
            if not path.is_file():
                raise FileNotFoundError("The SQLite checkpoint file does not exist")
            mode = "rw" if self.apply else "ro"
            self.connection = sqlite3.connect(
                f"{path.as_uri()}?mode={mode}", uri=True, isolation_level=None, timeout=30
            )
        else:
            import psycopg

            self.connection = psycopg.connect(str(self.database), autocommit=False)
        try:
            if self.backend == "sqlite":
                self.connection.execute("BEGIN IMMEDIATE" if self.apply else "BEGIN")
                existing = {
                    row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                required, optional = ["checkpoints"], ["writes"]
            else:
                if self.apply:
                    self.connection.execute("SET LOCAL lock_timeout = '30s'")
                else:
                    self.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                existing = {
                    row[0]
                    for row in self.connection.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = %s AND table_type = 'BASE TABLE'",
                        (self.schema,),
                    )
                }
                required, optional = ["checkpoints", "checkpoint_blobs"], ["checkpoint_writes"]
            missing = set(required) - existing
            if missing:
                raise ValueError(f"Missing checkpoint tables: {', '.join(sorted(missing))}")
            self.tables = required + [table for table in optional if table in existing]
            if self.backend == "postgres" and self.apply:
                tables = ", ".join(self.table(table) for table in self.tables)
                self.connection.execute(f"LOCK TABLE {tables} IN SHARE ROW EXCLUSIVE MODE")
            return self
        except BaseException:
            self.connection.rollback()
            self.connection.close()
            raise

    def __exit__(self, exception_type, exception, traceback):
        try:
            if exception_type is None and self.apply:
                self.connection.commit()
            else:
                self.connection.rollback()
        finally:
            self.connection.close()

    def table(self, name: str) -> str:
        name = _identifier(name)
        return name if self.backend == "sqlite" else f"{_identifier(self.schema)}.{name}"

    def counts(self, thread_ids: list[str]) -> dict[str, dict[str, int]]:
        """Count rows in bounded batches without reading checkpoint content."""
        counts = {table: {} for table in self.tables}
        for table in self.tables:
            for offset in range(0, len(thread_ids), 500):
                batch = thread_ids[offset : offset + 500]
                parameters = ", ".join([self.parameter] * len(batch))
                query = (
                    f"SELECT thread_id, COUNT(*) FROM {self.table(table)} "
                    f"WHERE thread_id IN ({parameters}) GROUP BY thread_id"
                )
                counts[table].update(dict(self.connection.execute(query, batch).fetchall()))
        return counts

    def latest_timestamps(self) -> dict[str, datetime]:
        latest = {}
        if self.backend == "postgres":
            query = (
                f"SELECT thread_id, MAX((checkpoint->>'ts')::timestamptz), "
                "COUNT(*) FILTER (WHERE NOT COALESCE((checkpoint->>'ts') ~ '(Z|[+-][0-9]{2}:[0-9]{2})$', FALSE)) "
                f"FROM {self.table('checkpoints')} WHERE thread_id ~ '^lat:v1:[0-9a-f]{{64}}$' GROUP BY thread_id"
            )
            for thread_id, timestamp, missing in self.connection.execute(query):
                if not re.fullmatch(r"lat:v1:[0-9a-f]{64}", thread_id):
                    continue
                if missing or timestamp is None:
                    raise ValueError("A checkpoint timestamp is missing; retention stopped")
                latest[thread_id] = timestamp.astimezone(timezone.utc)
            return latest

        query = f"SELECT thread_id, type, checkpoint FROM {self.table('checkpoints')} WHERE thread_id LIKE 'lat:v1:%'"
        for thread_id, encoding, data in self.connection.execute(query):
            if not re.fullmatch(r"lat:v1:[0-9a-f]{64}", thread_id):
                continue
            if encoding == "json":
                checkpoint = json.loads(data)
            elif encoding == "msgpack":
                import ormsgpack

                # Ignore extension objects. Retention only reads the plain timestamp.
                checkpoint = ormsgpack.unpackb(
                    data, ext_hook=lambda code, value: None, option=ormsgpack.OPT_NON_STR_KEYS
                )
            else:
                raise ValueError("Unsupported checkpoint encoding; retention stopped")
            timestamp = _timestamp(checkpoint.get("ts") if isinstance(checkpoint, dict) else None)
            latest[thread_id] = max(latest.get(thread_id, timestamp), timestamp)
        return latest


def _migrate(store: _Store, mappings: list[MigrationEntry]) -> dict[str, Any]:
    thread_ids = [entry.old_thread_id for entry in mappings] + [entry.destination for entry in mappings]
    counts = store.counts(thread_ids)
    for entry in mappings:
        if not counts["checkpoints"].get(entry.old_thread_id):
            raise ValueError(f"Source thread {entry.old_thread_id!r} has no checkpoints")
        if any(counts[table].get(entry.destination) for table in store.tables):
            raise ValueError(f"Destination for {entry.old_thread_id!r} already exists")
    rows = {table: sum(counts[table].get(entry.old_thread_id, 0) for entry in mappings) for table in store.tables}
    if store.apply:
        for entry in mappings:
            for table in store.tables:
                query = (
                    f"UPDATE {store.table(table)} SET thread_id = {store.parameter} WHERE thread_id = {store.parameter}"
                )
                store.connection.execute(query, (entry.destination, entry.old_thread_id))
    return {
        "operation": "migration",
        "applied": store.apply,
        "rows": rows,
        "threads": [
            {**entry.model_dump(), "thread_id": entry.public_thread_id, "storage_thread_id": entry.destination}
            for entry in mappings
        ],
    }


def migrate_sqlite(path: str | Path, entries: Any, *, apply: bool = False) -> dict[str, Any]:
    """Plan or atomically apply a SQLite thread migration."""
    mappings = _validate_manifest(entries)
    with _Store("sqlite", path, apply=apply) as store:
        return _migrate(store, mappings)


def migrate_postgres(conninfo: str, entries: Any, *, schema: str = "public", apply: bool = False) -> dict[str, Any]:
    """Plan or atomically apply a PostgreSQL thread migration."""
    mappings = _validate_manifest(entries)
    with _Store("postgres", conninfo, schema=schema, apply=apply) as store:
        return _migrate(store, mappings)


def _retain(store: _Store, before: datetime) -> dict[str, Any]:
    latest = store.latest_timestamps()
    threads = sorted(thread_id for thread_id, timestamp in latest.items() if timestamp < before)
    counts = store.counts(threads)
    rows = {table: sum(counts[table].values()) for table in store.tables}
    if store.apply:
        for thread_id in threads:
            for table in store.tables:
                store.connection.execute(
                    f"DELETE FROM {store.table(table)} WHERE thread_id = {store.parameter}", (thread_id,)
                )
    return {
        "operation": "retention",
        "applied": store.apply,
        "before": before.isoformat(),
        "rows": rows,
        "threads": threads,
    }


def retain_sqlite(path: str | Path, before: str, *, apply: bool = False) -> dict[str, Any]:
    """Plan or delete complete inactive SQLite threads before a timestamp."""
    cutoff = _timestamp(before)
    with _Store("sqlite", path, apply=apply) as store:
        return _retain(store, cutoff)


def retain_postgres(conninfo: str, before: str, *, schema: str = "public", apply: bool = False) -> dict[str, Any]:
    """Plan or delete complete inactive PostgreSQL threads before a timestamp."""
    cutoff = _timestamp(before)
    with _Store("postgres", conninfo, schema=schema, apply=apply) as store:
        return _retain(store, cutoff)


def main(
    backend: str,
    manifest: Optional[str] = None,
    before: Optional[str] = None,
    sqlite_path: Optional[str] = None,
    postgres_env: str = "DATABASE_URL",
    schema: str = "public",
    apply: bool = False,
) -> None:
    """Preview maintenance. Set --apply=true to commit after review."""
    if isinstance(apply, str):
        if apply.lower() not in ("true", "false"):
            raise ValueError("--apply must be true or false")
        apply = apply.lower() == "true"
    if bool(manifest) == bool(before):
        raise ValueError("Set exactly one of --manifest or --before")
    if backend not in ("sqlite", "postgres"):
        raise ValueError("backend must be sqlite or postgres")
    value = load_manifest(manifest) if manifest else before
    if backend == "sqlite":
        if not sqlite_path:
            raise ValueError("SQLite maintenance requires --sqlite-path")
        operation = migrate_sqlite if manifest else retain_sqlite
        result = operation(sqlite_path, value, apply=apply)
    else:
        conninfo = os.environ.get(postgres_env)
        if not conninfo:
            raise ValueError("The named PostgreSQL connection environment variable is empty")
        operation = migrate_postgres if manifest else retain_postgres
        result = operation(conninfo, value, schema=schema, apply=apply)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
