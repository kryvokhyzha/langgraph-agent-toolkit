import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from langgraph_agent_toolkit.core.memory.migration import load_manifest, migrate_postgres, migrate_sqlite, retain_sqlite
from langgraph_agent_toolkit.service.auth import storage_thread_id


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "checkpoints.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
            "type TEXT, checkpoint BLOB, PRIMARY KEY(thread_id, checkpoint_ns, checkpoint_id));"
            "CREATE TABLE writes (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, value BLOB);"
        )
        connection.executemany(
            "INSERT INTO checkpoints VALUES (?, ?, ?, 'json', ?)",
            [
                ("old", "", "1", b'{"ts":"2025-01-01T00:00:00Z"}'),
                ("old", "child", "2", b'{"ts":"2025-02-01T00:00:00Z"}'),
            ],
        )
        connection.execute("INSERT INTO writes VALUES ('old', '', '1', ?)", (b"unchanged payload",))
    return path


def entries(old="old", user="alice", agent="chatbot", thread=None):
    value = {"old_thread_id": old, "user_id": user, "agent_id": agent}
    if thread is not None:
        value["thread_id"] = thread
    return [value]


def test_sqlite_dry_run_and_apply_preserve_all_checkpoint_namespaces(database):
    before = Path(database).read_bytes()
    plan = migrate_sqlite(database, entries())
    assert plan["applied"] is False
    assert plan["rows"] == {"checkpoints": 2, "writes": 1}
    assert Path(database).read_bytes() == before
    result = migrate_sqlite(database, entries(), apply=True)
    assert result["applied"] is True
    expected = storage_thread_id("alice", "chatbot", "old")
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall() == [(expected,)]
        assert connection.execute("SELECT thread_id, value FROM writes").fetchall() == [
            (expected, b"unchanged payload")
        ]


@pytest.mark.parametrize("table", ["checkpoints", "writes"])
def test_sqlite_rejects_destination_collisions_without_changes(database, table):
    target = storage_thread_id("alice", "chatbot", "old")
    with sqlite3.connect(database) as connection:
        if table == "writes":
            connection.execute("INSERT INTO writes VALUES (?, '', 'other', 'x')", (target,))
        else:
            connection.execute("INSERT INTO checkpoints VALUES (?, '', 'other', 'json', '{}')", (target,))
    before = Path(database).read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        migrate_sqlite(database, entries(), apply=True)
    assert Path(database).read_bytes() == before


def test_sqlite_migration_rolls_back_all_tables_on_write_failure(database):
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TRIGGER stop_write BEFORE UPDATE ON writes BEGIN SELECT RAISE(ABORT, 'simulated failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="simulated failure"):
        migrate_sqlite(database, entries(), apply=True)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall() == [("old",)]
        assert connection.execute("SELECT DISTINCT thread_id FROM writes").fetchall() == [("old",)]


def test_manifest_requires_explicit_owner_and_unique_mappings(database, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(entries()))
    assert load_manifest(path)[0].user_id == "alice"
    for invalid in [
        [{"old_thread_id": "old", "agent_id": "chatbot"}],
        entries() + entries(),
        entries() + entries(old="other", thread="old"),
        entries() + entries(old=storage_thread_id("alice", "chatbot", "old")),
    ]:
        with pytest.raises(ValueError):
            migrate_sqlite(database, invalid, apply=True)


def test_missing_source_and_database_fail_without_creation(database, tmp_path):
    with pytest.raises(ValueError, match="no checkpoints"):
        migrate_sqlite(database, entries(old="missing"), apply=True)
    missing = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        migrate_sqlite(missing, entries())
    assert not missing.exists()


def test_only_known_optional_table_can_be_missing(database):
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE writes")
    assert migrate_sqlite(database, entries())["rows"] == {"checkpoints": 2}
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE checkpoints")
    with pytest.raises(ValueError, match="checkpoints"):
        migrate_sqlite(database, entries())


def test_retention_uses_latest_checkpoint_across_namespaces(database):
    migrate_sqlite(database, entries(), apply=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO checkpoints VALUES ('legacy', '', 'legacy', 'json', ?)", (b'{"ts":"2020-01-01T00:00:00Z"}',)
        )
    assert retain_sqlite(database, "2025-01-15T00:00:00Z")["threads"] == []
    plan = retain_sqlite(database, "2025-03-01T00:00:00Z")
    assert plan["applied"] is False
    assert len(plan["threads"]) == 1
    result = retain_sqlite(database, "2025-03-01T00:00:00Z", apply=True)
    assert result["rows"] == {"checkpoints": 2, "writes": 1}
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT thread_id FROM checkpoints").fetchall() == [("legacy",)]
        assert connection.execute("SELECT COUNT(*) FROM writes").fetchone()[0] == 0


@pytest.mark.parametrize("cutoff", ["yesterday", "2025-01-01", "2025-01-01T00:00:00"])
def test_retention_rejects_ambiguous_timestamps(database, cutoff):
    with pytest.raises(ValueError, match="timestamp"):
        retain_sqlite(database, cutoff, apply=True)


def test_retention_rejects_missing_checkpoint_timestamp(database):
    migrate_sqlite(database, entries(), apply=True)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE checkpoints SET checkpoint = '{}' WHERE checkpoint_ns = 'child'")
    with pytest.raises(ValueError, match="timestamp"):
        retain_sqlite(database, "2025-03-01T00:00:00Z", apply=True)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 2


@pytest.mark.parametrize("apply", [False, True])
def test_postgres_plan_and_apply_include_all_checkpoint_tables(apply):
    connection = MagicMock()
    statements = []

    def execute(query, parameters=()):
        statements.append((query, parameters))
        if "information_schema.tables" in query:
            return [("checkpoints",), ("checkpoint_blobs",), ("checkpoint_writes",)]
        if "COUNT(*)" in query:
            cursor = MagicMock()
            cursor.fetchall.return_value = [("old", 2)]
            return cursor
        return MagicMock()

    connection.execute.side_effect = execute
    with patch("psycopg.connect", return_value=connection):
        result = migrate_postgres("dummy", entries(), schema="checkpoint_schema", apply=apply)
    assert result["rows"] == {"checkpoints": 2, "checkpoint_blobs": 2, "checkpoint_writes": 2}
    updates = [(query, params) for query, params in statements if query.startswith("UPDATE")]
    if apply:
        assert len(updates) == 3
        assert all('"checkpoint_schema".' in query for query, _ in updates)
        expected = (storage_thread_id("alice", "chatbot", "old"), "old")
        assert all(params == expected for _, params in updates)
        assert any(query.startswith("LOCK TABLE") for query, _ in statements)
        connection.commit.assert_called_once()
        connection.rollback.assert_not_called()
    else:
        assert updates == []
        assert "READ ONLY" in statements[0][0]
        connection.commit.assert_not_called()
        connection.rollback.assert_called_once()
    connection.close.assert_called_once()


def test_postgres_write_failure_rolls_back_the_migration():
    connection = MagicMock()

    def execute(query, parameters=()):
        if "information_schema.tables" in query:
            return [("checkpoints",), ("checkpoint_blobs",), ("checkpoint_writes",)]
        if "COUNT(*)" in query:
            cursor = MagicMock()
            cursor.fetchall.return_value = [("old", 1)]
            return cursor
        if query.startswith('UPDATE "public"."checkpoint_blobs"'):
            raise RuntimeError("simulated write failure")
        return MagicMock()

    connection.execute.side_effect = execute
    with patch("psycopg.connect", return_value=connection):
        with pytest.raises(RuntimeError, match="simulated write failure"):
            migrate_postgres("dummy", entries(), apply=True)
    connection.commit.assert_not_called()
    connection.rollback.assert_called_once()
    connection.close.assert_called_once()


def test_retention_reads_msgpack_without_reconstructing_objects(database):
    import ormsgpack

    migrate_sqlite(database, entries(), apply=True)
    checkpoint = ormsgpack.packb({"ts": "2025-01-01T00:00:00Z", "channel_values": {"messages": []}})
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE checkpoints SET type='msgpack', checkpoint=?", (checkpoint,))
    assert len(retain_sqlite(database, "2025-02-01T00:00:00Z")["threads"]) == 1


def test_cli_defaults_to_dry_run_and_requires_explicit_apply(database, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries()))
    command = [
        sys.executable,
        "-m",
        "langgraph_agent_toolkit.core.memory.migration",
        "--backend=sqlite",
        f"--sqlite-path={database}",
        f"--manifest={manifest}",
    ]
    environment = {"PATH": os.environ.get("PATH", ""), "PYTHON_DOTENV_DISABLED": "1", "ENV_MODE": "development"}
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["applied"] is False
    result = subprocess.run(command + ["--apply=false"], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["applied"] is False
    result = subprocess.run(command + ["--apply=true"], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["applied"] is True


@pytest.mark.asyncio
async def test_migrated_real_sqlite_graph_resumes_subgraph_and_keeps_history(tmp_path):
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.types import Command, interrupt

    def approved_reply(state):
        decision = interrupt("Approve the reply")
        return {"messages": [AIMessage(content=f"{decision}: {state['messages'][-1].content}")]}

    child = StateGraph(MessagesState)
    child.add_node("reply", approved_reply)
    child.add_edge(START, "reply")
    child.add_edge("reply", END)
    parent = StateGraph(MessagesState)
    parent.add_node("assistant", child.compile())
    parent.add_edge(START, "assistant")
    parent.add_edge("assistant", END)
    path = tmp_path / "real-graph.db"
    original = {"configurable": {"thread_id": "legacy"}}
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        graph = parent.compile(checkpointer=saver)
        result = await graph.ainvoke({"messages": [HumanMessage(content="before migration")]}, original)
        assert result["__interrupt__"]
        assert (await graph.aget_state(original)).tasks[0].interrupts

    plan = await asyncio.to_thread(migrate_sqlite, path, entries(old="legacy"))
    assert plan["rows"]["writes"] > 0
    await asyncio.to_thread(migrate_sqlite, path, entries(old="legacy"), apply=True)
    target = storage_thread_id("alice", "chatbot", "legacy")
    owned = {"configurable": {"thread_id": target}}
    with sqlite3.connect(path) as connection:
        namespaces = connection.execute("SELECT DISTINCT checkpoint_ns FROM checkpoints").fetchall()
        assert any(namespace for (namespace,) in namespaces)
        assert connection.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall() == [(target,)]

    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        graph = parent.compile(checkpointer=saver)
        assert (await graph.aget_state(original)).values == {}
        result = await graph.ainvoke(Command(resume="approved"), owned)
        assert [message.content for message in result["messages"]] == [
            "before migration",
            "approved: before migration",
        ]
        result = await graph.ainvoke({"messages": [HumanMessage(content="after migration")]}, owned)
        assert result["__interrupt__"]
        result = await graph.ainvoke(Command(resume="approved again"), owned)
        assert [message.content for message in result["messages"]] == [
            "before migration",
            "approved: before migration",
            "after migration",
            "approved again: after migration",
        ]
