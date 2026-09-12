"""Check migration against an explicitly selected disposable PostgreSQL database."""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import psycopg
import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from psycopg import sql
from psycopg.conninfo import make_conninfo

from langgraph_agent_toolkit.core.memory.migration import migrate_postgres, retain_postgres
from langgraph_agent_toolkit.service.auth import storage_thread_id


TEST_DSN = os.environ.get("LAT_TEST_POSTGRES_DSN")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.asyncio,
    pytest.mark.skipif(not TEST_DSN, reason="Set LAT_TEST_POSTGRES_DSN to a disposable PostgreSQL database"),
]


@pytest_asyncio.fixture
async def migration_database():
    schema = "lat_test_migration_" + uuid4().hex
    async with await psycopg.AsyncConnection.connect(TEST_DSN, autocommit=True) as connection:
        await connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield schema, make_conninfo(TEST_DSN, options=f"-c search_path={schema}")
    finally:
        async with await psycopg.AsyncConnection.connect(TEST_DSN, autocommit=True) as connection:
            await connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


async def test_real_postgres_migration_read_append_and_retention(migration_database):
    schema, saver_dsn = migration_database

    def reply(state):
        return {"messages": [AIMessage(content="reply: " + state["messages"][-1].content)]}

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    old_config = {"configurable": {"thread_id": "legacy"}}
    untouched_config = {"configurable": {"thread_id": "unlisted-legacy"}}
    manifest = [{"old_thread_id": "legacy", "user_id": "alice", "agent_id": "chatbot"}]
    owned_key = storage_thread_id("alice", "chatbot", "legacy")
    owned_config = {"configurable": {"thread_id": owned_key}}

    async with AsyncPostgresSaver.from_conn_string(saver_dsn) as saver:
        await saver.setup()
        graph = builder.compile(checkpointer=saver)
        await graph.ainvoke({"messages": [HumanMessage(content="before migration")]}, old_config)
        await graph.ainvoke({"messages": [HumanMessage(content="keep unlisted")]}, untouched_config)

    preview = await asyncio.to_thread(migrate_postgres, TEST_DSN, manifest, schema=schema)
    assert preview["applied"] is False
    assert all(preview["rows"][table] > 0 for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"))
    async with AsyncPostgresSaver.from_conn_string(saver_dsn) as saver:
        graph = builder.compile(checkpointer=saver)
        assert (await graph.aget_state(owned_config)).values == {}
        assert [message.content for message in (await graph.aget_state(old_config)).values["messages"]] == [
            "before migration",
            "reply: before migration",
        ]

    applied = await asyncio.to_thread(migrate_postgres, TEST_DSN, manifest, schema=schema, apply=True)
    assert applied["applied"] is True
    assert applied["rows"] == preview["rows"]
    async with AsyncPostgresSaver.from_conn_string(saver_dsn) as saver:
        graph = builder.compile(checkpointer=saver)
        assert (await graph.aget_state(old_config)).values == {}
        result = await graph.ainvoke({"messages": [HumanMessage(content="after migration")]}, owned_config)
        assert [message.content for message in result["messages"]] == [
            "before migration",
            "reply: before migration",
            "after migration",
            "reply: after migration",
        ]

    cutoff = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
    retention = await asyncio.to_thread(retain_postgres, TEST_DSN, cutoff, schema=schema)
    assert retention["applied"] is False
    assert retention["threads"] == [owned_key]
    async with AsyncPostgresSaver.from_conn_string(saver_dsn) as saver:
        assert await saver.aget_tuple(owned_config) is not None
    removed = await asyncio.to_thread(retain_postgres, TEST_DSN, cutoff, schema=schema, apply=True)
    assert removed["applied"] is True
    assert removed["rows"] == retention["rows"]
    async with AsyncPostgresSaver.from_conn_string(saver_dsn) as saver:
        assert await saver.aget_tuple(owned_config) is None
        assert await saver.aget_tuple(untouched_config) is not None
    async with await psycopg.AsyncConnection.connect(TEST_DSN, autocommit=True) as connection:
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            cursor = await connection.execute(
                sql.SQL("SELECT COUNT(*) FROM {} WHERE thread_id=%s").format(sql.Identifier(schema, table)),
                (owned_key,),
            )
            assert (await cursor.fetchone())[0] == 0
