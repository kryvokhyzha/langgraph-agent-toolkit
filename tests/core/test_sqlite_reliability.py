"""Check SQLite cursor cleanup and failed write isolation."""

import asyncio
import sqlite3
from contextlib import aclosing

import pytest
from langgraph.checkpoint.base import empty_checkpoint

from langgraph_agent_toolkit.core.memory.sqlite import SQLiteMemoryBackend
from langgraph_agent_toolkit.core.settings import settings


pytestmark = pytest.mark.asyncio


async def test_checkpoint_iterator_releases_lock_before_yield(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "history.db"))
    async with SQLiteMemoryBackend().get_checkpoint_saver() as saver:
        config = {"configurable": {"thread_id": "history", "checkpoint_ns": ""}}
        saved = await saver.aput(config, empty_checkpoint(), {}, {})
        async with aclosing(saver.alist(config, limit=2)) as checkpoints:
            async for checkpoint in checkpoints:
                async with asyncio.timeout(0.2):
                    assert await saver.aget_tuple(checkpoint.config) is not None
        assert await saver.aget_tuple(saved) is not None


async def test_cancelled_sqlite_write_does_not_commit_with_the_next_request(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "cancel.db"))
    async with SQLiteMemoryBackend().get_checkpoint_saver() as saver:
        await saver.setup()
        before_commit = asyncio.Event()
        original_commit = saver.conn.commit

        async def delayed_commit():
            before_commit.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(saver.conn, "commit", delayed_commit)
        cancelled_config = {"configurable": {"thread_id": "cancelled", "checkpoint_ns": ""}}
        pending = asyncio.create_task(saver.aput(cancelled_config, empty_checkpoint(), {}, {}))
        await asyncio.wait_for(before_commit.wait(), 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        monkeypatch.setattr(saver.conn, "commit", original_commit)
        successful_config = {"configurable": {"thread_id": "successful", "checkpoint_ns": ""}}
        await saver.aput(successful_config, empty_checkpoint(), {}, {})
        assert await saver.aget_tuple(cancelled_config) is None
        assert await saver.aget_tuple(successful_config) is not None


async def test_partial_sqlite_write_batch_rolls_back_before_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "batch.db"))
    async with SQLiteMemoryBackend().get_checkpoint_saver() as saver:
        config = {"configurable": {"thread_id": "batch", "checkpoint_ns": ""}}
        saved = await saver.aput(config, empty_checkpoint(), {}, {})
        await saver.conn.execute(
            "CREATE TRIGGER reject_write BEFORE INSERT ON writes WHEN NEW.channel = 'reject' "
            "BEGIN SELECT RAISE(ABORT, 'rejected write'); END"
        )
        await saver.conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="rejected write"):
            await saver.aput_writes(saved, [("accepted", "partial"), ("reject", "failure")], "failed-task")
        await saver.aput_writes(saved, [("accepted", "complete")], "successful-task")
        checkpoint = await saver.aget_tuple(saved)
        assert checkpoint.pending_writes == [("successful-task", "accepted", "complete")]


async def test_repeated_cancellation_keeps_sqlite_lock_until_rollback(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "repeat.db"))
    async with SQLiteMemoryBackend().get_checkpoint_saver() as saver:
        await saver.setup()
        before_commit = asyncio.Event()
        before_rollback = asyncio.Event()
        release_rollback = asyncio.Event()
        original_commit = saver.conn.commit
        original_rollback = saver.conn.rollback

        async def delayed_commit():
            before_commit.set()
            await asyncio.Event().wait()

        async def delayed_rollback():
            before_rollback.set()
            await release_rollback.wait()
            await original_rollback()

        monkeypatch.setattr(saver.conn, "commit", delayed_commit)
        monkeypatch.setattr(saver.conn, "rollback", delayed_rollback)
        config = {"configurable": {"thread_id": "cancelled", "checkpoint_ns": ""}}
        pending = asyncio.create_task(saver.aput(config, empty_checkpoint(), {}, {}))
        try:
            await asyncio.wait_for(before_commit.wait(), 1)
            pending.cancel()
            await asyncio.wait_for(before_rollback.wait(), 1)
            pending.cancel()
            await asyncio.sleep(0)
            assert saver.lock.locked()
        finally:
            release_rollback.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 1)
        monkeypatch.setattr(saver.conn, "commit", original_commit)
        assert not saver.lock.locked()
        assert await saver.aget_tuple(config) is None


async def test_cancellation_during_failed_write_rollback_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "cancel-rollback.db"))
    async with SQLiteMemoryBackend().get_checkpoint_saver() as saver:
        config = {"configurable": {"thread_id": "failed", "checkpoint_ns": ""}}
        saved = await saver.aput(config, empty_checkpoint(), {}, {})
        await saver.conn.execute(
            "CREATE TRIGGER reject_write BEFORE INSERT ON writes WHEN NEW.channel = 'reject' "
            "BEGIN SELECT RAISE(ABORT, 'rejected write'); END"
        )
        await saver.conn.commit()
        before_rollback = asyncio.Event()
        release_rollback = asyncio.Event()
        original_rollback = saver.conn.rollback

        async def delayed_rollback():
            before_rollback.set()
            await release_rollback.wait()
            await original_rollback()

        monkeypatch.setattr(saver.conn, "rollback", delayed_rollback)
        pending = asyncio.create_task(
            saver.aput_writes(saved, [("accepted", "partial"), ("reject", "failure")], "failed-task")
        )
        try:
            await asyncio.wait_for(before_rollback.wait(), 1)
            pending.cancel()
            await asyncio.sleep(0)
            assert saver.lock.locked()
            assert not pending.done()
        finally:
            release_rollback.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 1)

        assert pending.cancelled()
        assert not saver.lock.locked()
        await saver.aput_writes(saved, [("accepted", "complete")], "successful-task")
        checkpoint = await saver.aget_tuple(config)
        assert checkpoint.pending_writes == [("successful-task", "accepted", "complete")]


@pytest.mark.parametrize("operation", ["write", "clear"])
async def test_failed_sqlite_checkpoint_change_keeps_saved_history(tmp_path, monkeypatch, operation):
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "failure.sqlite"))
    async with SQLiteMemoryBackend().get_checkpoint_saver() as saver:
        config = {"configurable": {"thread_id": "retained", "checkpoint_ns": ""}}
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"]["messages"] = ["saved message"]
        saved = await saver.aput(config, checkpoint, {}, {})
        await saver.aput_writes(saved, [("messages", "saved pending write")], "completed-task")
        before = await saver.aget_tuple(config)
        event, table = ("INSERT", "checkpoints") if operation == "write" else ("DELETE", "writes")
        await saver.conn.execute(
            f"CREATE TRIGGER reject_change BEFORE {event} ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'rejected checkpoint change'); END"
        )
        await saver.conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="rejected checkpoint change"):
            if operation == "write":
                await saver.aput(saved, empty_checkpoint(), {}, {})
            else:
                await saver.adelete_thread("retained")
        after = await saver.aget_tuple(config)
        assert after == before
        await saver.conn.execute("DROP TRIGGER reject_change")
        await saver.conn.commit()
        await saver.adelete_thread("retained")
        assert await saver.aget_tuple(config) is None
