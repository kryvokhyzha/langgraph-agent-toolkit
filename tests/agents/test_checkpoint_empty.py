import uuid

from langgraph.checkpoint.base import empty_checkpoint

from langgraph_agent_toolkit.agents.components.checkpoint.empty import NoOpSaver


def test_sync_writes_return_config_without_retaining_state():
    """Keep request configuration and discard checkpoints and pending writes."""
    saver = NoOpSaver()
    cfg = {"configurable": {"thread_id": "t1", "user_id": "u1"}, "tags": ["request"]}

    result = saver.put(cfg, checkpoint=empty_checkpoint(), metadata={}, new_versions={})
    saver.put_writes(result, [("channel", "value")], "task-1")

    uuid.UUID(result["configurable"]["checkpoint_id"])
    assert result["configurable"]["checkpoint_ns"] == ""
    assert result["configurable"]["thread_id"] == "t1"
    assert result["configurable"]["user_id"] == "u1"
    assert result["tags"] == ["request"]
    assert "checkpoint_id" not in cfg["configurable"]
    assert saver.get(result) is None
    assert saver.get_tuple(result) is None
    assert list(saver.list(cfg)) == []


async def test_async_writes_return_config_without_retaining_state():
    """Use the same no-persistence contract through the asynchronous methods."""
    saver = NoOpSaver()
    cfg = {"configurable": {"thread_id": "t1", "user_id": "u1"}, "tags": ["request"]}

    result = await saver.aput(cfg, empty_checkpoint(), {}, {})
    await saver.aput_writes(result, [("channel", "value")], "task-1")

    uuid.UUID(result["configurable"]["checkpoint_id"])
    assert result["configurable"]["checkpoint_ns"] == ""
    assert result["configurable"]["thread_id"] == "t1"
    assert result["configurable"]["user_id"] == "u1"
    assert result["tags"] == ["request"]
    assert "checkpoint_id" not in cfg["configurable"]
    assert await saver.aget(result) is None
    assert await saver.aget_tuple(result) is None
    assert [c async for c in saver.alist(cfg)] == []
