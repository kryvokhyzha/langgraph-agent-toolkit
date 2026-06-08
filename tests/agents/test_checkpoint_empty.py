import uuid

from langgraph_agent_toolkit.agents.components.checkpoint.empty import NoOpSaver


def test_get_and_list_are_empty():
    saver = NoOpSaver()
    cfg = {"configurable": {"thread_id": "t1"}}
    assert saver.get(cfg) is None
    assert saver.get_tuple(cfg) is None
    assert list(saver.list(cfg)) == []


def test_put_returns_fake_checkpoint_id_and_preserves_configurable():
    saver = NoOpSaver()
    cfg = {"configurable": {"thread_id": "t1"}}

    result = saver.put(cfg, checkpoint={}, metadata={}, new_versions={})

    # A valid uuid checkpoint_id is returned, the namespace is set, and existing keys survive.
    uuid.UUID(result["configurable"]["checkpoint_id"])
    assert result["configurable"]["checkpoint_ns"] == ""
    assert result["configurable"]["thread_id"] == "t1"


def test_put_writes_is_noop():
    saver = NoOpSaver()
    assert saver.put_writes({"configurable": {}}, [("channel", "value")], "task-1") is None


async def test_async_get_and_list_are_empty():
    saver = NoOpSaver()
    cfg = {"configurable": {"thread_id": "t1"}}
    assert await saver.aget(cfg) is None
    assert await saver.aget_tuple(cfg) is None
    assert [c async for c in saver.alist(cfg)] == []


async def test_aput_mirrors_put_and_aput_writes_is_noop():
    saver = NoOpSaver()
    cfg = {"configurable": {"thread_id": "t1"}}

    result = await saver.aput(cfg, {}, {}, {})
    uuid.UUID(result["configurable"]["checkpoint_id"])
    assert result["configurable"]["thread_id"] == "t1"

    assert await saver.aput_writes(cfg, [("channel", "value")], "task-1") is None
