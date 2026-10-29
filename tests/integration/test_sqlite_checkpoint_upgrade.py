"""Resume a database produced by the previous released SQLite saver."""

import sqlite3
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from langgraph_agent_toolkit.core.memory.coordinated_sqlite import CoordinatedSqliteSaver


async def test_sqlite_3_1_1_pending_writes_survive_schema_upgrade_and_resume(tmp_path):
    """Keep completed work when setup adds the task path column."""
    path = tmp_path / "legacy.sqlite"
    fixture = Path(__file__).parents[1] / "fixtures" / "checkpoint_sqlite_3_1_1.sql"
    with sqlite3.connect(path) as conn:
        conn.executescript(fixture.read_text())
        assert "task_path" not in {row[1] for row in conn.execute("PRAGMA table_info(writes)")}

    replayed = []

    def completed(state):
        replayed.append("completed")
        return {"messages": [AIMessage("Completed before pause.", id="completed")]}

    def paused(state):
        answer = interrupt("Choose the next action.")
        return {"messages": [AIMessage(answer["message"], id="resumed")]}

    builder = StateGraph(MessagesState)
    builder.add_node("a_completed", completed)
    builder.add_node("b_paused", paused)
    for name in ("a_completed", "b_paused"):
        builder.add_edge(START, name)
        builder.add_edge(name, END)
    config = {"configurable": {"thread_id": "legacy-sqlite-upgrade"}}

    async with CoordinatedSqliteSaver.from_conn_string(str(path)) as saver:
        graph = builder.compile(checkpointer=saver)
        before = await graph.aget_state(config)
        assert before.next == ("b_paused",)
        assert len(before.interrupts) == 1
        assert [message.content for message in before.values["messages"]] == ["Begin.", "Completed before pause."]
        resumed = await graph.ainvoke(Command(resume={"message": "Continued."}), config, durability="sync")
        assert [message.content for message in resumed["messages"]] == [
            "Begin.",
            "Completed before pause.",
            "Continued.",
        ]
        assert replayed == []
        assert not (await graph.aget_state(config)).interrupts

    async with CoordinatedSqliteSaver.from_conn_string(str(path)) as saver:
        restored = await builder.compile(checkpointer=saver).aget_state(config)
        assert [message.content for message in restored.values["messages"]] == [
            "Begin.",
            "Completed before pause.",
            "Continued.",
        ]
        assert replayed == []
        assert not restored.interrupts
