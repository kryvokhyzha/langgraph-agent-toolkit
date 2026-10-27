"""Verify native PostgreSQL checkpoint pruning through the toolkit saver."""

import asyncio
from typing import Annotated, TypedDict
from uuid import uuid4

import pytest
import test_postgres_reliability as postgres_tests
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.channels import DeltaChannel
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt


pytestmark = [
    pytest.mark.postgres,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not postgres_tests.TEST_DSN, reason="Set LAT_TEST_POSTGRES_DSN to a disposable PostgreSQL database"
    ),
]
postgres_backend = postgres_tests.backend


def message_graph(saver):
    def reply(state):
        text = state["messages"][-1].content
        if text == "pause":
            text = interrupt("Choose the next action.")
        return {"messages": [AIMessage("Reply: " + text)]}

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile(checkpointer=saver)


async def test_default_prune_keeps_state_and_pending_interrupt_after_saver_restart(postgres_backend):
    thread = uuid4().hex
    config = {"configurable": {"thread_id": thread}}
    other = {"configurable": {"thread_id": uuid4().hex}}
    async with asyncio.timeout(20):
        async with postgres_backend.get_checkpoint_saver() as saver:
            await saver.setup()
            graph = message_graph(saver)
            await graph.ainvoke({"messages": [HumanMessage("first")]}, config, durability="sync")
            await graph.ainvoke({"messages": [HumanMessage("pause")]}, config, durability="sync")
            await graph.ainvoke({"messages": [HumanMessage("private")]}, other, durability="sync")
            before = await graph.aget_state(config)
            assert before.interrupts
            assert len([item async for item in saver.alist(config)]) > 1

            await saver.aprune([thread])
            after = await graph.aget_state(config)
            assert after.values == before.values
            assert after.interrupts == before.interrupts
            assert after.config == before.config
            assert len([item async for item in saver.alist(config)]) == 1
            assert (await graph.aget_state(other)).values["messages"][-1].content == "Reply: private"

        async with postgres_backend.get_checkpoint_saver() as saver:
            await saver.setup()
            graph = message_graph(saver)
            resumed = await graph.ainvoke(Command(resume="continued"), config, durability="sync")
            assert [message.content for message in resumed["messages"]] == [
                "first",
                "Reply: first",
                "pause",
                "Reply: continued",
            ]
            assert not (await graph.aget_state(config)).interrupts


def merge_delta_values(current, writes):
    return current + [item for write in writes for item in write]


class DeltaState(TypedDict):
    values: Annotated[list[str], DeltaChannel(merge_delta_values)]


async def test_prune_refuses_delta_history_before_deleting_any_requested_thread(postgres_backend):
    config = {"configurable": {"thread_id": uuid4().hex}}
    other = {"configurable": {"thread_id": uuid4().hex}}
    builder = StateGraph(DeltaState)
    builder.add_node("append", lambda state: {"values": ["stored"]})
    builder.add_edge(START, "append")
    builder.add_edge("append", END)

    async with asyncio.timeout(20):
        async with postgres_backend.get_checkpoint_saver() as saver:
            await saver.setup()
            graph = builder.compile(checkpointer=saver)
            regular = message_graph(saver)
            await graph.ainvoke({"values": ["input"]}, config, durability="sync")
            await regular.ainvoke({"messages": [HumanMessage("private")]}, other, durability="sync")
            before = await graph.aget_state(config)
            counts = [len([item async for item in saver.alist(scope)]) for scope in (other, config)]

            with pytest.raises(NotImplementedError, match="DeltaChannel"):
                await saver.aprune([other["configurable"]["thread_id"], config["configurable"]["thread_id"]])

            assert [len([item async for item in saver.alist(scope)]) for scope in (other, config)] == counts
            assert (await graph.aget_state(config)).values == before.values == {"values": ["input", "stored"]}

            await saver.aprune([config["configurable"]["thread_id"]], strategy="delete")
            assert await saver.aget_tuple(config) is None
            assert (await graph.aget_state(config)).values == {}
            assert (await regular.aget_state(other)).values["messages"][-1].content == "Reply: private"
