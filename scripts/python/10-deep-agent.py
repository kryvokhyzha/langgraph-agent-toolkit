"""Run a Deep Agent with SQLite and virtual files. The default needs no API key."""

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import fire
import rootutils
from dotenv import find_dotenv, load_dotenv
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langsmith import tracing_context


rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True, dotenv=False)


class DemoModel(FakeMessagesListChatModel):
    """Produce fixed tool calls so the offline example exercises the real graph."""

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name: str, arguments: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": uuid4().hex}])


async def run(database: str, thread_id: str, user_id: str, live: bool, message: str) -> None:
    if live:
        load_dotenv(find_dotenv(".local.env"), override=True)
        os.environ["USE_FAKE_MODEL"] = "false"
    else:
        os.environ["USE_FAKE_MODEL"] = "true"

    from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph
    from langgraph_agent_toolkit.service.auth import storage_thread_id

    path = f"/reports/delivery-{uuid4().hex[:8]}.md"
    report = "# Delivery estimate\n\nSix batches of seven items require 42 items."
    tasks = [{"content": "Calculate and save the delivery estimate", "status": "in_progress"}]
    if live:
        graph = build_graph()
    else:
        graph = build_graph(
            model=DemoModel(
                responses=[
                    tool_call("write_todos", {"todos": tasks}),
                    tool_call("multiply", {"a": 6, "b": 7}),
                    tool_call("write_file", {"file_path": path, "content": report}),
                    tool_call("task", {"subagent_type": "reviewer", "description": "Check the 6 by 7 estimate."}),
                    tool_call("read_file", {"file_path": path}),
                    tool_call("write_todos", {"todos": [{**tasks[0], "status": "completed"}]}),
                    AIMessage(content=f"The reviewed estimate is 42 items. The virtual report is {path}."),
                ]
            ),
            subagents=[
                {
                    "name": "reviewer",
                    "description": "Check a delivery estimate.",
                    "system_prompt": "Check the arithmetic and return a short review.",
                    "tools": [],
                    "model": DemoModel(responses=[AIMessage(content="Reviewed: 6 times 7 equals 42.")]),
                }
            ],
        )
    if database != ":memory:":
        Path(database).expanduser().parent.mkdir(parents=True, exist_ok=True)
        database = str(Path(database).expanduser())
    config = {
        "configurable": {"thread_id": storage_thread_id(user_id, "deep-agent", thread_id), "user_id": user_id},
        "recursion_limit": 64,
    }
    # Disable external tracing for the offline demonstration, including child calls.
    with tracing_context(enabled=None if live else False):
        async with AsyncSqliteSaver.from_conn_string(database) as saver:
            await saver.setup()
            graph.checkpointer = saver
            async with asyncio.timeout(120):
                result = await graph.ainvoke({"messages": [HumanMessage(message)]}, config=config)
            saved = await graph.aget_state(config)
    print(result["messages"][-1].content)
    files = saved.values.get("files", {})
    print(f"Saved {len(files)} virtual file(s) for thread {thread_id!r} in {database}.")
    if not live:
        if path not in files or files[path]["content"] != report:
            raise RuntimeError("The offline demonstration did not save its report.")
        print(files[path]["content"])


def main(
    database: str = "data/deep-agent-demo.sqlite",
    thread_id: str = "deep-agent-demo",
    user_id: str = "demo-user",
    live: bool = False,
    message: str = (
        "Plan a delivery estimate for six batches of seven items. Save a report and ask a reviewer to check it."
    ),
) -> None:
    """Run offline by default. Set live=True to use the configured model."""
    asyncio.run(run(database, thread_id, user_id, live, message))


if __name__ == "__main__":
    fire.Fire(main)
