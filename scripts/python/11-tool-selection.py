"""Compare optional tool selection with fixed responses and real agent graphs."""

import asyncio
import json
import os
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import fire
import rootutils
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from langsmith import tracing_context
from pydantic import Field


rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True, dotenv=False)


class DemoModel(FakeMessagesListChatModel):
    """Return fixed actions and record the catalog visible to each model call."""

    visible_tools: list[list[str]] = Field(default_factory=list)

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "DemoModel":
        self.visible_tools.append([tool.name for tool in tools])
        return self

    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

    def get_num_tokens_from_messages(self, messages: list[BaseMessage], tools: Any = None) -> int:
        return sum(self.get_num_tokens(str(message.content)) for message in messages)


class DemoSelector(FakeMessagesListChatModel):
    """Return the tool required by the fixed arithmetic task."""

    calls: int = 0

    def _select(self, value: Any) -> dict[str, list[str]]:
        self.calls += 1
        return {"tools": ["multiply"]}

    async def _aselect(self, value: Any) -> dict[str, list[str]]:
        return self._select(value)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        return RunnableLambda(self._select, afunc=self._aselect)


def tool_call(name: str, **arguments: Any) -> AIMessage:
    """Build one model tool call without an external model request."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": uuid4().hex}])


def unrelated_tool(value: str) -> str:
    """Return a value for an unrelated catalog entry."""
    return value


async def run(builder: str, mode: str) -> None:
    """Run an isolated arithmetic task and print the observed model catalogs."""
    from langgraph_agent_toolkit.core.tool_selection import ToolSelectionConfig

    if builder not in ("native", "deep"):
        raise ValueError("builder must be native or deep")
    if mode not in ("off", "llm"):
        raise ValueError("This offline example supports only off and llm modes")
    config = ToolSelectionConfig(mode=mode, max_tools=1)
    selector = DemoSelector(responses=[])
    distractors = [
        StructuredTool.from_function(
            unrelated_tool,
            name=f"catalog_lookup_{index}",
            description=f"Look up reference entry {index}. This tool does not calculate products.",
        )
        for index in range(12)
    ]
    responses = [tool_call("multiply", a=6, b=7)]
    if builder == "deep":
        responses.append(tool_call("write_file", file_path="/reports/result.txt", content="The result is 42."))
    responses.append(AIMessage(content="The result is 42."))
    model = DemoModel(responses=responses)
    if builder == "native":
        from langgraph_agent_toolkit.agents.blueprints.create_agent._shared import build_tool_graph

        graph = build_tool_graph(model, distractors, tool_selection=config, selector_model=selector)
    else:
        from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph

        graph = build_graph(
            distractors,
            model=model,
            tool_selection=config,
            selector_model=selector,
        )
    with tracing_context(enabled=False):
        async with asyncio.timeout(30):
            result = await graph.ainvoke(
                {"messages": [HumanMessage("Calculate six multiplied by seven and report the result.")]},
                config={"recursion_limit": 32},
            )
    answer = result["messages"][-1].content
    if answer != "The result is 42.":
        raise RuntimeError("The graph did not return the expected arithmetic result")
    if mode == "llm":
        if selector.calls == 0:
            raise RuntimeError("The selector did not run")
        if any(name.startswith("catalog_lookup_") for names in model.visible_tools for name in names):
            raise RuntimeError("The model received an unrelated tool after selection")
        if any("multiply" not in names for names in model.visible_tools):
            raise RuntimeError("The selector removed the required arithmetic tool")
    elif selector.calls:
        raise RuntimeError("The disabled selector made a call")
    files = sorted(result.get("files", {}))
    if builder == "deep" and "/reports/result.txt" not in files:
        raise RuntimeError("The Deep Agent did not write its virtual result file")
    print(
        json.dumps(
            {
                "builder": builder,
                "mode": mode,
                "answer": answer,
                "selector_calls": selector.calls,
                "main_tool_counts": [len(names) for names in model.visible_tools],
                "main_visible_tools": model.visible_tools,
                "virtual_files": files,
                "measurement": "Fixed responses. These results do not measure provider speed or model quality.",
            },
            indent=2,
        )
    )


def main(builder: str = "native", mode: str = "off") -> None:
    """Run without dotenv files, provider credentials, or external tracing."""
    environment = {"USE_FAKE_MODEL": "true", "OBSERVABILITY_BACKEND": "empty"}
    with patch.dict(os.environ, environment, clear=True), patch("dotenv.find_dotenv", return_value=""):
        asyncio.run(run(builder, mode))


if __name__ == "__main__":
    fire.Fire(main)
