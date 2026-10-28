"""Run real tool selection with scripted models and local tools."""

import json
import os
import re
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.core.tool_selection import ToolSelectionConfig


_EVENTS = Path(os.environ["LAT_TEST_TOOL_SELECTION_EVENTS"])
_WRITE_LOCK = Lock()
_BUILDER = os.environ["LAT_TEST_TOOL_SELECTION_BUILDER"]


def record_event(**values: Any) -> None:
    """Append one event without interleaving concurrent model and tool calls."""
    with _WRITE_LOCK, _EVENTS.open("a") as output:
        output.write(json.dumps(values, sort_keys=True) + "\n")


def current_turn(messages):
    index = max(index for index, message in enumerate(messages) if isinstance(message, HumanMessage))
    instruction = str(messages[index].content)
    match = re.search(r"case=([\w-]+)", instruction)
    case = match.group(1) if match else "specialist"
    results = [message for message in messages[index + 1 :] if isinstance(message, ToolMessage)]
    return instruction, case, results


class BenchmarkModel(BaseChatModel):
    """Use the SDK's structured-output parser and execute a fixed workflow."""

    agent_id: str
    role: str = "main"
    builder: str = "native"
    bound_schemas: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "tool-selection-process-test"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"bound_schemas": [convert_to_openai_tool(tool) for tool in tools]})

    def get_num_tokens(self, text):
        return len(text.split())

    def get_num_tokens_from_messages(self, messages, tools=None):
        return sum(self.get_num_tokens(str(message.content)) for message in messages)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        instruction, case, results = current_turn(messages)
        names = [schema["function"]["name"] for schema in self.bound_schemas]
        record_event(
            kind=self.role,
            agent_id=self.agent_id,
            case=case,
            tools=names,
            schema_bytes=len(json.dumps(self.bound_schemas, sort_keys=True).encode("utf-8")),
        )
        if self.role == "selector":
            if "bad-selector" in instruction:
                reply = AIMessage(content="This response has no structured tool selection.")
            else:
                selected = [] if instruction.startswith("read-note") else self.workflow_tools(instruction)
                reply = self.call(names[0], tools=selected)
        elif self.role == "specialist":
            reply = AIMessage(content="Checked.") if results else self.call("verify_result", value=case)
        elif instruction.startswith("read-note"):
            reply = (
                AIMessage(content=results[-1].content) if results else self.call("read_file", file_path="/result.txt")
            )
        else:
            reply = self.workflow_reply(instruction, case, results)
        return ChatResult(generations=[ChatGeneration(message=reply)])

    @staticmethod
    def workflow_tools(instruction):
        suffix = "secondary" if "secondary" in instruction else "primary"
        return [f"lookup_{suffix}", f"record_{suffix}"]

    def workflow_reply(self, instruction, case, results):
        lookup, save = self.workflow_tools(instruction)
        value = f"verified-{case}"
        steps = [(lookup, {"resource_id": case}), (save, {"value": value})]
        if self.builder == "deepagents":
            steps = [
                ("write_todos", {"todos": [{"content": "Verify and save the resource.", "status": "in_progress"}]}),
                steps[0],
                ("task", {"subagent_type": "verifier", "description": f"Verify the resource. case={case}"}),
                steps[1],
                ("write_file", {"file_path": "/result.txt", "content": value}),
                ("read_file", {"file_path": "/result.txt"}),
            ]
        if len(results) >= len(steps):
            return AIMessage(content=f"Complete: {value}")
        name, arguments = steps[len(results)]
        if name.startswith("record_"):
            lookup_result = next(message for message in results if message.name == lookup)
            arguments = {"value": json.loads(lookup_result.content)["value"]}
        return self.call(name, **arguments)

    def call(self, name, **arguments):
        available = {schema["function"]["name"] for schema in self.bound_schemas}
        if name not in available:
            raise ValueError(f"The model needs a tool that selection removed: {name}")
        return AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": uuid4().hex}])


def workflow_catalog(agent_id, distractors):
    def lookup(resource_id: str) -> str:
        """Read a resource and return the verified value for the next step."""
        record_event(kind="tool", agent_id=agent_id, case=resource_id, name="lookup")
        return json.dumps({"value": f"verified-{resource_id}"})

    def save(value: str) -> str:
        """Record the verified value exactly once."""
        record_event(kind="effect", agent_id=agent_id, case=value.removeprefix("verified-"), value=value)
        return "Saved."

    def unused(value: str) -> str:
        """Return a value from an unrelated test catalog."""
        raise AssertionError("The workflow called an unrelated tool")

    tools = []
    for suffix in ("primary", "secondary"):
        tools.extend(
            [
                StructuredTool.from_function(lookup, name=f"lookup_{suffix}"),
                StructuredTool.from_function(save, name=f"record_{suffix}"),
            ]
        )
    for index in range(distractors):
        tools.append(
            StructuredTool.from_function(
                unused,
                name=f"unrelated_{index:03d}",
                description=(
                    "Read an unrelated catalog entry. This tool cannot read or record workflow resources. "
                    "Use it only for the separate catalog feature."
                ),
            )
        )
    return tools


def build_agent(mode, distractors, *, parsing_failure="error"):
    name = f"selection-{mode}-{distractors}"
    if parsing_failure != "error":
        name += f"-{parsing_failure}"
    model = BenchmarkModel(agent_id=name, builder=_BUILDER)
    selector = BenchmarkModel(agent_id=name, role="selector")
    config = ToolSelectionConfig(
        mode="off" if mode == "off" else "llm",
        max_tools=2,
        always_include=["add"],
        max_retries=1,
        on_parsing_failure=parsing_failure,
    )
    arguments = dict(tool_selection=config, selector_model=selector)
    if _BUILDER == "native":
        from langgraph_agent_toolkit.agents.blueprints.create_agent._shared import build_tool_graph

        graph = build_tool_graph(model, workflow_catalog(name, distractors), **arguments)
    else:
        from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph

        def verify_result(value: str) -> str:
            """Verify a resource in the delegated workflow."""
            record_event(kind="specialist_tool", agent_id=name, case=value, name="verify_result")
            return "Checked."

        graph = build_graph(
            workflow_catalog(name, distractors),
            model=model,
            subagents=[
                {
                    "name": "verifier",
                    "description": "Verify a resource.",
                    "model": BenchmarkModel(agent_id=name, role="specialist"),
                    "tools": [StructuredTool.from_function(verify_result)],
                }
            ],
            **arguments,
        )
    return Agent(name, "A tool selection API benchmark agent.", graph)


for _size in (8, 32, 128):
    for _mode in ("off", "llm"):
        globals()[f"agent_{_mode}_{_size}"] = build_agent(_mode, _size)
agent_fallback = build_agent("llm", 32, parsing_failure="all")
