"""Use real Deep Agents file tools with a deterministic model over HTTP."""

from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph


class FileModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "deepagents-process-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        instruction = next(message.content for message in reversed(messages) if isinstance(message, HumanMessage))
        if isinstance(messages[-1], ToolMessage):
            reply = AIMessage(content="Saved." if instruction.startswith("write:") else messages[-1].content)
        else:
            arguments = {"file_path": "/note.txt"}
            name = "read_file"
            if instruction.startswith("write:"):
                name = "write_file"
                arguments["content"] = instruction.removeprefix("write:")
            reply = AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": uuid4().hex}])
        return ChatResult(generations=[ChatGeneration(message=reply)])


deep_agent = Agent("deep-agent", "A deterministic Deep Agent for process tests.", build_graph(model=FileModel()))
