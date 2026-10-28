"""Use Phoenix prompts and real local tools without a model service."""

import os
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, MessagesState, StateGraph
from openinference.instrumentation import using_attributes

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.core.observability.phoenix import PhoenixObservability


class CalculatorModel(BaseChatModel):
    mode: str = "native"

    @property
    def _identifying_params(self):
        return {"model_name": "lat-test-calculator"}

    @property
    def _llm_type(self):
        return "phoenix-process-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def get_num_tokens(self, text):
        return len(text.split())

    def get_num_tokens_from_messages(self, messages, tools=None):
        return sum(self.get_num_tokens(str(message.content)) for message in messages)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1]
        if self.mode == "specialist":
            reply = (
                AIMessage("Verified result: 21.") if isinstance(last, ToolMessage) else self.call("multiply", a=7, b=3)
            )
        elif not isinstance(last, ToolMessage):
            reply = self.call("add", a=2, b=5)
        elif self.mode == "deepagents" and last.name == "add":
            reply = self.call("task", subagent_type="verifier", description="Verify seven multiplied by three.")
        else:
            reply = AIMessage("The checked result is 21." if self.mode == "deepagents" else "The sum is 7.")
        reply.usage_metadata = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
        return ChatResult(generations=[ChatGeneration(message=reply)])

    @staticmethod
    def call(name, **arguments):
        return AIMessage("", tool_calls=[{"name": name, "args": arguments, "id": uuid4().hex}])


def build_calculator():
    mode = os.environ["LAT_TEST_PHOENIX_BUILDER"]
    if mode == "native":
        from langgraph_agent_toolkit.agents.blueprints.create_agent._shared import build_tool_graph

        return build_tool_graph(CalculatorModel())
    from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph
    from langgraph_agent_toolkit.agents.components.tools import multiply

    return build_graph(
        model=CalculatorModel(mode="deepagents"),
        subagents=[
            {
                "name": "verifier",
                "description": "Verify arithmetic results.",
                "model": CalculatorModel(mode="specialist"),
                "tools": [multiply],
            }
        ],
    )


prompts = PhoenixObservability()
calculator = build_calculator()


async def reply(state, config):
    """Load the current remote prompt and record its native version ID."""
    prompt, remote = await prompts.apull_prompt(
        os.environ["LAT_TEST_PHOENIX_PROMPT_NAME"],
        return_with_prompt_object=True,
        template_format="f-string",
    )
    request = state["messages"][-1].content
    chain = prompt | RunnableLambda(lambda value: {"messages": value.to_messages()}) | calculator
    with using_attributes(prompt_template_version=remote.id, prompt_template_variables={"request": request}):
        result = await chain.ainvoke({"request": request}, config=config)
    return {"messages": result["messages"]}


builder = StateGraph(MessagesState)
builder.add_node("reply", reply)
builder.add_edge(START, "reply")
builder.add_edge("reply", END)
phoenix_agent = Agent(
    "lat-process-agent", "A Phoenix API process test agent.", builder.compile(), observability=prompts
)
