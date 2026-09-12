"""Run at most six real OpenAI calls in one explicitly enabled test process."""

import asyncio
import json
import os
from hmac import compare_digest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel
from starlette.responses import JSONResponse

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory


CALL_LIMIT = 6
OPENAI_URL = "https://api.openai.com/v1"


class CallState(BaseModel):
    attempted: int = 0
    active: int = 0
    peak_active: int = 0
    completed: int = 0
    cancelled: int = 0
    failed: int = 0
    tools_completed: int = 0


calls = CallState()


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return their sum."""
    return a + b


async def call_model(messages, config: RunnableConfig, *, streaming: bool, use_tools: bool = False) -> AIMessage:
    """Check the call budget before constructing or calling the provider model."""
    if os.environ.get("LAT_TEST_LLM_CHILD") != "yes":
        raise RuntimeError("The live model test process is not enabled")
    if calls.attempted >= CALL_LIMIT:
        raise RuntimeError("The live model test call budget is exhausted")
    calls.attempted += 1
    calls.active += 1
    calls.peak_active = max(calls.peak_active, calls.active)
    try:
        parameters = json.loads(os.environ.get("LAT_TEST_OPENAI_MODEL_KWARGS", "{}"))
        if not isinstance(parameters, dict) or set(parameters) - {"reasoning_effort", "temperature"}:
            raise ValueError("Live model kwargs can contain only reasoning_effort and temperature")
        model = CompletionModelFactory.create(
            "openai",
            os.environ["LAT_TEST_OPENAI_MODEL"],
            configurable_fields=(),
            config_prefix="",
            model_parameter_values=(),
            **parameters,
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=OPENAI_URL,
            max_completion_tokens=256,
            max_retries=0,
            n=1,
            store=False,
            timeout=60,
            stream_usage=True,
            streaming=streaming,
            disable_streaming=not streaming,
            use_responses_api=False,
        )
        if use_tools:
            model = model.bind_tools([add], tool_choice="add", parallel_tool_calls=False)
        response = await model.ainvoke(messages, config=config)
        if not isinstance(response, AIMessage):
            raise TypeError("The provider did not return an AIMessage")
        calls.completed += 1
        return response
    except asyncio.CancelledError:
        calls.cancelled += 1
        raise
    except Exception:
        calls.failed += 1
        raise
    finally:
        calls.active -= 1


def build_graph(extra_tools=()):
    """Keep provider messages and real local tool results in checkpoint state."""
    if extra_tools:
        raise ValueError("The live test does not accept external tools")

    def route(state: MessagesState):
        human = next(message for message in reversed(state["messages"]) if isinstance(message, HumanMessage))
        return "tool_model" if human.content.startswith("live-tools:") else "reply"

    async def reply(state: MessagesState, config: RunnableConfig):
        response = await call_model(state["messages"], config, streaming=True)
        return {"messages": [response]}

    async def tool_model(state: MessagesState, config: RunnableConfig):
        instructions = SystemMessage("Call add exactly once with a=2 and b=5. Do not call another tool.")
        response = await call_model([instructions, *state["messages"]], config, streaming=False, use_tools=True)
        return {"messages": [response]}

    async def run_tool(state: MessagesState):
        requested = state["messages"][-1].tool_calls
        if len(requested) != 1 or requested[0]["name"] != "add":
            raise ValueError("The model must request exactly one add call")
        result = await add.ainvoke(requested[0])
        calls.tools_completed += 1
        return {"messages": [result]}

    async def final_reply(state: MessagesState, config: RunnableConfig):
        instructions = SystemMessage("State the tool result in one short sentence.")
        response = await call_model([instructions, *state["messages"]], config, streaming=False)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_conditional_edges(START, route, ["reply", "tool_model"])
    graph.add_node("reply", reply)
    graph.add_node("tool_model", tool_model)
    graph.add_node("run_tool", run_tool)
    graph.add_node("final_reply", final_reply)
    graph.add_edge("reply", END)
    graph.add_edge("tool_model", "run_tool")
    graph.add_edge("run_tool", "final_reply")
    graph.add_edge("final_reply", END)
    return graph.compile()


live_agent = Agent("live-agent", "A bounded live provider test agent.", build_graph(), graph_factory=build_graph)


class MetricsApp:
    """Expose safe counters without taking a request admission slot."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/live/metrics":
            authorization = dict(scope.get("headers", [])).get(b"authorization", b"")
            expected = f"Bearer {os.environ['AUTH_SECRET']}".encode()
            if (
                scope["method"] != "GET"
                or scope.get("client", ("",))[0] not in {"127.0.0.1", "::1"}
                or not compare_digest(authorization, expected)
            ):
                await JSONResponse({"detail": "Access denied"}, status_code=403)(scope, receive, send)
                return
            state = self.app.state
            admission = getattr(state, "request_admission", None)
            manager = getattr(state, "llm_transport_manager", None)
            values = {
                **calls.model_dump(),
                "call_limit": CALL_LIMIT,
                "admission_active": admission.active if admission is not None else 0,
                "admission_waiting": admission.waiting if admission is not None else 0,
                "stalled_cleanups": getattr(state, "stalled_request_cleanups", 0),
                "transport_pools": len(manager._pools) if manager is not None else 0,
            }
            await JSONResponse(values)(scope, receive, send)
            return
        await self.app(scope, receive, send)


def create_app():
    """Add test counters to the real service factory."""
    from langgraph_agent_toolkit.service.handler import create_app as create_service

    return MetricsApp(create_service())
