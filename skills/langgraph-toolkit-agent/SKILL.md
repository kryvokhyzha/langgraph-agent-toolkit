---
name: langgraph-toolkit-agent
description:
  Build or modify application agents served by langgraph-agent-toolkit. Use for
  Agent registration, graph factories, native LangChain agents, and optional
  runtime MCP or Deep Agents integration.
---

# Build a toolkit agent

Read the application's project instructions and dependency lockfile first. Check
its installed toolkit version before using an API from a newer release. Keep
application agents in the application repository unless the task changes the
toolkit itself. Preserve the existing model, authentication, and storage choices
unless the task requires a change.

## Choose the graph

- Use a fixed `StateGraph` when the application needs explicit control flow.
- Use native LangChain `create_agent` for a new tool-calling agent. The
  toolkit's `create_agent` blueprint shows its model factory and middleware
  integration.
- Retain an existing custom `create_react_agent` integration unless migration is
  part of the task. The native builder and custom builder have different hooks.
- Use [extensions](references/extensions.md) only when the task needs MCP, Deep
  Agents, or tool selection. These are optional application features.

## Register an importable agent

The service accepts an `Agent` or a compiled graph through `AGENT_PATHS`. An
`Agent` gives the public agent ID an explicit name. Keep that name stable for
existing conversations. Create an empty `my_app/__init__.py` and this
`my_app/agent.py` for a deterministic starting point:

```python
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph_agent_toolkit.agents.agent import Agent


def reply(state: MessagesState):
    return {"messages": [AIMessage(content="The support agent is running.")]}


builder = StateGraph(MessagesState)
builder.add_node("reply", reply)
builder.add_edge(START, "reply")
builder.add_edge("reply", END)

agent = Agent(
    name="support-agent",
    description="A deterministic support agent.",
    graph=builder.compile(),
)
```

Set these values in the application's service configuration:

```ini
AGENT_PATHS=["my_app.agent:agent"]
DEFAULT_AGENT=support-agent
```

Install or copy `my_app` into the service image. The toolkit's standard API
image does not include arbitrary application modules. This registration exposes
`/support-agent/invoke` and the corresponding stream and history routes.

## Keep resource ownership explicit

- Keep network requests out of module imports. Use async tool functions for
  asynchronous I/O. Bound external operations and preserve cancellation.
- Supply `Agent.graph_factory` when the service must rebuild models or tools
  during each lifespan. It receives the configured MCP tools as one positional
  list. Return a fresh compiled graph and retain those supplied tools.
- Leave `checkpointer=None` when the service should assign its configured saver.
  The service preserves an explicit `NoOpSaver`, `False`, or custom saver. No
  configured memory backend means process-local checkpoint memory.
- Keep injected clients and stores open while the graph runs. The application
  owns their shutdown. Direct graph calls do not inherit HTTP authentication,
  admission limits, or the service's conversation locks.
- Use `user_id` for an explicit long-term memory namespace. Use `thread_id` for
  one conversation. Passing `user_id` alone does not create a memory store.

## Verify the changed behavior

First run a deterministic graph or fake model with no external tool actions.
Check that `/info` lists the agent and that a request reaches its named route.
For state changes, send two turns to one thread and read the saved history.
Check another thread separately. Use a disposable persistent database when
restart behavior is part of the task. Do not describe an in-memory graph test as
proof of database persistence or live model compatibility.

Use the matching local sources when a toolkit checkout is available:
`agents/agent.py`, `agents/blueprints/create_agent/`, and `service/handler.py`
inside `langgraph_agent_toolkit/`. Otherwise inspect the installed package. The
[application guide](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/usage.rst)
provides the full service setup. Match those instructions to the installed
release.
