# Optional agent extensions

Read only the section for the requested feature. Check the installed package and
its lockfile before adding an extra or using a new builder parameter.

## MCP tools in the application

This configures tools for the running application agent. It does not configure
MCP servers in the coding agent's editor.

The toolkit uses `langchain.mcp.MCPAdapter` and FastMCP through its `mcp` extra.
Reuse `MCP_SERVERS` and `MCP_AGENT_SERVERS` when the service owns the
connection. Do not add the older `langchain-mcp-adapters` or the separate
`langchain-mcp` package to implement this path.

```ini
MCP_SERVERS={"documents":{"transport":"http","url":"https://documents.example.com/mcp","headers_env":{"Authorization":"DOCUMENTS_MCP_AUTH"},"tool_allowlist":["search"]}}
MCP_AGENT_SERVERS={"support-agent":["documents"]}
```

Use the real endpoint selected for the application. The secret manager supplies
`DOCUMENTS_MCP_AUTH`. Its value must include the complete header, including
`Bearer` when required. The API's `AUTH_SECRET` is not forwarded to MCP.

The selected agent must expose `graph_factory(extra_tools)`. Combine its local
tools with `extra_tools` through `langgraph_agent_toolkit.core.mcp.merge_tools`.
The factory returns the rebuilt graph. Do not ignore `extra_tools` or compile
one shared graph that retains another lifespan's connections.

The allowlist uses original server tool names. The final registered name in this
example is `documents_search`. Discovery occurs before readiness. A missing tool
or duplicate final name fails startup. Restart workers after tool schema
changes. Use a local protocol fixture to verify discovery and calls. Do not run
a real remote write merely to test registration.

MCP elicitation requires `CHECK_INTERRUPTS=true` and checkpoint storage.
Preserve interrupt IDs and structured answers through the API. Do not turn an
MCP failure into an automatic write retry: the remote write can already exist.

Source:
[MCP guide](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/mcp.rst)
and `langgraph_agent_toolkit/core/mcp.py` in the matching release.

## Deep Agents

Use the `deepagents` extra when planning, delegated tasks, or virtual files are
part of the requirement. The toolkit builder is
`langgraph_agent_toolkit.agents.blueprints.deep_agent.agent.build_graph`. Its
built-in registration name is `deep-agent`. Inspect the installed builder
signature before using its optional parameters.

- The default `StateBackend` keeps virtual files in the conversation state. It
  does not grant host filesystem access or create a long-term store.
- The application owns an injected `backend` or `store` and its lifecycle.
  Filesystem permissions constrain built-in file tools. They do not constrain
  arbitrary custom tools or compiled subagents.
- `skills` and `memory` paths refer to trusted files in that backend. This
  coding-agent skill collection is not installed into an application's runtime
  backend automatically.
- Keep `offload_binary_content` disabled unless external blob storage is part of
  the task. Offloading requires storage outside checkpoint state at the
  backend's blob path. Restore access to those blobs when restoring a
  checkpoint. Deleting checkpoint history does not delete external blobs.
- Keep the native planning and file tools available. Tool selection on the main
  agent does not configure every subagent's tool selection.

Use a deterministic model for graph, tool, and checkpoint tests. Use the real
provider only when the task requires it and its test configuration is available.
Source:
[Deep Agents guide](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/deepagents.rst).

## Tool selection

Tool selection is off by default. Reuse `ToolSelectionConfig` and the toolkit's
bridge to LangChain middleware when the task requires a smaller model tool
catalog. The worker setting is one JSON value:

```ini
TOOL_SELECTION={"mode":"llm","max_tools":8}
```

`llm` mode can add a selector-model request. `provider` mode requires a
supported model and API route. For OpenAI, the model must use the Responses API
with tool search support. A Chat Completions-only gateway is insufficient.

`always_include` uses final registered tool names, including MCP prefixes.
Required tools can exceed `max_tools`. Provider search does not guarantee a
smaller HTTP request payload. Keep selection off as the comparison baseline.
Test that necessary tools remain available and compare provider calls, latency,
and task completion before changing an application's default.

Source:
[tool selection guide](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/tool_selection.rst),
`core/tool_selection_config.py`, and `core/tool_selection.py` inside the
package.
