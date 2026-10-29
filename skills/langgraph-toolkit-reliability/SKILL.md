---
name: langgraph-toolkit-reliability
description:
  Diagnose persistence, connection failures, concurrent requests, slow runs, and
  worker recovery in applications that use langgraph-agent-toolkit. Use when a
  task concerns checkpoint loss, database locks, request limits, model
  transports, or production recovery.
---

# Diagnose toolkit reliability

Identify the failing operation and its persistence boundary before changing
limits. Use the installed toolkit version and effective deployment settings. Do
not infer production settings from `.env.example`.

## Find the boundary

Trace one affected request through the API, executor, graph, and checkpointer.
Keep public `thread_id`, authenticated owner, and agent ID consistent. Redact
credentials from evidence.

Distinguish these cases:

| Symptom                                   | First check                                                       |
| ----------------------------------------- | ----------------------------------------------------------------- |
| History differs between workers           | Saver type and shared database configuration                      |
| Progress disappears after restart         | Durable storage, mounted files, and checkpoint durability         |
| One conversation waits                    | Conversation queue and the active run for that storage key        |
| All conversations wait                    | Request admission, pool checkout, event loop, and provider limits |
| Tokens arrive, then history is incomplete | Stream errors, completion, and checkpoint write failures          |
| `/threads` omits readable history         | Creation checkpoint and public identity metadata                  |

Read [diagnostics.md](references/diagnostics.md) for the relevant storage,
connection, or worker procedure. Do not change every timeout together.

## Preserve persistence semantics

- `thread_id` selects short-term conversation state. A store can use `user_id`
  for long-term memory. Passing `user_id` does not create a store.
- The HTTP service scopes checkpoint keys by owner, agent, and public thread ID.
  Use the service identity helpers when changing service code.
- A graph with no explicit saver receives the configured service saver during
  startup. An explicit saver remains explicit. `MemorySaver` is local to one
  process. `NoOpSaver` is a valid stateless choice.
- `CHECKPOINT_DURABILITY=sync` waits for each checkpoint before the next graph
  super-step. The executor still uses asynchronous graph and database methods.
  Do not switch it to `async` as a fix for event-loop blocking.
- Checkpointing does not make external tool actions exactly once. A retry of an
  uncertain run can repeat an external write.

## Make a measured change

Use current settings declarations and the deployment configuration to establish
limits and units. Count all workers and replicas when sizing pools. Keep owned
resources inside their existing lifespan. Preserve cancellation and bounded
cleanup when changing connection code.

Prefer the dependency's connection and retry controls where they meet the
contract. Preserve toolkit coordination that prevents two workers from changing
one conversation at the same time. Test the failure before removing that code.

Verify the changed boundary with a real graph and a disposable database or local
transport. For recovery changes, include restart or connection loss. State which
checks used a real database and which used a scripted model.

## Source map

In a toolkit checkout, start with `core/memory/`, `core/models/transport.py`,
`agents/agent_executor.py`, and `service/handler.py` under
`langgraph_agent_toolkit/`. In an application, inspect the installed package or
the source for its pinned release.

- [Connections, durability, and limits](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/reliability.rst)
- [Deployment and worker supervision](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/deployment.rst)
- [Load test procedures](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/load_testing.rst)

The links show the current source branch. Check the installed version before
applying a newer option.
