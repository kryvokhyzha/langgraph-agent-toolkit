# Recovery and advanced requests

Use only the sections needed for the client task. For the current service
contract, see the
[usage guide](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/usage.rst),
[reliability guide](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/reliability.rst),
and
[MCP guide](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/mcp.rst).

## Failed or interrupted requests

`AgentClientError` exposes `status_code`, `error_code`, and `retry_after` when
the server provides them. `retry_after` can contain seconds or an HTTP date. An
error event after streaming starts does not provide a new HTTP status.

Do not replay a failed run automatically. Tools may have completed and
checkpoints may have been saved before the response failed. Read history with
the same owner, agent, and thread when recovery needs saved state. A history
read does not execute the graph. A shared thread ID is not an idempotency key.

Distinguish service authentication errors from provider configuration errors.
HTTP 401 or 403 can indicate an invalid service token or an owner mismatch.
`model_authentication_failed` indicates a server-side model credential error.
Changing the client bearer token does not fix that error. Use the returned error
code and the application retry policy for busy or unavailable services.

## Pending questions

Read `ChatMessage.custom_data.interrupts` after a run pauses. History returns
the same saved questions in `pending_interrupts`. Every history page includes
all pending questions. The message page size does not limit that list.

Match answers to the actual interrupt IDs. Do not use their list positions or
invent an ID. For typed interrupts, use the returned `response_schema` and send
the answer under `input.resume`:

```python
reply = await client.ainvoke(
    {"resume": {pending_id: answer}},
    thread_id=thread_id,
    user_id=user_id,
)
```

Keep the original agent, user, and thread. Typed answers can fail validation
while the question remains pending. Existing raw interrupt replies keep their
format. MCP forms use a `responses` envelope. Read the MCP guide before adding
MCP form or URL handling. Do not convert every interrupt to plain text.

## Feedback and message metadata

Retain the returned `run_id` and `feedback_token` with the run result. Use the
same agent and user for `create_feedback()` or `acreate_feedback()`. User-bound
tokens require `feedback_token=reply.feedback_token`. The service needs its own
`FEEDBACK_SIGNING_SECRET` for that flow. Trusted backend feedback keeps its
existing request shape. Do not create or infer a feedback token from history.

Preserve tool call IDs, tool status, and `tool_artifact` when importing or
displaying tool messages. A message with empty content and tool calls can be
valid. `usage_metadata=None` means token counts are unavailable. Counts on one
message are not necessarily the total for all model calls in the run.

## Timeouts and connection ownership

`timeout` applies to ordinary requests. `stream_timeout` controls stream
timeouts; its read limit measures idle time between reads. It is not a total run
deadline. Keep values consistent with the service and proxy timeouts.

The constructor's `verify` option validates agent information. It is not a TLS
verification option. Configure transport and TLS options on injected HTTPX
clients when required. Close those clients in the application that owns them.

## Missing conversations

`/threads` lists one owner and agent. It does not search all users or agents.
Use the same identity that created the conversation. Continue through empty
pages while `next_cursor` is present.

Toolkit-managed SQLite and PostgreSQL support listing. In-memory lists are local
to one process. `NoOpSaver` returns an empty list. An unsupported custom saver
returns HTTP 501 for listing.

Listing needs the initial root checkpoint with `step=0`. Pruning it can remove a
conversation from the list while its latest history remains readable by its
known ID. Older scoped checkpoints need matching public identity metadata. Do
not interpret a missing list entry as proof that all history was deleted.
