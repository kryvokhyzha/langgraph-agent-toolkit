---
name: langgraph-toolkit-client
description:
  Integrate applications with the langgraph-agent-toolkit REST API or
  AgentClient. Use for authentication, conversation identity, sync or async
  calls, streaming, history pagination, thread lists, feedback, and interrupt
  recovery. Use the project's existing service configuration.
---

# Integrate a toolkit client

Check the installed toolkit version and the target service version. Use the
matching API schema or source when a method differs from this guidance. In a
toolkit checkout, start with `langgraph_agent_toolkit/client/client.py`,
`langgraph_agent_toolkit/schema/schema.py`, and
`langgraph_agent_toolkit/service/auth.py`. In another project, use the
[client source](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/langgraph_agent_toolkit/client/client.py)
and
[usage guide](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/usage.rst).
Do not assume that the main branch matches an older installed release.

## Keep conversation identity consistent

- Send the service credential as `Authorization: Bearer <token>`.
  `AgentClient(auth_secret=...)` sets this header. The client can also read
  `AUTH_SECRET` from the environment. It does not load `.env`.
- Keep the deployment's authentication mode. The default `AUTH_MODE=trusted`
  permits one shared service token. `user_id` is optional. Omitting it uses the
  token's owner; shared-token callers then use the same service owner. A trusted
  backend can supply its authenticated user's ID. User-bound tokens cannot
  select a different owner. Do not require per-user tokens for shared-token
  access.
- Use `user_id` for a stable user across conversations. Long-term memory still
  requires agent store logic. Use `thread_id` for one conversation's short-term
  checkpoint state. Keep the agent, user, and thread together on later calls.
- Use public thread IDs from requests or responses. Do not calculate storage
  hashes in the client. Create a thread ID before a new stream when the
  application must recover it after a lost response.
- Put identity fields at the request's top level. Do not override them through
  `agent_config`. Start a new thread when the user starts a new conversation.

## Select the correct client lifecycle

Keep one client open for repeated calls. Use `with` for synchronous calls and
`async with` for asynchronous calls. An async client must stay in one event
loop. The caller must close injected `http_client` and `async_http_client`
objects. The client closes the HTTP clients that it creates.

For async applications, set `get_info=False` and select an explicit agent. The
default constructor otherwise makes a synchronous `/info` request.

```python
from langgraph_agent_toolkit.client import AgentClient


async def main(api_url, agent_id, token, thread_id, user_id=None):
    async with AgentClient(
        base_url=api_url,
        agent=agent_id,
        auth_secret=token,
        get_info=False,
    ) as client:
        return await client.ainvoke(
            {"message": "Hello."}, thread_id=thread_id, user_id=user_id
        )
```

This example owns a client for one operation. In a service, create that client
in the application lifespan and pass it to request handlers.

## Consume streams and errors

Use `stream()` or `astream()` for SSE. Use `stream_jsonl()` or `astream_jsonl()`
for JSON Lines. SDK iterators yield text tokens as `str` and complete messages
as `ChatMessage`. A complete message can be a tool or custom message. Do not
append both tokens and complete message text to the answer.

Handle `AgentClientError` around the full iteration. HTTP 200 and partial text
do not prove success. SSE requires `[DONE]`; the SDK rejects a missing marker.
JSON Lines has no completion marker. Establish completion from the application
response contract. Close an iterator if reading stops early. Use
`contextlib.closing` or `contextlib.aclosing` for this purpose.

## Read all requested pages

| Operation                            | Default page      | Continue with             | End condition         |
| ------------------------------------ | ----------------- | ------------------------- | --------------------- |
| `get_history()` / `aget_history()`   | 100 messages      | `offset=page.next_offset` | `next_offset is None` |
| `list_threads()` / `alist_threads()` | 100 conversations | `cursor=page.next_cursor` | `next_cursor is None` |

History requires one `thread_id`. Its requested limit is at most 1000, subject
to the server's smaller `HISTORY_MAX_PAGE_SIZE`. Messages are oldest first.
Thread lists contain identifiers, not messages. Their maximum limit is 100. They
use creation order, newest first. Keep the same user and agent between pages. An
empty thread page can still have a next cursor. Both SDK methods return one
page; neither loads later pages automatically.

## Check the integration

Test the selected protocol through the application boundary. Include a later
history page, owner isolation, and a stream failure after partial text when
those paths are used. In a toolkit checkout, use the relevant tests under
`tests/client/`, `tests/service/test_threads.py`, and
`tests/e2e/test_api_journeys.py`. Process tests require `--run-e2e`.

Read [recovery and advanced requests](references/recovery.md) when the task
includes retries, feedback, interrupts, transport settings, or missing threads.
