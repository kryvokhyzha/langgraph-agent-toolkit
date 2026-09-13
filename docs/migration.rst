Migrate authentication and checkpoint storage
=============================================

This release binds each stored conversation to a user and an agent. The public
``thread_id`` can stay the same. The database uses a new ``lat:v1:`` storage key.
Existing raw thread IDs need an explicit migration before the service can read
them through the new API.

Keep the two memory types separate
----------------------------------

``thread_id`` selects short-term memory: the messages and graph state of one
conversation. A checkpointer stores this state. ``user_id`` identifies the user
whose long-term memories can be shared across conversations in a separate store.
This follows the `LangGraph memory model
<https://docs.langchain.com/oss/python/concepts/memory>`_.

For example, requests with ``user_id="user-1"`` and different thread IDs have
separate conversation histories. An agent can share that user's saved preferences
through a store namespace such as ``("users", "user-1", "memories")``. Do not
include ``thread_id`` in that namespace when memories must span conversations.

The ``lat:v1:`` key scopes checkpoint access to the user, agent, and thread.
It does not merge the two memory types or change the user ID passed to the agent.
The agent still receives the stable ID in ``config["configurable"]["user_id"]``.
Custom agents retain their configured store.

History reads, additions, and clearing operate on short-term checkpoints only.
They require a ``thread_id``. Clearing one conversation does not delete the user's
long-term memories. The migration and retention commands below also affect only
checkpoint tables. They do not rename user IDs or change long-term store namespaces.

The service initializes its configured checkpointer. Long-term memory requires
an agent with a configured store and explicit logic to read and write memories.
Supplying ``user_id`` alone does not create a memory store. The PostgreSQL backend
exposes ``get_memory_store()`` for this use. The SQLite backend provides checkpoints
only. Keep a custom store and its connections open for the lifetime of its agents.

Choose an authentication mode
-----------------------------

``AUTH_MODE`` is an API worker environment setting. Do not add it to REST
request bodies or headers. The code default is ``trusted``. This preserves
0.9.2 shared bearer-token authentication. The example ``.env`` uses the same
mode. An existing deployment can keep only its shared ``AUTH_SECRET``.

For a separate deployment for each client, keep one shared bearer token. The
client's trusted application can supply user IDs for the users it serves::

   AUTH_MODE=trusted
   AUTH_SECRET=CHANGE_ME

Replace ``CHANGE_ME`` with your deployment secret. Set ``AUTH_SECRET`` in the
shell that runs curl. No per-user token list is required. ``user_id`` remains
optional. This request works before and after the upgrade::

   curl http://localhost:8080/chatbot-agent/invoke \
     -H "Authorization: Bearer ${AUTH_SECRET}" \
     -H 'Content-Type: application/json' \
     -d '{"input":{"message":"Hello"},"thread_id":"conversation-1"}'

To select a user's memory identity, the trusted application can still supply
``"user_id":"user-1"`` in the same JSON body.

The shared token authorizes that client application to supply ``user_id``. The
service scopes each stored conversation by user, agent, and public thread ID.
Every request for the same conversation must use the same user and agent IDs.
If ``user_id`` is omitted, the service uses ``AUTH_SERVICE_USER_ID`` (default:
``service``). Do not switch between omitted and explicit IDs for one conversation.
Use a separate database or PostgreSQL schema for each client deployment. Process
isolation alone does not isolate deployments that share checkpoint tables.

If the deployment explicitly sets ``AUTH_MODE=token``, remove that override or
set it to ``trusted`` on every worker. An explicit value takes precedence over
the default. The new storage keys still require the database migration below.
Keep the shared token inside the trusted client application. Possession of
this token grants access to all user identities in that deployment.

Optional mode for direct user access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If untrusted users connect directly to the API, use individual tokens instead::

   AUTH_MODE=token
   AUTH_USERS={"user-1":"replace-with-user-token"}

The server derives the user from that token. A supplied ``user_id`` must match.
``AUTH_SECRET`` in token mode identifies only ``AUTH_SERVICE_USER_ID``; it does
not authorize arbitrary user IDs. Individual ``AUTH_USERS`` tokens keep these
restrictions even when a deployment enables trusted mode.

Production startup requires configured credentials. Unauthenticated operation
is limited to development mode.

Authorize feedback from token users
-----------------------------------

Earlier feedback requests authenticated the caller but did not verify the
owner of ``run_id``. Token users now need a ``feedback_token`` from the
corresponding invoke, SSE, or JSON Lines response. The proof binds the user,
agent, and run. Submit feedback to that same agent.

Set ``FEEDBACK_SIGNING_SECRET`` to a separate random value with at least 32
characters. Store it only on the server. Do not use ``AUTH_SECRET`` or an
``AUTH_USERS`` bearer token: callers know those values and could forge proof.
Use the same signing secret across workers, replicas, and restarts.
Use a different signing secret for each client deployment.
Changing the signing secret invalidates previously issued feedback tokens.

This requirement applies to every non-trusted principal. It includes
``AUTH_SECRET`` in token mode and ``AUTH_USERS`` tokens in either mode.
The shared ``AUTH_SECRET`` with ``AUTH_MODE=trusted`` keeps its existing
feedback request shape and does not require a signing secret.

Before this change, a token user could send:

.. code-block:: json

   {"run_id":"returned-run-id","key":"quality","score":1.0}

After this change, the token user sends the proof returned with that run:

.. code-block:: json

   {"run_id":"returned-run-id","key":"quality","score":1.0,"feedback_token":"returned-feedback-token"}

With the Python client:

.. code-block:: python

   reply = client.invoke({"message": "Hello"}, thread_id="conversation-1")
   client.create_feedback(
       run_id=reply.run_id,
       key="quality",
       score=1.0,
       feedback_token=reply.feedback_token,
   )

The client does not cache proof. Pass it explicitly to ``create_feedback()``
or ``acreate_feedback()``. The included UI sends it from the generated message.
A missing server secret returns 503 for token-user feedback. Missing, altered,
or mismatched proof returns 403. Invocation and streaming still work when the
signing secret is unset.

Saved history and imported messages do not receive new feedback proof. Old
responses without proof cannot authorize token-user feedback. Keep the proof
with a response if feedback must be submitted later. A trusted backend can
continue its existing feedback flow. Feedback ``kwargs`` cannot override the
target run, user, trace, observation, or provider record ID. Comments and
ordinary metadata remain supported.

Update clients and process configuration
----------------------------------------

The endpoint paths and HTTP methods remain available. This does not mean that
all 0.9.2 behavior is unchanged. Trusted mode preserves the shared-token invoke,
stream, and ordinary feedback request shapes. It does not disable the history,
validation, error-handling, or storage changes below.

The following changes require a client review before deployment:

* Supply ``thread_id`` for history reads, message additions, and history clearing.
  A user ID alone no longer selects a conversation. Use the same agent ID that
  owns the conversation.
* Invocation can create a public thread ID when the request omits it. Save the
  ``thread_id`` from the response and send it with subsequent requests.
* History reads return a page of 100 messages by default. Use ``offset`` and
  ``limit`` to read later pages. The maximum limit is 1000, or the smaller value
  set by ``HISTORY_MAX_PAGE_SIZE``. Continue with ``next_offset`` until it is null.
* Clearing history deletes all stored checkpoint versions and pending writes
  for the selected conversation. It does not only empty the latest message list.
  Back up data that must remain available.
* SSE and JSON Lines error events raise ``AgentClientError`` in the client.
  Handle this exception separately from model text. Do not store an error event
  as an assistant reply.

The default ``AUTH_MODE=trusted`` keeps the existing trusted-backend request
shape. The included Streamlit UI is suitable for a single-user or private
deployment. It does not provide a multiuser login system.
Do not use one backend token as a substitute for user authentication in a public
Streamlit deployment.

ASGI workers have separate process state. Set authentication and other settings
for every worker before it imports the service. A Python setting changed in one
process does not update the other processes. The service factory validates
``custom_settings`` values before it starts a backend. Use the same configuration
for all workers.

``CHECKPOINT_DURABILITY`` defaults to ``sync``. Each checkpoint must finish
before the next graph step starts. Database calls still use async methods.
An explicit ``async`` override keeps overlapping saves and graph steps. Remove
that override or set it to ``sync`` to use the new default. Both modes report
write errors. See :doc:`reliability` for recovery limits and tuning guidance.

Check existing persistence
--------------------------

The migration can move only checkpoints that exist in the selected database.
Earlier built-in agents with ``MemorySaver`` kept state inside each worker even
when a service database was configured. Those checkpoints cannot be recovered
from the database after the worker stops. Export any required live history before
shutdown. Importing messages does not reconstruct pending tools or interrupted
subgraphs. Plan those runs separately before upgrading.

Prepare the migration
---------------------

1. Stop every service worker and background job that can write checkpoints.
   Do not run old and new workers against the database at the same time.
2. Back up the database. For SQLite, use the SQLite backup API or stop all
   connections before copying the database and any required WAL files. Verify
   that you can restore the backup.
3. Create a JSON manifest. Use the actual owner and agent for each conversation.
   The migration does not infer ownership from checkpoint content::

      [
        {
          "old_thread_id": "conversation-1",
          "user_id": "alice",
          "agent_id": "chatbot-agent"
        },
        {
          "old_thread_id": "legacy-conversation-2",
          "user_id": "bob",
          "agent_id": "chatbot-agent",
          "thread_id": "conversation-2"
        }
      ]

``thread_id`` is the public ID after migration. If you omit it, the public ID is
``old_thread_id``. Each source must have checkpoints. A source cannot map to more
than one owner or agent. Duplicate destinations, existing destination rows, and
overlapping source and destination IDs cause the whole operation to fail.

Preview and apply
-----------------

The command performs a dry run unless you set ``--apply=true``. The report shows
the mappings and row counts. It does not print checkpoint content or database
credentials.

Preview SQLite migration::

   uv run python -m langgraph_agent_toolkit.core.memory.migration \
     --backend=sqlite --sqlite-path=checkpoints.db --manifest=manifest.json

Review the report. Apply the same manifest while all workers remain stopped::

   uv run python -m langgraph_agent_toolkit.core.memory.migration \
     --backend=sqlite --sqlite-path=checkpoints.db --manifest=manifest.json \
     --apply=true

For PostgreSQL, put a libpq connection string in an environment variable through
your normal secret management process. Do not put a password in command arguments.
The default variable name is ``DATABASE_URL``::

   uv run python -m langgraph_agent_toolkit.core.memory.migration \
     --backend=postgres --postgres-env=DATABASE_URL --schema=public \
     --manifest=manifest.json

Add ``--apply=true`` only after you review the dry run. Set ``--schema`` to the
schema that the checkpoint saver uses.

Each apply operation validates the current database again. It runs in one
transaction. SQLite updates ``checkpoints`` and ``writes``. PostgreSQL updates
``checkpoints``, ``checkpoint_blobs``, and ``checkpoint_writes``. All checkpoint
namespaces move together. Missing write tables are permitted for older schemas.
The checkpoint table is required. PostgreSQL also requires the blob table.
Other missing tables or SQL errors are not ignored.

After the transaction commits, start only the new workers. Verify a migrated
conversation through its original public ID and the correct user identity. Use
the shared deployment token in trusted mode. Verify that a different user ID
selects a separate conversation. Keep the backup until validation is complete.
For rollback, stop all workers and restore the verified backup. Do not attempt
to repair ownership by editing storage hashes manually.

Remove inactive conversation history
------------------------------------

Retention is explicit maintenance. It is not an automatic background deletion
policy. Stop all checkpoint writers and make a verified backup before this work.
Use ``--before`` instead of ``--manifest``::

   uv run python -m langgraph_agent_toolkit.core.memory.migration \
     --backend=sqlite --sqlite-path=checkpoints.db \
     --before=2025-01-01T00:00:00Z

Review the candidate thread IDs and row counts. Repeat with ``--apply=true`` to
delete those threads. PostgreSQL uses the same flags as migration, with
``--before`` in place of ``--manifest``.

The cutoff must include a timezone. A thread is eligible only when its latest
checkpoint across all namespaces is older than the cutoff. Only canonical
``lat:v1:`` storage IDs are eligible. Legacy raw IDs remain untouched. The command
deletes whole threads, including blobs and pending writes, in one transaction.
It does not compact checkpoints inside active threads.

Age refers to the saved checkpoint timestamp. It does not prove that an external
job or a human approval is complete. Review these cases before applying retention.
Missing or invalid timestamps stop the operation. SQLite scans the complete
stored history for eligible storage IDs. It reads JSON and msgpack timestamps
without constructing serialized Python objects. Other encodings require a
separate reviewed conversion. PostgreSQL aggregates timestamps in the database.
Plan a maintenance window that permits this scan.
Connection and Admission Limits
-------------------------------

Version 0.10.0 accepts eight active HTTP requests per worker by default. Excess
requests receive 503 with ``error_code=service_busy`` and ``Retry-After: 1``
before an agent run starts. Tune ``REQUEST_MAX_CONCURRENT`` with the worker,
database, and provider limits. The optional waiting queue defaults to zero.
Clients must handle overload responses on streaming endpoints before parsing
stream events.

Managed OpenAI and Azure model calls use connection limits, keepalive expiry,
phase timeouts, and a retry count from the OpenAI Python SDK 3.13.0 baseline:

.. code-block:: ini

   LLM_HTTP_MAX_CONNECTIONS=1000
   LLM_HTTP_MAX_KEEPALIVE_CONNECTIONS=100
   LLM_HTTP_KEEPALIVE_EXPIRY=5.0
   LLM_HTTP_CONNECT_TIMEOUT=5.0
   LLM_HTTP_READ_TIMEOUT=600.0
   LLM_HTTP_WRITE_TIMEOUT=600.0
   LLM_HTTP_POOL_TIMEOUT=600.0
   LLM_HTTP_MAX_RETRIES=2

Timeouts are seconds. These are fixed toolkit defaults. They do not change
automatically when the SDK version changes. Existing environment overrides
retain their values. Remove old numeric ``LLM_HTTP_*`` overrides to adopt these
defaults, or keep deliberate deployment-specific limits. Explicit model timeouts
and retry counts still take precedence. ``LLM_HTTP_MAX_POOLS=32`` and
``LLM_HTTP_SHUTDOWN_TIMEOUT=10.0`` remain unchanged.

If an existing ``LLM_HTTP_MAX_CONNECTIONS`` override is below 100, also set
``LLM_HTTP_MAX_KEEPALIVE_CONNECTIONS`` at or below that limit. The new idle
default of 100 otherwise fails configuration validation.

The read timeout measures inactivity for each provider attempt. It is not the
total API request deadline. ``REQUEST_TIMEOUT`` remains 300 seconds by default
and can be lower in a deployment. A 600-second provider timeout does not extend
that API deadline. Client and ingress timeouts require separate configuration.

The API Docker image installs and selects the OpenAI SDK's aiohttp adapter.
Rebuild the image and recreate the container. Remove any explicit ``httpx``
transport setting to inherit the new image default. Keep that setting to retain
HTTPX. The example ``.env`` no longer sets a transport. Python installations
outside the image retain HTTPX; install ``openai-aiohttp`` and set
``LLM_HTTP_ASYNC_TRANSPORT=aiohttp`` to select the same adapter there.
The tested SDK 3.13.0 aiohttp adapter does not enforce the separate HTTPX write
timeout or maximum idle-connection count. See :doc:`reliability` for its other
transport differences.

Service startup rebuilds agents that register a ``graph_factory``. The factory
must create the graph and its concrete model clients for the current lifespan.
MCP tools are passed to that same build. Graphs without factories retain their
explicit dependencies and resource ownership.

Configure ``/health/live`` to trigger supervisor recovery for stalled request
cleanup. ``/health/ready`` remains a traffic-routing check and can fail during
database saturation. See :doc:`reliability` for the settings and cleanup limits.

Provider messages and stream completion
---------------------------------------

``ChatMessage`` now includes optional ``usage_metadata`` with input, output, and
total token counts. ``null`` means counts are unavailable. AI history imports
accept the same field and ``response_metadata``. Clients that reject unknown
response fields must allow ``usage_metadata``.

Refusal text is preserved in ``response_metadata["refusal"]``. If the provider
returns empty content with a refusal, the public message uses that refusal text
as its content. Ordinary nonempty model content remains unchanged.

``AgentClient.stream`` and ``astream`` now raise ``AgentClientError`` when an SSE
response reaches EOF without ``[DONE]``. This includes a stream that sent an AI
message but did not complete its framing. Existing toolkit SSE routes already
send this marker. A caller can still close a stream early. The JSON Lines
protocol does not use ``[DONE]``.

Rejected model-provider credentials now return HTTP 503 with
``error_code=model_authentication_failed`` from ``/invoke``. This is a service
configuration failure. The caller's API authentication is separate. Streams
that have sent headers emit a fixed error message. Correct the provider
credential before retrying. Toolkit responses and logs omit provider error
text and tracebacks for these authentication failures in all environment modes.
