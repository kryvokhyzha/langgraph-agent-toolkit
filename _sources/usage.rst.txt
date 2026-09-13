Usage Guide
===========

Start the service with :doc:`quickstart`. Use :doc:`installation` to select
optional dependencies. Use :doc:`environment_setup` for model and database
configuration. This guide covers the toolkit HTTP API and ``AgentClient``.

Connect to a running service
----------------------------

The API listens on port 8080 by default. Use ``http://localhost:8080`` from
the same computer. Use the service's HTTPS address from another computer.
The interactive API reference is at ``/docs``.

Set these variables in the shell that runs the examples. Replace the secret
with the service's configured bearer token:

.. code-block:: bash

   export AGENT_API_URL=http://localhost:8080
   export AUTH_SECRET=replace-with-your-deployment-secret

Check readiness, then read the available agent names:

.. code-block:: bash

   curl --fail-with-body "$AGENT_API_URL/health/ready"
   curl --fail-with-body "$AGENT_API_URL/info" \
     -H "Authorization: Bearer ${AUTH_SECRET}"

``/health/ready`` is public. ``/info`` requires authentication. It returns
``agents`` and ``default_agent``. It does not list model configurations.
Use an agent's ``key`` in request paths. The examples below use the built-in
``create-agent``. Replace this value if your service loads another agent:

.. code-block:: bash

   export AGENT_ID=create-agent

Authentication and memory identity
----------------------------------

For one deployment per client application, keep the default
``AUTH_MODE=trusted`` and one ``AUTH_SECRET``. ``user_id`` is optional.
The trusted application can supply it for a user's memory identity. If it
omits the field, the service uses ``AUTH_SERVICE_USER_ID`` (default: ``service``).
Keep the shared token in that application.
Anyone who has this token can select user identities in that deployment.

The examples below assume trusted mode and use ``user_id="user-1"``.
In ``AUTH_MODE=token``, the server derives the identity from the bearer token.
Omit ``user_id`` or supply the same identity. A shared token in token mode
identifies ``AUTH_SERVICE_USER_ID``, which defaults to ``service``. It does
not authorize other user IDs. See :doc:`migration` for both authentication
modes and existing-data migration.

Use the two memory identifiers for different purposes:

* ``thread_id`` selects one conversation and its short-term checkpoint state.
  Omit it on an invocation to create a new public thread ID. Save the returned
  ``thread_id`` and send it on later requests for that conversation.
* ``user_id`` is the stable user identity for long-term memory across threads.
  The agent must have a store and logic to read and write that memory.
  Supplying this value alone does not create long-term memory.

The API scopes checkpoints by user, agent, and public thread ID. Use all three
consistently for invocation, streaming, and history requests. Reusing a public
thread ID for a different user or agent selects different checkpoint state.
In trusted mode, an omitted ``user_id`` selects ``AUTH_SERVICE_USER_ID``.

Configure SQLite or PostgreSQL for durable checkpoints. If an agent has no
checkpointer and ``MEMORY_BACKEND`` is unset, the service uses process-local
memory. It is lost when the worker restarts. Use :doc:`deployment` and
:doc:`reliability` before adding workers.
See :doc:`integrations` for the distinction between checkpoints and stores.

Call the HTTP API
-----------------

Send the agent input inside the ``input`` object. Identity and model selection
fields belong at the top level:

.. code-block:: bash

   curl --fail-with-body "$AGENT_API_URL/$AGENT_ID/invoke" \
     -H "Authorization: Bearer ${AUTH_SECRET}" \
     -H 'Content-Type: application/json' \
     -d '{"input":{"message":"Hello"},"thread_id":"conversation-1","user_id":"user-1"}'

The response is a ``ChatMessage``. It includes ``type``, ``content``,
``thread_id``, and ``run_id``. The ``run_id`` identifies this agent run and can
be used for feedback. It is not a conversation identifier.

Use ``/invoke``, ``/stream``, or ``/stream/jsonl`` without an agent prefix to
select the service's default agent. The corresponding history and feedback
routes also have default-agent aliases.

To select a configured model, add ``model_config_key`` to the request. The key
must refer to your ``MODEL_CONFIGS`` configuration. ``model_name`` and
``model_provider`` are also request fields. The selected agent must support
runtime model selection. A custom graph can use a fixed model. Do not put
identity or checkpoint fields in ``agent_config``; the API rejects them.

``input.message`` can also contain LangChain content blocks. For example, use
this helper with an open client and a real image URL:

.. code-block:: python

   def describe_image(client, image_url):
       return client.invoke(
           {"message": [
               {"type": "text", "text": "Describe this image."},
               {"type": "image", "url": image_url},
           ]},
           user_id="user-1",
       )

The selected agent and model must support the content type. URL accessibility,
file size, and model input limits remain provider requirements. Base64 media
blocks also require ``mime_type``. ``MULTIMODAL_MAX_ATTACHMENTS`` can limit the
number of attachments in one request.

Use the synchronous client
--------------------------

Import ``AgentClient`` from the installed package. Supply the full service URL,
including its port. The constructor reads ``/info`` by default and selects the
service's default agent when ``agent`` is omitted.

Use a context manager to close the client's HTTP connections:

.. code-block:: python

   import os

   from langgraph_agent_toolkit.client import AgentClient, AgentClientError


   try:
       with AgentClient(
           base_url=os.environ["AGENT_API_URL"],
           agent=os.environ["AGENT_ID"],
           auth_secret=os.environ["AUTH_SECRET"],
       ) as client:
           first = client.invoke({"message": "My preferred language is English."}, user_id="user-1")
           print(first.content)
           print("Conversation:", first.thread_id)

           second = client.invoke(
               {"message": "Which language did I request?"},
               thread_id=first.thread_id,
               user_id="user-1",
           )
           print(second.content)
           print("Run:", second.run_id)
           print("Usage:", second.usage_metadata)
   except AgentClientError as exc:
       print(f"The request failed: {exc}")

``AgentClient`` reads ``AUTH_SECRET`` from the process environment when
``auth_secret`` is omitted. It does not load your ``.env`` file. Set environment
variables before starting the client or load that file in your application.
Keep a shared client open for the lifetime of your application. Do not create
a new client for each request.

Stream responses
----------------

The SSE endpoint emits frames with these payloads:

* ``type="token"``: incremental text.
* ``type="message"``: a complete ``ChatMessage``. Tool and custom messages can
  occur before the final assistant message.
* ``type="error"``: a failed run. The content is an error description.

A normal SSE stream ends with ``data: [DONE]``. JSON Lines uses the same typed
payloads, one JSON object per line, without a ``data:`` prefix or completion
marker. HTTP 200 alone does not establish that a stream succeeded.

Use ``curl -N`` to print frames as they arrive:

.. code-block:: bash

   curl --fail-with-body -N "$AGENT_API_URL/$AGENT_ID/stream" \
     -H "Authorization: Bearer ${AUTH_SECRET}" \
     -H 'Content-Type: application/json' \
     -d '{"input":{"message":"Continue our conversation."},"thread_id":"conversation-1","user_id":"user-1","stream_tokens":true}'

Replace ``/stream`` with ``/stream/jsonl`` for JSON Lines. Set
``stream_tokens=false`` to receive complete messages without token events.
Consume error events even when the HTTP request succeeds.

The client returns ``str`` for tokens and ``ChatMessage`` for complete
messages. It raises ``AgentClientError`` for error events, invalid frames, HTTP
failures, and an SSE stream that ends without its completion marker. JSON Lines
does not have a completion marker; a clean end alone cannot prove that the
graph finished. A final response and your application contract must establish
completion. Token events and complete messages can contain the same text.
Do not append both to the final answer.

Use ``get_info=False`` and an explicit agent for an asynchronous client. This
avoids the constructor's synchronous ``/info`` request:

.. code-block:: python

   import asyncio
   import os
   from contextlib import aclosing

   from langgraph_agent_toolkit.client import AgentClient, AgentClientError


   async def main():
       async with AgentClient(
           base_url=os.environ["AGENT_API_URL"],
           agent=os.environ["AGENT_ID"],
           auth_secret=os.environ["AUTH_SECRET"],
           get_info=False,
       ) as client:
           try:
               async with aclosing(client.astream(
                   {"message": "Give me a short greeting."},
                   thread_id="conversation-2",
                   user_id="user-1",
               )) as events:
                   async for event in events:
                       if isinstance(event, str):
                           print(event, end="", flush=True)
                       else:
                           print("\nMessage:", event.type, event.content)
                           print("Conversation:", event.thread_id)
           except AgentClientError as exc:
               print(f"\nThe stream failed: {exc}")


   asyncio.run(main())

Use ``stream()`` for synchronous SSE. Use ``stream_jsonl()`` or
``astream_jsonl()`` for JSON Lines. Close a stream iterator if you stop reading
early. Use ``contextlib.closing`` for synchronous streams and
``contextlib.aclosing`` for asynchronous streams.

An async client must stay in one event loop until it closes. ``async with``
closes its owned sync and async HTTP clients. If you inject ``http_client`` or
``async_http_client``, your application must close that injected client.
The ``stream_timeout`` read limit measures idle time between reads. See
:doc:`reliability` for timeout and connection settings.

Read and change history
-----------------------

History operations require ``thread_id``. They read or change short-term
checkpoints only. Supply the same agent and user that own the conversation.

Read a page through HTTP:

.. code-block:: bash

   curl --fail-with-body --get "$AGENT_API_URL/$AGENT_ID/history" \
     -H "Authorization: Bearer ${AUTH_SECRET}" \
     --data-urlencode 'thread_id=conversation-1' \
     --data-urlencode 'user_id=user-1' \
     --data-urlencode 'limit=100' \
     --data-urlencode 'offset=0'

The response contains ``messages``, ``total``, and ``next_offset``. Read the
next page with ``offset=next_offset`` until that value is null. The default
page has at most 100 messages. The requested limit must be from 1 to 1000.
``HISTORY_MAX_PAGE_SIZE`` can set a smaller server limit.

With an open ``AgentClient``, use these methods:

.. code-block:: python

   def read_conversation(client, thread_id):
       offset = 0
       while True:
           page = client.get_history(thread_id, user_id="user-1", offset=offset, limit=100)
           for message in page.messages:
               print(message.type, message.content)
           if page.next_offset is None:
               break
           offset = page.next_offset


   def delete_conversation(client, thread_id):
       return client.clear_history(thread_id=thread_id, user_id="user-1")

``clear_history()`` deletes all checkpoint versions for that thread. It does
not delete the user's long-term store records. ``add_messages()`` appends
``MessageInput`` values or dictionaries with ``type`` and ``content``. It does
not run the agent. Use ``aget_history()``, ``aclear_history()``, and
``aadd_messages()`` for asynchronous calls. See :doc:`migration` before changing
an existing deployment's checkpoint identifiers.

Interpret metadata and failures
-------------------------------

``usage_metadata`` contains provider token counts when available. ``None``
means that counts are unavailable. It does not mean zero tokens. These counts
belong to the returned model message; they are not a total for every model call
in a graph with several steps.

Inspect ``response_metadata`` when the provider supplies completion details.
For example, ``finish_reason="length"`` indicates a response that reached its
output limit. OpenAI and Azure refusals preserve their text in
``response_metadata["refusal"]``. An empty message with ``tool_calls`` can be a
valid intermediate message. Do not treat every empty content field as a
transport failure.

Use ``create_feedback()`` or ``acreate_feedback()`` to record a score for a
returned ``run_id``. Supply ``key``, ``score``, and the same ``user_id``. The
selected agent's observability backend receives the feedback. Token users must
also pass ``feedback_token=reply.feedback_token`` from the corresponding run
response. Use the same agent as the original request. The server requires a
separate ``FEEDBACK_SIGNING_SECRET`` for these users. Trusted backend feedback
keeps its existing request shape. See :doc:`migration` for the upgrade and
:doc:`langfuse_compatibility` for Langfuse configuration.

Before a response starts, the API uses HTTP status codes. Its JSON body contains
``detail`` and can include ``error_code``:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Status
     - Meaning
   * - 401 / 403
     - The service token is invalid, or the requested user does not match it.
   * - 409
     - The conversation is busy or its queue wait expired.
   * - 413 / 422
     - The request exceeds a configured size limit or fails validation.
   * - 429
     - The model provider rate limit was exceeded.
   * - 503
     - The worker is full, the database or provider is unavailable, or the
       provider rejected its configured credentials. Use ``error_code`` when
       present to distinguish these cases.
   * - 504
     - The request deadline or model timeout expired.

For example, ``service_busy`` means the worker admission limit was reached.
``model_authentication_failed`` means the provider rejected the server's
configured credentials. Changing the client's service token does not fix
that provider error.

After stream headers are sent, failures use an error event. Handle
``AgentClientError`` and retain partial text separately from completed answers.
The client does not automatically retry agent requests. A failed request can
already have run tools or saved checkpoints. See :doc:`reliability` for retry,
admission, and recovery behavior.

For HTTP failures, ``AgentClientError.status_code`` contains the response status.
``error_code`` contains a valid service error code when available.
``retry_after`` preserves the ``Retry-After`` header as a string. The header can
contain seconds or an HTTP date. These attributes are ``None`` when unavailable.
Errors received after streaming starts do not imply a new HTTP status.
Use this information to handle overload or configuration failures. Do not
automatically replay a run that may have performed a tool action.

Resume a conversation in the UI
-------------------------------

The Streamlit URL contains ``agent`` and ``thread_id``. The page selects and
validates the agent before reading history. New Chat creates a new thread.
Changing the agent starts a new conversation. Opening another conversation
URL in an existing session replaces the displayed history.

Share/resume chat uses the current app URL and keeps its deployment path and
HTTP scheme. The link does not contain a bearer token or user ID. Opening it
does not grant access to another user. The API still applies authentication
and conversation ownership. The UI uses its configured user identity; it does
not provide a public multiuser login system.

Register a custom agent
-----------------------

Create an importable Python module that exports an ``Agent`` or a compiled
LangGraph graph. This deterministic example calls no model or external service.
Create an empty ``my_app/__init__.py``. Put this code in ``my_app/agent.py``:

.. code-block:: python

   from langchain_core.messages import AIMessage
   from langgraph.graph import END, START, MessagesState, StateGraph

   from langgraph_agent_toolkit.agents.agent import Agent


   def reply(state: MessagesState):
       return {"messages": [AIMessage(content="The custom agent is running.")]}


   builder = StateGraph(MessagesState)
   builder.add_node("reply", reply)
   builder.add_edge(START, "reply")
   builder.add_edge("reply", END)

   agent = Agent(
       name="support-agent",
       description="A deterministic custom agent example.",
       graph=builder.compile(),
   )

The service puts request messages in ``state["messages"]``. Each node returns
new messages under the same key. ``MessagesState`` adds them to the conversation.
The service assigns its checkpointer when this graph loads.

Set the import string and registered agent name in the service environment:

.. code-block:: ini

   AGENT_PATHS=["my_app.agent:agent"]
   DEFAULT_AGENT=support-agent

Run the service from the directory that contains ``my_app`` or install
``my_app`` as a package. The source file does not need to be inside the toolkit.
For a custom image, copy or install ``my_app`` into the image too. The standard
API Dockerfile copies only the toolkit package. Restart the service after
changing ``AGENT_PATHS``. This example exposes ``/support-agent/invoke``.

Supply ``Agent.graph_factory`` when the service must rebuild concrete models
or tools for each lifespan. Keep application-owned stores and injected clients
open while their graphs run. Do not change ``core/settings.py`` to register
agents.

For a model that calls tools, see the existing
`native create_agent blueprint
<https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/langgraph_agent_toolkit/agents/blueprints/create_agent/agent.py>`_.
Its `shared graph builder
<https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/langgraph_agent_toolkit/agents/blueprints/create_agent/_shared.py>`_
shows model setup, tools, and middleware.

Choose a graph pattern with :doc:`integrations`. See :doc:`mcp` for MCP tools
and :doc:`deepagents` for Deep Agents. Use :doc:`deployment` for Uvicorn,
Gunicorn, container settings, and worker recovery. Use :doc:`testing` for tests
that need no external model and :doc:`live_llm_testing` for an explicit live
provider check.
