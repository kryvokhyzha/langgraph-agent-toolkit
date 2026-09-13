Application Onboarding
======================

Use this guide to add the toolkit to an application or upgrade an existing
integration. Keep application agents, tools, and deployment files in your own
repository. Use :doc:`quickstart` for a local demonstration without model keys.

1. Define the application integration
-------------------------------------

Decide how the application uses the toolkit:

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Application need
     - Integration
     - Next guide
   * - Call an existing agent service
     - Use the HTTP API or ``AgentClient``. The service owns graph execution.
     - :doc:`usage`
   * - Serve application agents
     - Register importable agents through ``AGENT_PATHS``. Run the toolkit API.
     - :doc:`usage`
   * - Run graphs inside another application
     - Own graph execution, authentication, resource lifetime, and concurrency.
       HTTP service controls do not apply to direct graph calls.
     - :doc:`integrations` and :doc:`reliability`

Choose the smallest agent pattern that meets the requirement. Start with a
native tool-calling agent or a fixed graph. Add MCP or Deep Agents when their
capabilities are required. See :doc:`integrations` for the differences.

2. Install and lock the required dependencies
---------------------------------------------

Select a supported Python version and only the extras that the application
uses. Keep the toolkit requirement and the resolved application lockfile in
version control. See :doc:`installation` for provider, server, UI, and tool
extras. Python 3.14 requires the conditional dependency minimums in that guide.

Choose one of ``langfuse-v2``, ``langfuse-v3``, or ``langfuse-v4`` when the
application must retain a specific SDK generation. The SDK selector does not
select the server version. Check :doc:`langfuse_compatibility` against the
deployed server. Source-checkout dependency groups are not published extras.

For an upgrade, read :doc:`migration` before changing the application lockfile.
Review the resulting changes to LangChain, LangGraph, Pydantic, model SDKs, and
observability SDKs together. Preserve the previous lockfile and application
image for rollback.

3. Register agents and define resource ownership
------------------------------------------------

Configure ``AGENT_PATHS`` with import strings and set ``DEFAULT_AGENT`` to a
registered agent name. The application module must be installed or copied into
the service image. The standard API image contains only the toolkit package.
Use the custom-agent example in :doc:`usage`.

Keep network operations out of agent imports. Use ``Agent.graph_factory`` to
create concrete models and tools during each service lifespan. The service can
then supply its managed connections and configured MCP tools. Keep
application-owned clients and stores open until their graphs stop. Close these
resources during application shutdown. See :doc:`reliability` and :doc:`mcp`.

4. Set identity and memory contracts
------------------------------------

For one service deployment per client application, keep the default
``AUTH_MODE=trusted`` and one ``AUTH_SECRET``. Keep the shared token in the
trusted application backend. ``user_id`` is optional. That backend can supply
a stable ID for each authenticated user. If it omits the field, the service
uses ``AUTH_SERVICE_USER_ID`` (default: ``service``). Use a separate database
or PostgreSQL schema for each deployment.

For direct access by untrusted users, use ``AUTH_MODE=token`` and individual
tokens. The service derives identity from the token. See :doc:`migration` for
both modes and token-user feedback requirements.

Keep these identifiers separate:

* ``user_id`` identifies one user across conversations. An agent can use it
  for a long-term store namespace. The agent must have a store and explicit
  memory logic. Passing this field alone does not create long-term memory.
* ``thread_id`` identifies one conversation and its short-term checkpoint
  state. Save the ID returned by the first request. Reuse it for later turns.
* The agent name selects the graph and forms part of the checkpoint scope.
  Keep it stable when resuming existing conversations.
* ``run_id`` identifies one execution. Use it for feedback and tracing.

The API scopes checkpoints by user, agent, and public thread ID. Use the same
values for invocation, streaming, and history. Configure ``MEMORY_BACKEND`` for
durability. SQLite supplies checkpoints only. A long-term store is a separate
resource. Clearing conversation history does not clear that store.

5. Connect the application client
---------------------------------

Use the request and response examples in :doc:`usage`. Keep one ``AgentClient``
open for the application lifetime. Close it during shutdown. For asynchronous
applications, use async methods and avoid a synchronous startup request on the
event loop.

An asynchronous client must stay in one event loop until ``aclose()`` finishes.
Streamlit creates a new loop on each ``asyncio.run()``. Close the cached client
in a ``finally`` block before each page run ends, including ``st.rerun()`` and
``st.stop()``. Keep the authenticated user's identity separate from cached
client connections. The included Streamlit page uses this cleanup pattern.

Handle HTTP failures and streaming error events. An SSE stream succeeds only
when the application receives a complete result and the completion marker.
Close stream iterators when a caller stops reading. Preserve partial text
separately from completed answers. Do not automatically retry a run that can
have performed a tool action. Use :doc:`reliability` for retry boundaries.

History requests need an explicit ``thread_id``. Follow ``next_offset`` when
reading paginated history. Token users must retain the returned
``feedback_token`` with the corresponding response. Trusted backend feedback
keeps its existing request shape. See :doc:`migration` for the wire changes.

6. Verify the application before production
-------------------------------------------

First, use deterministic models and a disposable database to check the
application's API contract. Then test the real deployment dependencies in a
separate environment. The toolkit test suite verifies package behavior; it
does not replace tests for application agents, tools, or infrastructure.

Verify these application outcomes:

* The service becomes ready and lists the expected agents in ``/info``.
* Invocation, streaming, history, and feedback use the correct user and agent.
* Two users cannot read each other's conversations. Two threads for one user
  have separate histories. Test shared user memories only if a store exists.
* A restart preserves history and any pending approval that the application
  supports.
* Cancellation and concurrent requests release capacity and preserve ordered
  updates. Failed tools do not cause duplicate external writes.
* Database interruption and worker failure recover within the deployment's
  limits. Overload responses reach the client as errors.
* The selected model returns valid tool calls and response formats. The
  configured observability server receives traces and feedback.

Use :doc:`testing` for test layers, :doc:`live_llm_testing` for a small real-model
check, and :doc:`load_testing` for capacity and failure tests. Configure
``/health/ready`` for traffic routing and ``/health/live`` for supervisor
recovery. Tune worker, request, model, and database limits together. See
:doc:`deployment` and :doc:`reliability`.

7. Upgrade existing data and release
------------------------------------

When upgrading from raw checkpoint IDs to 0.10.x, create the ownership manifest
described in :doc:`migration`. Verify a database backup and the migration dry
run before applying changes. Stop all checkpoint writers during migration.
Do not run old and new workers against the same checkpoint tables.

Test the migrated conversations through their public IDs and original user
identities. Roll out the tested application image and configuration together.
Check readiness, error rates, request duration, database capacity, and trace
delivery. If data migration must be rolled back, stop writers and restore the
verified backup as described in :doc:`migration`.
