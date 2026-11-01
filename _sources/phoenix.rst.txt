Phoenix observability
=====================

Use Phoenix to store OpenInference traces, prompt versions, and user feedback.
Phoenix is an optional backend. It can replace Langfuse for these toolkit
operations. It does not replace the checkpointer or long-term memory store.

Install and select the backend
-------------------------------

Add the optional dependencies to the application:

.. code-block:: bash

   uv add 'langgraph-agent-toolkit[phoenix]'

For a source checkout, use:

.. code-block:: bash

   uv sync --frozen --no-install-project --extra phoenix

The extra installs the Phoenix client, OpenTelemetry SDK, and LangChain callback
dependencies. It does not install or start the Phoenix server.
Run the server separately or use an existing remote deployment.

Set these values before the application starts:

.. code-block:: ini

   OBSERVABILITY_BACKEND=phoenix
   PHOENIX_ENDPOINT=http://localhost:6006
   PHOENIX_PROJECT_NAME=agent-service
   PHOENIX_TIMEOUT=5
   # PHOENIX_API_KEY=<system-api-key>

.. list-table:: Toolkit settings
   :header-rows: 1
   :widths: 31 23 46

   * - Setting
     - Default
     - Purpose
   * - ``PHOENIX_ENDPOINT``
     - ``http://localhost:6006``
     - Base HTTP URL for the Phoenix API and trace collector.
   * - ``PHOENIX_PROJECT_NAME``
     - ``langgraph-agent-toolkit``
     - Project for traces from this service.
   * - ``PHOENIX_API_KEY``
     - Unset
     - API key for a Phoenix server that requires authentication.
   * - ``PHOENIX_TIMEOUT``
     - ``5`` seconds
     - Network timeout for REST and trace export clients.

``PHOENIX_ENDPOINT`` is a toolkit setting. Use the base URL, without
``/v1/traces``. The adapter derives the trace path from this URL.
For example, a base URL of ``https://phoenix.example.com/monitoring`` uses
``https://phoenix.example.com/monitoring/v1/traces`` for trace export.
Configure the proxy to preserve that path.

For Phoenix Cloud, copy the deployment base URL from its settings.
Keep any space or proxy path in that URL. Set ``PHOENIX_API_KEY`` to a key for
that deployment. For an authenticated self-hosted server, use a system API key.
The adapter sends bearer authentication to the API and trace collector.
Toolkit ``AUTH_SECRET`` authenticates callers of the agent API.
It serves a different connection from ``PHOENIX_API_KEY``.
See the official `API key guide
<https://arize.com/docs/phoenix/settings/api-keys>`_.

The toolkit exports OTLP over HTTP. Phoenix serves this collector on port
``6006`` by default. Port ``4317`` serves its separate gRPC collector.
Use ``https://`` for a remote production endpoint. Keep certificate validation
enabled. Configure the server, proxy, and certificate trust for the deployment.
See the official `encryption guide
<https://arize.com/docs/phoenix/self-hosting/advanced/encryption>`_.

Trace behavior
---------------

The API attaches a separate callback handler to each run. It records the
LangGraph execution tree, model calls, tools, and reported token usage.
Native LangChain agents and Deep Agents use the same callback path.
An agent must propagate the LangChain run configuration to custom nested
runnables for those operations to appear in the trace.

The response ``run_id`` identifies the Phoenix trace. The adapter converts the
UUID to the 32-character hexadecimal trace ID that OpenTelemetry uses.
The public ``thread_id`` becomes the trace session ID.
The user identity becomes the trace user ID when it is available.
These attributes group telemetry. They do not grant access to checkpoints or
create long-term memory.

The adapter owns a separate tracer provider. It does not register a global
provider or enable process-wide LangChain instrumentation.
Do not add ``LangChainInstrumentor().instrument()`` for the same agent runs.
That can produce duplicate spans and change tracing in other components.

Trace export uses a background batch processor with a one-second batch delay.
It does not send an HTTP request for each model callback.
``OTEL_BSP_SCHEDULE_DELAY`` can override that delay.
Prompt and feedback operations use the
Phoenix REST client. The API runs blocking feedback operations outside its
event loop. Use ``apull_prompt`` when custom async code reads a prompt.

The service flushes telemetry during normal shutdown.
``OBSERVABILITY_SHUTDOWN_TIMEOUT`` limits this cleanup to 10 seconds by default.
A service-owned tracer provider does not register a second exit hook.
The service closes each distinct platform attached to its agents once.
This includes an explicit ``Agent(observability=...)`` override.
It also closes these platforms if later graph setup fails.
A standalone script must call ``before_shutdown`` for the platform it creates.
A forced process exit can lose spans that remain in memory.
The batch queue is not a durable message queue. A long collector outage or a
full queue can also cause trace loss. Keep checkpoint persistence independent
from trace delivery.

Use the native OpenTelemetry batch processor settings to tune export after a
load test. ``LANGFUSE_FLUSH_AT`` and ``LANGFUSE_FLUSH_INTERVAL`` do not
configure Phoenix export.
``OTEL_BSP_MAX_QUEUE_SIZE`` limits queued spans.
``OTEL_BSP_MAX_EXPORT_BATCH_SIZE`` limits spans in one export batch.
``OTEL_BSP_SCHEDULE_DELAY`` sets the batch delay in milliseconds.
Queue capacity counts spans, rather than API requests. A delegated agent can
create many spans for one request. A larger queue uses more memory and does
not add durable storage. See the `OpenTelemetry batch processor reference
<https://opentelemetry-python.readthedocs.io/en/latest/sdk/trace.export.html>`_.
See the `Phoenix tracing setup guide
<https://arize.com/docs/phoenix/tracing/how-to-tracing/setup-tracing/setup-using-phoenix-otel>`_
for the SDK transport and batching model.

Prompt management
------------------

The toolkit supports Phoenix prompt creation, new versions, reads, and deletion.
Phoenix requires model metadata when it creates a prompt version.
This metadata describes the stored prompt. It does not change the model that
the toolkit agent uses.

.. code-block:: python

   from langgraph_agent_toolkit.core.observability.factory import ObservabilityFactory

   observability = ObservabilityFactory.create("phoenix")
   try:
       observability.push_prompt(
           name="example-answer",
           prompt_template=[
               {"role": "system", "content": "Give one short answer."},
               {"role": "user", "content": "Answer this question: {question}"},
           ],
           metadata={
               "model_name": "example-model",
               "model_provider": "OPENAI",
               "template_format": "F_STRING",
               "labels": ["staging"],
           },
           force_create_new_version=False,
       )
       prompt, version = observability.pull_prompt(
           "example-answer",
           label="staging",
           return_with_prompt_object=True,
       )
       print(prompt.format_messages(question="What does this service do?"))
       print(version.id)
   finally:
       observability.before_shutdown()

This example stores and reads a prompt. It makes no model request.
Replace ``example-model`` with the model name for the application.
``model_provider`` defaults to ``OPENAI``. Set it explicitly for another
provider. ``description`` sets the prompt description.
The adapter stores other metadata keys as Phoenix prompt metadata.
``PHOENIX_PROJECT_NAME`` groups traces. It does not create a separate prompt
repository. Use distinct prompt names for each application and environment
when they share one Phoenix server.

Set ``force_create_new_version=False`` to keep an existing prompt.
``remote_first=True`` on the factory has the same effect for an existing
remote prompt. Normal creation adds a new version.
Only a missing prompt permits creation on these read-first paths.
A connection or authorization failure must be resolved first.

Use the same template format when you create and read a prompt.
Phoenix supports ``f-string`` and ``mustache`` through this adapter.
It does not support ``jinja2``. Convert an existing Jinja template before you
move it to Phoenix.
Use ``template_format=F_STRING`` or ``template_format=MUSTACHE`` in creation
metadata. ``NONE`` stores literal text without substitution.
The adapter rejects a read whose requested format differs from the stored
``F_STRING`` or ``MUSTACHE`` format. A ``NONE`` prompt keeps literal braces.
The toolkit converts text chat templates to ``ChatPromptTemplate``.
It rejects message placeholders and multimodal prompt templates.

Phoenix prompt version IDs differ from Langfuse integer versions.
Its prompt version tags can select a deployed revision.
The creation ``labels`` list creates these tags.
``pull_prompt(..., label="staging")`` and ``pull_prompt(..., tag="staging")``
select a tag. ``version`` and ``prompt_version_id`` select the native string
version ID. Select a tag or a version ID in one call.
``delete_prompt(name)`` deletes the remote prompt and all its versions.
Changing the observability backend does not copy prompt content or version
labels from another service. Create and verify the target prompts first.
See the official `prompt client guide
<https://arize.com/docs/phoenix/prompt-engineering/quickstart-prompts/quickstart-prompts-ui>`_.

Use ``ObservabilityChatPromptTemplate`` for a prompt in an agent graph.
Pass the existing platform instance so that the application owns one set of
connections and closes it at shutdown:

.. code-block:: python

   from langgraph_agent_toolkit.core.prompts.chat_prompt_template import (
       ObservabilityChatPromptTemplate,
   )

   def build_runtime_prompt(observability):
       return ObservabilityChatPromptTemplate.from_observability_platform(
           prompt_name="example-answer",
           observability_platform=observability,
           prompt_label="staging",
           load_at_runtime=True,
           cache_ttl_seconds=60,
           template_format="f-string",
       )

Call this function with the service's platform instance.
Use ``ainvoke`` in an async graph.
Set ``prompt_version`` to a native string version ID when the deployment needs
an immutable revision. Set ``prompt_label`` to follow a mutable version tag.

Custom ``PromptManager`` subclasses can publish text prompt files to Phoenix.
Set ``template_format="f-string"`` or ``"mustache"`` on the manager.
Pass ``prompt_metadata`` with ``model_name`` and the matching Phoenix
``template_format`` value. Use ``has_messages_placeholder=False``.
The manager's default ``jinja2`` format does not work with Phoenix.
Close ``manager.observability`` at shutdown when custom application code owns
the manager. Existing local ``ChatPromptTemplate`` objects do not need these
Phoenix prompt settings.

To link a model call to the exact remote prompt revision, get the native
prompt object and use its ID in OpenInference context:

.. code-block:: python

   from openinference.instrumentation import using_attributes

   async def run_with_prompt(observability, model, question, config):
       prompt, version = await observability.apull_prompt(
           "example-answer",
           label="staging",
           return_with_prompt_object=True,
       )
       with using_attributes(
           prompt_template_version=version.id,
           prompt_template_variables={"question": question},
       ):
           return await (prompt | model).ainvoke({"question": question}, config=config)

Run this chain inside the agent's callback context.
Pass its ``RunnableConfig`` to nested calls in a custom graph node.
Reading a remote prompt alone does not link it to later model spans.

Feedback
---------

Use the existing ``POST /feedback`` or ``POST /{agent_id}/feedback`` endpoint.
Send the ``run_id`` from the agent response. Token-bound authentication also
requires the response ``feedback_token``. Shared-token authentication retains
the same request contract. See :doc:`environment_setup` and :doc:`migration`.

The adapter records numeric feedback as a Phoenix trace annotation.
The feedback key becomes the annotation name. The annotation uses
``annotator_kind=HUMAN``. This is a trace-level result, so it does not require
the client to locate an internal model span.

The Phoenix adapter accepts these values in feedback ``kwargs``:

.. list-table:: Phoenix feedback options
   :header-rows: 1
   :widths: 25 75

   * - Option
     - Use
   * - ``comment`` or ``explanation``
     - Annotation explanation. ``comment`` has precedence.
   * - ``label``
     - Optional text label for the annotation result.
   * - ``metadata``
     - Additional annotation metadata.
   * - ``feedback_id``
     - Optional annotation identifier.

The API adds the authorized user identity to annotation metadata.
The adapter waits for normal background trace export.
It retries a missing trace while Phoenix completes background ingestion.
It does not flush the full telemetry queue for each feedback request.
It does not retry authorization errors.
It waits for the annotation request to finish. A successful agent response
does not imply that a later feedback request will succeed.
The retry window includes the configured batch delay and ``PHOENIX_TIMEOUT``.
Network timeouts apply to each attempt. ``PHOENIX_TIMEOUT`` is not a total
deadline for the full feedback operation.

The toolkit exposes feedback through the common observability interface.
Phoenix evaluation jobs, dataset management, experiments, and session
annotations use the native Phoenix client or UI.

Trace content and access
------------------------

Traces can contain prompts, messages, tool inputs, tool results, and model
outputs. Limit access to the Phoenix project and set an appropriate retention
period. A telemetry project name does not enforce conversation ownership.

Use OpenInference privacy settings before application startup when content
must stay out of traces:

.. code-block:: ini

   OPENINFERENCE_HIDE_INPUTS=true
   OPENINFERENCE_HIDE_OUTPUTS=true
   OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS=true

These settings hide the selected trace attributes. They do not remove data
from LangGraph state, checkpoints, application logs, or other integrations.
Review the `OpenInference privacy configuration
<https://github.com/Arize-ai/openinference/blob/main/spec/configuration.md>`_
for more specific controls.

Never put API keys in ordinary prompt strings or tool results.
Content hiding and secret masking cannot identify every sensitive plain string.
The adapter masks typed Pydantic ``SecretStr`` and ``SecretBytes`` values
before serialization. It handles nested models, dataclasses, and containers.
This protection does not classify passwords or personal data in plain text.

Self-hosted storage
-------------------

Phoenix stores its data in SQLite by default. Mount persistent storage at
``PHOENIX_WORKING_DIR`` for a local server whose data must survive replacement.
For a shared production deployment, configure PostgreSQL with
``PHOENIX_SQL_DATABASE_URL`` on the Phoenix server.
Back up that database and test restore before an upgrade.
These are Phoenix server settings. They do not configure toolkit checkpoints.
See the official `storage and deployment guide
<https://arize.com/docs/phoenix/self-hosting/architecture>`_.

Enable authentication before exposing a self-hosted server to other networks.
Provision a system key for the application. Keep the Phoenix server signing
secret in the server's secret store. Do not give it to the toolkit application.
See the official `authentication guide
<https://arize.com/docs/phoenix/self-hosting/features/authentication>`_.

The Phoenix server has separate web analytics settings.
Set ``PHOENIX_TELEMETRY_ENABLED=false`` on that server to disable its UI analytics.
Set ``PHOENIX_ALLOW_EXTERNAL_RESOURCES=false`` for the documented restricted
network mode. These settings do not disable traces that your application
explicitly exports. See the official `self-hosted privacy guide
<https://arize.com/docs/phoenix/self-hosting/security/privacy>`_.

Verification and limits
-----------------------

The tests use synthetic requests and a disposable Phoenix server.
They do not require a paid model key.
The server and SDK have independent versions.
The baseline uses Phoenix server ``20.26.1``, ``arize-phoenix-client 3.5.0``,
``openinference-instrumentation-langchain 0.1.80``, and OpenTelemetry ``1.45.1``.
The toolkit uses the OpenTelemetry HTTP exporter directly.
It does not call ``phoenix.otel.register``.
It does not claim support for every
historical Phoenix REST API or all Langfuse features.

The callback uses the upstream OpenInference LangChain callback implementation.
That dependency does not yet expose the callback through a public import.
The package range and SDK integration tests guard this boundary.
Test a dependency update before deploying it.

To repeat the live checks from a source checkout, start Docker and run:

.. code-block:: bash

   uv sync --frozen --no-install-project --extra all
   uv run --no-sync python scripts/ci/phoenix_stack.py prepare \
     --directory=data/phoenix-test
   uv run --no-sync python scripts/ci/phoenix_stack.py start \
     --directory=data/phoenix-test
   set -a
   source data/phoenix-test/server.env
   set +a
   uv run --no-sync pytest \
     tests/integration/test_phoenix_live.py tests/e2e/test_phoenix_api.py \
     --run-phoenix --run-e2e --no-cov

The helper starts the pinned server on a random loopback port.
It creates a unique Compose project and a dedicated persistent SQLite volume.
The generated test variables select only that disposable server.
The process tests verify that the server matches the managed stack before
they restart it. Do not substitute a normal deployment for this test stack.

The live prompt checks cover ``f-string`` and ``mustache`` creation, reads,
versions, tags, duplicate prevention, and deletion. They also test publication
from a text file through ``PromptManager``.
The process checks call the real API with native LangChain and Deep Agents
graphs. They use scripted model replies and real local tools.
They check stored trace parents, model and tool spans, prompt version links,
public session identity, feedback, and checkpoint history after API restart.
They restart Phoenix and read the same stored records again.
An additional process check uses a one-hour batch delay and sends no feedback.
It requires an empty server trace before API shutdown.
It then verifies the complete stored trace after normal shutdown.
This checks that the service drains an explicit agent provider's pending batch.
These checks validate the integration. They do not measure model quality,
external provider behavior, or maximum production throughput.
The disposable server uses local HTTP without authentication.
SDK wire tests check bearer headers and URL prefixes separately.
Verify the actual certificate trust, proxy paths, and access keys in the target
remote deployment before sending production traffic.

Run cleanup after the checks, including when a test fails:

.. code-block:: bash

   uv run --no-sync python scripts/ci/phoenix_stack.py stop \
     --directory=data/phoenix-test

Cleanup saves logs and removes the containers and volume that the helper owns.
For a manual persistence check, use ``restart`` before cleanup.
It keeps the same volume. ``stop`` deletes that test data.

Remote content stays in its original service when you change backends.
Phoenix trace ingestion is separate from prompt and feedback REST APIs.
A healthy agent response alone does not prove that all three paths work.
Verify stored traces, prompt reads, and feedback after a deployment.
