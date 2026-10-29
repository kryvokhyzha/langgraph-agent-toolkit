Langfuse compatibility
======================

Langfuse Server and the Python SDK have separate versions.
Select the SDK for the server in each client deployment.
The toolkit selects its adapter from the installed SDK version.
It does not change the server or its database.

Supported combinations
----------------------

.. list-table:: Deployment choices
   :header-rows: 1
   :widths: 25 25 50

   * - Langfuse Server
     - Python SDK
     - Use
   * - v2
     - 2.60.10
     - Legacy trace ingestion. SDK v3 and v4 cannot trace to server v2.
   * - v3 before 3.63.0
     - 2.60.10
     - Retain the legacy SDK until the server is upgraded.
   * - v3 at 3.63.0 or later
     - 4.18.0, 3.15.0, or 2.60.10
     - SDK v4 is the preferred choice. SDK v3 remains compatible.
       SDK v2 uses legacy trace ingestion.
   * - v4 before 4.59.0
     - 4.18.0
     - Current tracing, prompts, and feedback. SDK v5 is not supported.
   * - v4 at 4.59.0 or later
     - 5.0.4 or 4.18.0
     - SDK v5 is the preferred choice. SDK v4 remains supported.
   * - v4 with SDK v3
     - 3.15.0
     - Deprecated tracing support. Data can take up to 15 minutes to appear.
       Deprecated read APIs are unavailable.

SDK v2 trace ingestion does not work with the server v4 default
``events_only`` mode. Server v4 ``legacy`` and ``dual`` modes retain that
path during a migration. Do not use those modes as a permanent SDK v2
compatibility solution.

These server requirements come from the official
`compatibility matrix <https://langfuse.com/self-hosting/upgrade/versioning>`_
and `server migration guide
<https://langfuse.com/self-hosting/upgrade/upgrade-guides/upgrade-v3-to-v4>`_.

Installation
------------

The SDK checks cover Python 3.11 through 3.14. On Python 3.14, the toolkit
requires Pydantic 2.13.0 or later. Pydantic 2.12 can install on this interpreter,
but the legacy SDK models fail during import. Pydantic 2.13 restores support in
the ``pydantic.v1`` namespace. See the
`Pydantic 2.13 release notes <https://pydantic.dev/articles/pydantic-v2-13-release>`_.

Select the Python SDK generation through a published package extra:

.. list-table:: SDK selectors
   :header-rows: 1
   :widths: 40 60

   * - Extra
     - Python SDK requirement
   * - ``langfuse-v2``
     - ``langfuse>=2.60.10,<3``
   * - ``langfuse-v3``
     - ``langfuse>=3.15.0,<4``
   * - ``langfuse-v4``
     - ``langfuse>=4.18.0,<5``
   * - ``langfuse-v5``
     - ``langfuse>=5.0.4,<6``

These extras select the SDK. They do not install or change a Langfuse server.
Use the compatibility table above to choose the SDK for your server.
For server v2, add this dependency to the application:

.. code-block:: bash

   uv add 'langgraph-agent-toolkit[langfuse-v2]'

For a deployment that must retain SDK v3, use:

.. code-block:: bash

   uv add 'langgraph-agent-toolkit[langfuse-v3]'

For server v3 at 3.63.0 or later, or server v4, use:

.. code-block:: bash

   uv add 'langgraph-agent-toolkit[langfuse-v4]'

For server v4 at 4.59.0 or later, select SDK v5:

.. code-block:: bash

   uv add 'langgraph-agent-toolkit[langfuse-v5]'

Use only one version selector in an environment. Multiple selectors require
incompatible SDK versions. When switching SDK generations, remove the previous
selector from the application's dependency declaration and update its lockfile.
The canonical names use hyphens. Package tools normalize underscore spellings,
such as ``langfuse_v2``, to ``langfuse-v2``.

The generic ``langfuse`` extra now permits ``langfuse>=2.60.10,<6``. It can
combine with any one selector. ``all-observability`` still includes this generic
extra and ``langsmith``. The ``all`` extra includes ``all-observability`` and the
other optional features. It does not force a version selector. The default
lockfile selection uses SDK 5.0.4. This changes the minimum server version
for installations without a selector. SDK v5 needs server 4.59.0 or later.
Keep ``langfuse-v4`` or ``langfuse-v3`` for an existing server v3 deployment.

The SDK selectors are published in the wheel metadata. Applications can use
them without a source checkout. They are separate from local dependency groups,
such as ``tests`` or ``client``, which are not published package extras.

An exact SDK pin is optional. Use one to reproduce a tested patch version:

.. code-block:: bash

   uv add 'langgraph-agent-toolkit[langfuse-v2]' 'langfuse==2.60.10'

Keep the complete application lockfile. The selector permits updates within
one SDK generation; the lockfile fixes the installed version.

Source checkout migration
~~~~~~~~~~~~~~~~~~~~~~~~~

Replace ``uv sync --all-extras`` with ``uv sync --extra all``. The old command
selects all four incompatible SDK generations. The ``all`` extra preserves
the complete feature set without selecting those alternatives together:

.. code-block:: bash

   uv sync --frozen --no-install-project --extra all

To use the same features with SDK v2, add one selector:

.. code-block:: bash

   uv sync --frozen --no-install-project --extra all --extra langfuse-v2

Use ``langfuse-v3``, ``langfuse-v4``, or ``langfuse-v5`` in the same way. All selected dependencies
must have compatible requirements. See :doc:`testing` for isolated SDK checks.

Configuration
~~~~~~~~~~~~~

Set ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, and ``LANGFUSE_HOST``
for the selected deployment. The adapter also accepts
``LANGFUSE_BASE_URL`` as the current SDK URL alias.
An exported ``LANGFUSE_BASE_URL`` takes precedence over the validated
``LANGFUSE_HOST`` setting. Set only one URL variable in a deployment.
Validated toolkit settings can supply the keys and host without exporting
credentials to the process environment.

Adapter behavior
----------------

Each run has a separate callback handler.
This prevents mutable callback state from mixing concurrent traces.
SDK v2 uses a toolkit callback adapter because its built-in callback imports
modules that LangChain 1 removed. The adapter records chains, model calls,
tools, retrieval, errors, and reported token usage.
It also records the first token time and separate cached and reasoning token counts.
SDK v2 supports prompt creation, reads, and versions. Its custom callback does
not attach prompt versions to generation observations. The live tests require
that link only for SDK v3, v4, and v5.

The adapter passes a native mask callback to every supported SDK.
It replaces typed Pydantic ``SecretStr`` and ``SecretBytes`` values with
``<secret>`` before serialization. It handles nested Pydantic v1 and v2
models, dataclasses, dictionaries, lists, and tuples. The original caller objects remain
unchanged. This closes a plaintext serialization gap in SDK v2 and v3.
Plain strings are not recognized as secrets. Applications must keep credentials
out of prompts and other ordinary strings.

SDK v3, v4, and v5 use their built-in LangChain callbacks.
The adapter binds each callback to its configured public key.
It records the root input and output and propagates the user and session.
With SDK v3 and v4, it also keeps legacy trace I/O for server v3 evaluators.
SDK v4 marks this legacy I/O method as deprecated.
SDK v5 records input and output on the root observation. It does not use the
removed legacy trace setters. Its score helper sends batches to the current
scores endpoint. Its OpenTelemetry exporter uses gzip by default.
Proxies must accept compressed trace requests.
See the `Python SDK v3 to v4 guide
<https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4>`_.

Prompts, feedback, and traces reuse the same client.
Shutdown flushes a client only if the adapter initialized it.
It also stops the SDK v2 client threads that the adapter owns.
SDK v3, v4, and v5 resources remain available to other users of their shared client.
The service sets a deadline for this cleanup.
Prompt reads that fail with a connection or authorization error do not create
new prompt versions. Only a ``404`` means that a prompt is absent.
Prompt deletion raises ``NotImplementedError`` with SDK v2.
SDK v3, v4, and v5 call their prompt deletion API.

Verification and server upgrades
--------------------------------

The SDK contract tests in ``tests/integration/test_langfuse_sdk.py`` run against
the real SDKs 2.60.10, 3.15.0, 4.18.0, and 5.0.4.
They keep HTTP requests and OpenTelemetry exports in local memory.
They check prompt requests, feedback IDs, concurrent trace links, user and
session attributes, and output updates.
These tests verify the SDK contracts. They do not replace an ingestion check
against each deployed server.

Before a server v4 upgrade, complete the server v3 background migrations and
back up PostgreSQL and ClickHouse. The official guide requires ClickHouse
25.12 or later, PostgreSQL 15 or later, and Redis 7.0 or later.
Keep the server upgrade separate from this package upgrade.
The package does not migrate Langfuse infrastructure.

SDK v4 also changes the default OpenTelemetry span filter.
It can omit unrelated HTTP, database, and framework spans.
If an application needs those spans, configure its SDK export filter and
verify the resulting trace tree before rollout.

SDK v5 migration limits
-----------------------

SDK and server majors do not match. Python SDK v5 targets Langfuse Server v4.
Server v4.71.0 is the selected self-hosted release for this review.
A server v3 application must keep an explicit SDK v3 or v4 selector.

Application code that calls the SDK directly must update removed APIs.
``api.legacy`` and ``api.ingestion`` no longer exist in SDK v5.
Use observation reads instead of deprecated trace reads.
``api.scores.create`` accepts a list of score requests.
``api.scores.get_many`` replaces ``api.scores_v3.get_many_v3``.
The toolkit's feedback method keeps its existing request format.
The SDK performs its batch conversion.

See the `SDK v4 to v5 migration guide
<https://langfuse.com/docs/observability/sdk/upgrade-path/python-v4-to-v5>`_
and :doc:`langfuse_server_upgrade` before rollout.

Reviewed SDK releases
---------------------

The review starts from SDK 4.15.2. SDK v2 and v3 have no newer stable
release than 2.60.10 and 3.15.0. Version 5.0.0 has no published PyPI artifact.

* `4.15.3 <https://github.com/langfuse/langfuse-python/releases/tag/v4.15.3>`_: Clarify SDK version guidance. This release has no listed runtime change.
* `4.15.4 <https://github.com/langfuse/langfuse-python/releases/tag/v4.15.4>`_: Mask Pydantic secret values during serialization.
* `4.15.5 <https://github.com/langfuse/langfuse-python/releases/tag/v4.15.5>`_: Sanitize nested NaN and infinity values after container conversion.
* `4.15.6 <https://github.com/langfuse/langfuse-python/releases/tag/v4.15.6>`_: Record OpenAI reasoning effort and verbosity model parameters.
* `4.16.0 <https://github.com/langfuse/langfuse-python/releases/tag/v4.16.0>`_: Cap default OpenTelemetry export batches at 64 MiB.
* `4.17.0 <https://github.com/langfuse/langfuse-python/releases/tag/v4.17.0>`_: Add optional gzip compression for the default OTLP exporter.
* `4.18.0 <https://github.com/langfuse/langfuse-python/releases/tag/v4.18.0>`_: Correct metadata and dataset JSON Schema types. Share cursor pagination models.
* `5.0.1 <https://github.com/langfuse/langfuse-python/releases/tag/v5.0.1>`_: Publish the first SDK v5 artifact. Remove deprecated APIs. Require server 4.59.0 or later. Enable gzip export and current score batches.
* `5.0.2 <https://github.com/langfuse/langfuse-python/releases/tag/v5.0.2>`_: Improve score batch serialization and OpenTelemetry attribute checks.
* `5.0.3 <https://github.com/langfuse/langfuse-python/releases/tag/v5.0.3>`_: Flush buffered score batches reliably. Preserve caller experiment IDs and serializer options.
* `5.0.4 <https://github.com/langfuse/langfuse-python/releases/tag/v5.0.4>`_: Fix streaming memoization error cleanup. Handle repeated shutdown and flush calls. Correct the gevent runtime check.

Delivery limits
---------------

Trace and score helpers use SDK buffers. A successful toolkit feedback request
does not confirm that Langfuse stored the score. The service flushes initialized
clients during graceful shutdown, subject to its shutdown deadline.
A killed process can lose buffered telemetry. This buffer is separate from
LangGraph checkpoints. It is not a durable application outbox.
Read the stored observations and scores to verify ingestion.

Telemetry session identity
--------------------------

The REST service uses the public ``thread_id`` for the telemetry session.
It keeps checkpoint keys and conversation locks scoped to the owner and agent.
Each trace still carries ``user_id`` separately.
An embedded ``AgentExecutor`` caller can pass ``trace_session_id`` explicitly.
Without that argument, the executor uses its ``thread_id``.
Historical telemetry with a scoped internal session key is not rewritten.

Shutdown batching limits
-------------------------

The default ``LANGFUSE_FLUSH_INTERVAL`` is 5 seconds.
``OBSERVABILITY_SHUTDOWN_TIMEOUT`` defaults to 10 seconds.
SDK v2, v3, and v4 flush methods wait for the pending score consumer.
They cannot wake a long batching wait through their public flush API.
Keep that interval below the shutdown budget, with time for network requests.
SDK v5 wakes the score consumer when flushing.
The live API test verifies its shutdown behavior with a 3,600-second interval.
A network outage or forced process exit can still lose buffered telemetry.
