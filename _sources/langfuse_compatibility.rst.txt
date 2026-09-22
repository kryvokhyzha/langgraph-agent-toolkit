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
     - 4.15.2, 3.15.0, or 2.60.10
     - SDK v4 is the preferred choice. SDK v3 remains compatible.
       SDK v2 uses legacy trace ingestion.
   * - v4
     - 4.15.2
     - Current tracing, prompts, and feedback. SDK 4.7.0 or later gives
       immediate visibility in the new data model.
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
     - ``langfuse>=4.15.2,<5``

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

Use only one version selector in an environment. Multiple selectors require
incompatible SDK versions. When switching SDK generations, remove the previous
selector from the application's dependency declaration and update its lockfile.
The canonical names use hyphens. Package tools normalize underscore spellings,
such as ``langfuse_v2``, to ``langfuse-v2``.

The generic ``langfuse`` extra still permits ``langfuse>=2.60.10,<5``. It can
combine with any one selector. ``all-observability`` still includes this generic
extra and ``langsmith``. The ``all`` extra includes ``all-observability`` and the
other optional features. It does not force a version selector. The default
lockfile selection uses SDK 4.15.2.

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
selects all three incompatible SDK generations. The ``all`` extra preserves
the complete feature set without selecting those alternatives together:

.. code-block:: bash

   uv sync --frozen --no-install-project --extra all

To use the same features with SDK v2, add one selector:

.. code-block:: bash

   uv sync --frozen --no-install-project --extra all --extra langfuse-v2

Use ``langfuse-v3`` or ``langfuse-v4`` in the same way. All selected dependencies
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

SDK v3 and v4 use their built-in LangChain callbacks.
The adapter binds each callback to its configured public key.
It records the root input and output and propagates the user and session.
It also keeps legacy trace I/O for server v3 evaluators.
SDK v4 marks this legacy I/O method as deprecated.
The SDK dependency excludes v5 until that contract is checked.
See the `Python SDK v3 to v4 guide
<https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4>`_.

Prompts, feedback, and traces reuse the same client.
Shutdown flushes a client only if the adapter initialized it.
It also stops the SDK v2 client threads that the adapter owns.
SDK v3 and v4 resources remain available to other users of their shared client.
The service sets a deadline for this cleanup.
Prompt reads that fail with a connection or authorization error do not create
new prompt versions. Only a ``404`` means that a prompt is absent.
Prompt deletion raises ``NotImplementedError`` with SDK v2.
SDK v3 and v4 call their prompt deletion API.

Verification and server upgrades
--------------------------------

The SDK contract tests in ``tests/integration/test_langfuse_sdk.py`` run against
the real SDKs 2.60.10, 3.15.0, and 4.15.2.
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
