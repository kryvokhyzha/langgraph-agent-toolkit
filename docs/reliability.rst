Connections and High Traffic
============================

The service bounds concurrent requests, connection pools, and network waits.
Tune these limits together. Larger pools cannot remove provider rate limits or
database capacity limits.

Request Admission
-----------------

Each worker accepts eight active HTTP requests by default. The limit covers body
buffering, conversation-lock waiting, graph execution, response sending, and
cleanup. It applies to invocation, SSE, JSON Lines, history, and feedback.
Health probes remain accessible during overload.
Probe requests must not contain a body.

.. code-block:: ini

   REQUEST_MAX_CONCURRENT=8
   REQUEST_QUEUE_MAX_WAITERS=0
   REQUEST_QUEUE_TIMEOUT=1.0
   REQUEST_TIMEOUT=300
   REQUEST_CLEANUP_TIMEOUT=10.0
   RESPONSE_SEND_TIMEOUT=30.0

When capacity is full, the service returns HTTP 503 with
``error_code=service_busy`` and ``Retry-After: 1``. Rejection occurs before body
buffering and before streaming response headers. No agent run starts for that
rejected request. Set a small positive ``REQUEST_QUEUE_MAX_WAITERS`` to absorb
short bursts. Waiters use no model or database connection. A disconnected waiter
can occupy its queue slot until admission or ``REQUEST_QUEUE_TIMEOUT``.

The queue exists in one worker. It is not a persistent job queue. A worker loss
does not transfer accepted work to another worker. Use a durable job system for
work that must continue without the original HTTP client.

The conversation queue is a separate limit. Calls for the same thread can occupy
active request slots while waiting for that thread. Keep
``THREAD_QUEUE_MAX_WAITERS`` small, or set it to zero, if one busy conversation
must not consume most of a worker's capacity. Admission does not guarantee equal
capacity for each user.

``REQUEST_TIMEOUT`` covers the admitted HTTP request. A deadline before response
headers returns 504. After headers, an error can only end the stream or close the
connection. The service does not send a second HTTP status. A client disconnect
cancels the active graph. Each response send has its own deadline so a client
that stops reading cannot hold the stream indefinitely.

Graph cleanup completes before its conversation lock and request slot are
released. Direct ``AgentExecutor`` calls keep their execution deadline. HTTP
calls use one deadline owner to avoid cancelling graph cleanup twice.

Custom tools must cooperate with cancellation and bound their own external I/O.
If cancellation cleanup exceeds ``REQUEST_CLEANUP_TIMEOUT``, both
``/health/live`` and ``/health/ready`` return 503 until cleanup finishes. The
service retains the occupied slot and lock while work is still active. Configure
the supervisor to replace a worker whose liveness probe keeps failing. The
library does not kill the host process. Prefer one worker per container when
the orchestrator probes and replaces containers; a shared multi-worker listener
can send a probe to a different, healthy worker. See :doc:`deployment`.

Feedback uses a separate bounded set of daemon threads. Its capacity equals
``REQUEST_MAX_CONCURRENT``. A cancelled HTTP caller does not free the thread slot
until the synchronous feedback call finishes. Further feedback requests fail
with 503 when all thread slots are occupied. Shutdown waits up to
``OBSERVABILITY_SHUTDOWN_TIMEOUT`` for feedback, then the same limit for telemetry
flush. Configure the process shutdown budget for both waits and resource cleanup.

Managed OpenAI and Azure Connections
------------------------------------

The service owns HTTP clients for OpenAI and Azure chat and embedding models
created through the toolkit factories. It binds the clients during agent startup
and each request, then closes them at shutdown. Matching endpoint and credential
configurations reuse connections. Different credentials, headers, proxies, and
endpoints receive separate pools. Caller-supplied clients retain their ownership.

The API Docker image selects the OpenAI SDK's ``DefaultAioHttpClient`` by
default. Python installations retain HTTPX as their default. To select aiohttp
outside the image:

.. code-block:: bash

   uv pip install "langgraph-agent-toolkit[openai-aiohttp]"

.. code-block:: ini

   LLM_HTTP_ASYNC_TRANSPORT=aiohttp

The API Docker image, ``all-llms``, and ``all`` include the aiohttp dependency.
Installing the dependency alone does not select the transport; the image sets
its default in the runtime environment. Set ``LLM_HTTP_ASYNC_TRANSPORT=httpx``
to override the image default. OpenAI documents aiohttp as an
alternative for high concurrency in its
`Python SDK reference <https://developers.openai.com/api/reference/python>`_.
Measure both transports with your workload before selecting one for latency.

These settings apply to each endpoint and credential configuration in one worker:

.. code-block:: ini

   LLM_HTTP_MAX_CONNECTIONS=100
   LLM_HTTP_MAX_KEEPALIVE_CONNECTIONS=20
   LLM_HTTP_KEEPALIVE_EXPIRY=30.0
   LLM_HTTP_CONNECT_TIMEOUT=10.0
   LLM_HTTP_READ_TIMEOUT=120.0
   LLM_HTTP_WRITE_TIMEOUT=30.0
   LLM_HTTP_POOL_TIMEOUT=10.0
   LLM_HTTP_MAX_RETRIES=2
   LLM_HTTP_MAX_POOLS=32
   LLM_HTTP_SHUTDOWN_TIMEOUT=10.0

Timeouts are seconds. Keepalive connections must not exceed total connections.
Sync calls have a separate HTTPX pool with the same limits, even when async calls
use aiohttp. ``LLM_HTTP_MAX_POOLS`` limits distinct configurations. Reaching this
limit fails model construction; it does not evict a pool used by another run.

Explicit model ``timeout`` and ``max_retries`` values override the managed
defaults. Caller-supplied HTTP/SDK clients or ``http_socket_options`` bypass
managed injection. With LangChain, the asynchronous HTTP client parameter is
``http_async_client``; the OpenAI ``AsyncOpenAI`` SDK calls it ``http_client``.
Do not pass an aiohttp async client as LangChain's synchronous ``http_client``.

The tested OpenAI 3.13.0 aiohttp adapter has these differences from HTTPX:

- It does not enforce a separate write timeout or an idle connection count.
- Its pool/connect phase combines pool waiting and connection establishment.
- It does not use LangChain's HTTPX TCP socket options.
- A dropped server socket can appear as a timeout error instead of a connection
  error. Both transports still obey the configured retry count.

Read timeouts measure inactivity, not the total generation duration. Phase
timeouts apply to each attempt. The HTTP request deadline bounds the whole run.
The OpenAI SDK normally retries connection failures, 408, 409, 429, and server
errors, with two retries by default. Increasing retries increases time and
provider traffic. Avoid adding another graph-level retry around the same call.
A stream that already delivered tokens is not automatically replayed.

Standalone applications can use the same ownership scope:

.. code-block:: python

   from langgraph_agent_toolkit.core.models import (
       CompletionModelFactory, LLMTransportManager,
   )
   from langgraph_agent_toolkit.core.settings import settings

   async def answer():
       async with LLMTransportManager.from_settings(settings) as manager:
           with manager.bind():
               model = CompletionModelFactory.create(
                   "openai", settings.OPENAI_MODEL_NAME,
                   max_retries=1, timeout=45,
               )
               return await model.ainvoke("Hello")

Use one manager for the application lifetime, not one for every request. The
manager belongs to one event loop. Configurable factory models created before
startup resolve managed clients during invocation. Concrete custom models
created before startup retain their original clients. Service startup calls
each registered ``graph_factory`` to build a fresh graph with current resources.
Factories selected for MCP receive their discovered tools in the same build.
Graphs without a factory keep their explicit dependencies. Other providers and
direct SDK calls keep their own connection configuration. This change does not
impose connection limits on arbitrary tools or external model objects.

Database Fault Handling
-----------------------

PostgreSQL pool checkout includes a bounded health check. Failed connections
are closed and replaced before ownership passes to a caller. Cancellation during
a query closes the uncertain session. A checkpoint run cannot reconnect and
continue writing after it loses its conversation lock.

.. code-block:: ini

   POSTGRES_CONNECT_TIMEOUT=10
   POSTGRES_HEALTH_CHECK_TIMEOUT=5.0
   POSTGRES_KEEPALIVES_IDLE=30
   POSTGRES_KEEPALIVES_INTERVAL=10
   POSTGRES_KEEPALIVES_COUNT=3
   POSTGRES_TCP_USER_TIMEOUT=60000

Connection and health-check timeouts are seconds. ``POSTGRES_TCP_USER_TIMEOUT``
is milliseconds. TCP keepalive options depend on the operating system and do not
apply to Unix sockets. ``connect_timeout`` applies separately to each host or IP
attempt. See the
`libpq connection parameters <https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-PARAMKEYWORDS>`_.

Keep ``POSTGRES_POOL_TIMEOUT``, ``POSTGRES_RECONNECT_TIMEOUT``, and
``POSTGRES_MAX_LIFETIME`` positive. ``POSTGRES_MAX_IDLE`` is idle seconds before
excess connections close, not a connection count. Zero lifetime does not disable
recycling. Statement, SQL-lock, and idle-transaction timeouts use milliseconds;
zero disables those server timeouts.

Pool-backed checkpoint operations no longer wait behind one saver-wide lock.
Coordinated operations still share their conversation session. Both checkpoint
and long-term store setup use a schema lock for concurrent worker startup. Lock
acquisition uses nonblocking polling; a waiting SQL snapshot must not block
concurrent index creation.

SQLite rolls back failed or cancelled writes before releasing its connection
lock. Checkpoint iterators for both databases release their cursor before
yielding, so a consumer can make another checkpoint query without deadlocking.
SQLite still serializes writers. Prefer PostgreSQL for sustained concurrent
writes across workers or hosts.

Client-side cancellation can finish before PostgreSQL stops its server query.
Keep server statement timeouts configured and monitor active queries. Neither
database recovery nor HTTP cancellation guarantees that an external action did
not complete. The toolkit does not replay an uncertain write or an entire graph.
Use application idempotency keys for tools that cause external changes.

Capacity Planning and Verification
----------------------------------

Start with the default admission limit below the lock-pool size: eight active
requests and ten lock connections per worker leave headroom for health checks.
Keep headroom when changing either limit. Readiness can return 503 for pool
saturation as well as a database outage; liveness does not query the database.

For four workers, the default request limit permits 32 active HTTP requests.
The database upper bound is 4 * (20 checkpoint + 10 lock) = 120 connections,
plus separately created stores and other services. Parallel graph branches can
make several model calls per HTTP request. Set graph concurrency and provider
quotas as well as HTTP pool limits. LangGraph's internal token queue is not
strictly bounded by these settings; send deadlines and cancellation stop a slow
consumer from keeping its producer alive indefinitely.

Align caller and ingress timeouts with the intended workload. ``AgentClient``
defaults to a 60-second normal read timeout and a 120-second stream inactivity
timeout. Set its ``timeout`` and ``stream_timeout`` arguments for longer work.
An earlier client or proxy timeout can disconnect and cancel the server request
before ``REQUEST_TIMEOUT`` expires.

Set the caller's idle connection expiry below the server or proxy keepalive
timeout, with room for scheduler and network delays. Equal idle deadlines can
race: the server closes a connection while the client is about to reuse it.
For example, with a five-second server keepalive, a caller can supply an
``httpx.AsyncClient(limits=httpx.Limits(keepalive_expiry=1.0))`` through
``AgentClient(async_http_client=...)`` and close that client at shutdown.
Use the same comparison for ``LLM_HTTP_KEEPALIVE_EXPIRY`` when an upstream idle
timeout is known. An expiry margin reduces this race; it cannot prevent every
connection failure. Do not replay an agent POST automatically after an ambiguous
transport failure, because its work may already have started.

Measure admitted throughput, first-token and completion latency, 429/503 rates,
retries, pool waiting, database locks, process memory, and file descriptors.
Increase limits gradually against representative prompts and tools. Local tests
verify reuse, bounded work, fault recovery, and cleanup. They do not establish a
production throughput number or prove that aiohttp is faster for your provider.
See :doc:`testing` for the fault tests and process journeys.
