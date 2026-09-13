Connections and High Traffic
============================

The service bounds concurrent requests, connection pools, and network waits.
Tune these limits together. Larger pools cannot remove provider rate limits or
database capacity limits.

Choose Deployment Limits
-------------------------

The defaults provide bounded behavior. They do not guarantee capacity or latency
for every workload. Keep deployment configuration limited to required credentials,
backend choices, and deliberate overrides. The settings listed in this guide are
reference values; most do not need an explicit environment variable.

Tune these limits first:

.. list-table::
   :header-rows: 1
   :widths: 32 25 43

   * - Control
     - Default
     - Tuning input
   * - ``REQUEST_MAX_CONCURRENT`` and worker count
     - 8 active requests per worker
     - Peak concurrent runs, CPU, memory, model calls per run, and provider quotas.
   * - ``REQUEST_TIMEOUT`` and client/proxy timeouts
     - 300 seconds at the API
     - Longest valid run, time without streamed output, and cleanup time.
       Client read defaults are 60 seconds for invoke and 120 seconds for streams.
   * - PostgreSQL pool sizes
     - 20 checkpoint and 10 lock connections per worker
     - Available database connections across all workers, replicas, stores,
       and other applications. These are maximum sizes, not initial allocations.
   * - Request and conversation queues
     - 0 admission waiters; 32 waiters per conversation
     - Whether excess requests should receive a busy response or wait.
       Conversation waiters can occupy active HTTP request slots.
   * - ``REQUEST_MAX_BYTES``
     - 20 MiB per request
     - Actual text, file, and image sizes. Eight maximum-size bodies alone use
       160 MiB before parsing and graph-state overhead.

Total workers means workers per replica multiplied by replicas. At eight workers,
the default HTTP limit permits 64 active requests. The two default PostgreSQL
pools can use 240 connections before separately opened stores or other services.
Check this budget before increasing worker count. See :doc:`deployment`.

Keep TCP keepalive, connection recycling, pool maintenance, heartbeat, and HTTP
phase limits at their defaults until a measurement identifies a problem. These
controls remain useful for other providers, databases, and deployment sizes.
Some controls apply only on selected paths. ``REQUEST_QUEUE_TIMEOUT`` controls
admission waiting only when that queue is enabled. The aiohttp adapter does not
apply the HTTPX write timeout or maximum idle-connection count. MCP, telemetry,
and store controls apply only when those features are used.

Measure p95 response time, time to first token, memory, pool waits, checkpoint
duration, and 429/503 responses. Tune one constraint at a time. Load tests must
include bursts and slow requests, not only average traffic. See :doc:`load_testing`.

Checkpoint Persistence
----------------------

A successful model response and a durable checkpoint are separate events.
Configure the storage backend and graph before relying on saved history.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Configuration
     - Persistence boundary
   * - PostgreSQL
     - Saved checkpoints survive worker replacement. Database storage, backups,
       and failover configuration remain deployment responsibilities.
   * - SQLite file
     - Saved checkpoints survive process restarts when the same file remains
       available. Mount persistent storage when the API runs in a container.
       The default Compose files mount PostgreSQL data, not an API SQLite file.
   * - No ``MEMORY_BACKEND`` or an explicit ``MemorySaver``
     - Checkpoints stay in one worker's memory. Other workers cannot read them.
       Worker exit removes them.
   * - SQLite ``:memory:``
     - Checkpoints disappear when the connection closes.
   * - Explicit ``NoOpSaver`` or ``checkpointer=False``
     - The agent intentionally does not save checkpoints. This is not a database
       failure. Existing ``NoOpSaver`` behavior remains unchanged.

The service logs warnings for volatile storage. It retains
an explicit graph checkpointer. A configured PostgreSQL backend does not replace
an agent's explicit ``NoOpSaver`` or custom saver. Verify custom savers separately.

``CHECKPOINT_DURABILITY`` controls when graph execution waits for checkpoint
writes. It does not select a synchronous database driver. The executor defaults
to ``sync``::

   CHECKPOINT_DURABILITY=sync

With ``sync``, each super-step's checkpoint must finish before the next
super-step starts. This reduces the progress that a hard worker failure can
lose. The executor still uses async graph and database methods. The current
run waits with ``await``; other requests can run on the event loop during
this wait. This adds waiting time to the current run.

Choose a different mode when the application needs a different balance between
step latency and recovery:

* ``async`` lets the next super-step run while the previous checkpoint saves.
  This can reduce latency. A hard worker failure can lose pending progress.
  The graph still waits for pending saves before successful completion.
* ``exit`` saves at graph exit. A hard process failure during execution can lose
  progress since the preceding run.

No mode makes a running node or its external actions atomic. An existing
``CHECKPOINT_DURABILITY=async`` environment value keeps that mode after an
upgrade. Remove the override or set it to ``sync`` to use the new default.

These are LangGraph's `durability modes
<https://docs.langchain.com/oss/python/langgraph/checkpointers#durability-modes>`_. The setting
applies to ``AgentExecutor`` runs. Applications that call graphs directly must
select their own durability mode. It cannot make in-memory storage durable.
Both ``sync`` and ``async`` propagate checkpoint write errors. The main
difference is whether the next step can start before that write finishes.
An explicit ``NoOpSaver`` remains a valid choice for a stateless agent.

Tokens and full message events can arrive before the final checkpoint write.
Treat them as provisional until the stream finishes without an error. For SSE,
require ``[DONE]`` and no preceding error event. An HTTP 200 alone does not prove
that a streamed run completed. JSON Lines uses the response body end as its
terminator; it has no independent completion marker. Prefer SSE when explicit
completion detection is required.

For a persistent agent, invoke returns its successful result after graph execution
and checkpoint cleanup. A write error propagates to the caller. Unsupported streamed output
also fails visibly instead of being skipped. No success signal makes a disabled
or volatile saver durable.

Cancellation preserves completed checkpoints when the backend remains
available. A running node can remain unfinished. A hard worker termination can
stop any in-flight operation. Checkpoints do not resume accepted HTTP requests
automatically or provide exactly-once external tool actions. Use idempotency keys
and a durable job system for work that must continue after the client disconnects.

The history API returns messages, not a complete checkpoint backup. Other state
channels, pending tasks, and external tool data require their own backup or
recovery process. The toolkit's view-only trimming middleware changes model
context without deleting saved messages. Custom reducers, summarization, or
``RemoveMessage`` updates can deliberately replace saved state.

Sending a custom stream event does not save it in graph state. Return data in a
declared state channel or write it to a durable store when it must survive the
run. The background-task example emits progress events without saving those
events as history. A configured PostgreSQL checkpointer cannot save data that
the graph never puts in its state or pending writes.

Long-term store writes and checkpoints are separate operations. They do not
share a transaction. Coordinate changes to the same user-memory key across
threads. Use an application transaction or an outbox when separate systems must
record one business operation together. Telemetry is also separate: an unavailable
observability service or a bounded shutdown flush can lose pending traces.

After a client error, read saved history before retrying. The server can have
committed progress before the connection failed. With ``async`` durability, a
later step can run and save data before an earlier checkpoint failure reaches
the caller. A failed run does not roll back all of its writes. The included UI keeps the
failed input separate, reloads saved history, and does not automatically resend
the request. Browser session state is not a durable conversation archive. Keep
public thread IDs and ownership records when users must discover previous chats.

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

The default connection limits, keepalive expiry, phase timeouts, and retry count
match the OpenAI Python SDK 3.13.0 baseline. These toolkit defaults do not change
automatically when the SDK version changes. The settings below apply to each
endpoint and credential configuration in one worker:

.. code-block:: ini

   LLM_HTTP_MAX_CONNECTIONS=1000
   LLM_HTTP_MAX_KEEPALIVE_CONNECTIONS=100
   LLM_HTTP_KEEPALIVE_EXPIRY=5.0
   LLM_HTTP_CONNECT_TIMEOUT=5.0
   LLM_HTTP_READ_TIMEOUT=600.0
   LLM_HTTP_WRITE_TIMEOUT=600.0
   LLM_HTTP_POOL_TIMEOUT=600.0
   LLM_HTTP_MAX_RETRIES=2
   LLM_HTTP_MAX_POOLS=32
   LLM_HTTP_SHUTDOWN_TIMEOUT=10.0

Timeouts are seconds. Keepalive connections must not exceed total connections.
These limits are upper bounds, not connections opened at startup or a target
request rate. The separate eight-request admission default still applies.
Sync calls have a separate HTTPX pool with the same limits, even when async calls
use aiohttp. ``LLM_HTTP_MAX_POOLS`` limits distinct configurations. Reaching this
limit fails model construction; it does not evict a pool used by another run.
``LLM_HTTP_MAX_POOLS`` and ``LLM_HTTP_SHUTDOWN_TIMEOUT`` are toolkit resource
controls. Their defaults remain 32 configurations and 10 seconds.

Existing environment and model overrides retain their values. Remove old
``LLM_HTTP_*`` numeric overrides to adopt these defaults. Keep transport selection
and any deliberate deployment-specific limits. See :doc:`migration`.

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
timeouts apply to each attempt. The 600-second phase defaults do not change
``REQUEST_TIMEOUT``. That separate deadline starts cancellation of the whole
admitted HTTP request after 300 seconds by default. Deployments can set a shorter
deadline. Cleanup can continue after cancellation starts.

The tested ``langchain-openai`` 1.6.2 has a separate ``stream_chunk_timeout``.
Its default is 120 seconds between parsed chunks in an async model stream.
SSE keepalive comments do not reset that timer. The model option or
``LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S`` can override it. A 600-second HTTP
read timeout does not change this guard or ``AgentClient.stream_timeout``.

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

``POSTGRES_STATEMENT_TIMEOUT`` limits each SQL statement. It is not a model
generation deadline. ``POSTGRES_LOCK_TIMEOUT`` limits SQL lock acquisition;
``THREAD_QUEUE_TIMEOUT`` controls conversation-lock waiting. Normal model calls
do not hold an idle SQL transaction open. Set SQL timeouts from database work,
not model latency. The toolkit sends zero values explicitly so they disable
inherited database or role timeouts.

Pool-backed checkpoint operations no longer wait behind one saver-wide lock.
Coordinated operations still share their conversation session. Both checkpoint
and long-term store setup use a schema lock for concurrent worker startup. Lock
acquisition uses nonblocking polling; a waiting SQL snapshot must not block
concurrent index creation.

SQLite rolls back failed or cancelled writes before releasing its connection
lock. Checkpoint iterators for both databases release their cursor before
yielding, so a consumer can make another checkpoint query without deadlocking.
The iterators collect the selected results in memory. Set ``limit`` when calling
a saver directly on a large history.
SQLite still serializes writers. Prefer PostgreSQL for sustained concurrent
writes across workers or hosts.

Client-side cancellation can finish before PostgreSQL stops its server query.
Keep server statement timeouts configured and monitor active queries. Neither
database recovery nor HTTP cancellation guarantees that an external action did
not complete. The toolkit does not replay an uncertain write or an entire graph.
Use application idempotency keys for tools that cause external changes.

Memory Coordination and Library Responsibilities
------------------------------------------------

The toolkit delegates checkpoint SQL and serialization to LangGraph.
It delegates connection limits, replacement, recycling, and reconnect backoff
to ``psycopg_pool``. Its memory adapters cover the following gaps:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Additional behavior
   * - ``concurrency.py``
     - Serialize a complete conversation operation across workers. Bound
       acquisition time. Cancel active work after its PostgreSQL session fails.
   * - ``coordinated_saver.py``
     - Keep checkpoint I/O on the session that owns the conversation lock.
       Allow unrelated pool connections to work concurrently.
   * - ``pool.py``
     - Include health checks in the checkout deadline. Preserve cancellation
       during those checks. Close sessions with uncertain query results.
   * - ``coordinated_sqlite.py``
     - Finish rollback before another operation can reuse the connection.
   * - ``schema_lock.py`` and ``coordinated_store.py``
     - Coordinate schema setup when several workers start at the same time.

A checkpointer protects individual storage operations. It does not serialize
the complete read, model call, and write sequence for one conversation.
``asyncio.Lock`` supplies this protection within one worker. PostgreSQL
advisory locks or ``filelock.AsyncFileLock`` extend it across workers.
SQLite coordination requires the same database file and lock directory on a
filesystem with working OS file locks. It is not a distributed lock service.

The PostgreSQL session retains its advisory lock without a heartbeat.
The heartbeat detects connection loss during a long model or tool call.
The saver uses that same session, so an old run cannot write through a new
connection after another worker acquires the lock.

The conversation queue is bounded per worker. It is not globally FIFO or
durable. Active PostgreSQL runs and workers waiting for an advisory lock use
connections from ``POSTGRES_LOCK_POOL_SIZE``. This limit therefore affects
both execution capacity and waiting capacity.

Cancellation can arrive while cleanup is already running. The rollback and
pool-return paths wait for cleanup before releasing shared resources, even
after repeated cancellation. One ``asyncio.shield()`` call does not provide
that guarantee by itself.

The adapters use upstream hooks, including the saver ``_cursor()`` method and
the connection ``wait()`` method. Check these contracts during dependency
upgrades. Remove an adapter only after its fault tests pass with the upstream
implementation. In the reviewed ``psycopg_pool`` 3.3.1 implementation, the
health-check loop catches cancellation and can retry. It also does not bound
the running check with the checkout deadline.

LangGraph Agent Server provides `enqueue and reject policies
<https://docs.langchain.com/langsmith/double-texting>`_. These policies are not
part of the OSS LangGraph framework. Using Agent Server would change the
service deployment and API architecture.

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
