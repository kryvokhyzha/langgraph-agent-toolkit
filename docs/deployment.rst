Deployment and Recovery
=======================

Run the service under a process manager. Configure the health probes and a
persistent checkpoint backend before increasing the number of workers.
See :doc:`reliability` for request admission, LLM pools, and database fault handling.

API Image and Model Transport
-----------------------------

The API Dockerfile installs ``openai-aiohttp`` and sets
``LLM_HTTP_ASYNC_TRANSPORT=aiohttp``. This selects aiohttp for managed async
OpenAI and Azure calls without a runtime override:

.. code-block:: bash

   docker build -f docker/api/Dockerfile -t toolkit-api .
   docker run --rm -p 8080:8080 --env-file .env toolkit-api

Rebuild older images and recreate the API container to use the new default.
Docker and Compose environment overrides take precedence over the image. If an
existing ``.env`` sets ``LLM_HTTP_ASYNC_TRANSPORT=httpx``, remove that line or
change it to ``aiohttp``. The example ``.env`` leaves the selection unset.
The toolkit's ``LANGGRAPH_LLM_HTTP_ASYNC_TRANSPORT`` override also takes
precedence when configured.
To select HTTPX, set ``LLM_HTTP_ASYNC_TRANSPORT=httpx`` and recreate the container.
Changing this override does not need an image rebuild.

The choice favors connection reuse for streamed model responses. In the tested
SDK, aiohttp reused streaming connections while HTTPX closed them at the final
stream marker. Both passed local fault and cleanup checks. The shared-host load
tests do not establish a general latency or throughput ranking. See
:doc:`load_test_results` for the measured behavior and limits.

Synchronous model calls still use HTTPX. Caller-supplied model clients keep
their own transport. Python installations outside this image retain HTTPX as
the default. See :doc:`reliability` for timeout differences and limits.

Startup and Worker Supervision
------------------------------

Service startup must complete before a worker serves requests. A failure to
load an agent or initialize its checkpoint backend now fails startup. The
worker does not remain alive with an unusable executor.

Use Uvicorn's import-string factory for multiple workers:

.. code-block:: bash

   uv run --no-sync uvicorn langgraph_agent_toolkit.service.handler:create_app \
     --factory --host 0.0.0.0 --port 8080 --workers 2 \
     --timeout-worker-healthcheck 10

Configure ``AGENT_PATHS`` in the environment for this entry point. Supply the
credentials required by the selected model and authentication mode. Settings are
process-wide: configure one service per worker before startup. Do not change
``ServiceRunner(custom_settings=...)`` per request or use it to host services
with different settings in the same process. The
``run_api.py`` entry point also uses the configured agent paths.

``ServiceRunner.run_uvicorn`` defaults ``timeout_worker_healthcheck`` to 10
seconds. Factory imports are deferred until the application is created to reduce
startup work. Spawned workers can import launcher modules before starting their
heartbeat thread. If these imports need more time, supply a larger value, such as
``ServiceRunner().run_uvicorn(workers=2, timeout_worker_healthcheck=30)``.
An explicit value replaces the default. Direct ``uvicorn`` commands use
Uvicorn's own default unless this option is supplied.

If a worker stops during a heartbeat check, detection can wait for the full
heartbeat timeout. Replacement also needs time for the supervisor's next check
and the new worker's startup. Checks for multiple unresponsive workers run in
sequence, so their waits can add. The heartbeat timeout is not a limit on total
recovery time.

Uvicorn's heartbeat is not an HTTP health check. Its separate thread can answer
while the application's event loop is blocked. Configure an external HTTP
health probe and a supervisor that restarts the affected process or container
when that probe fails.
Prefer one worker per container when the orchestrator probes a shared port.
With multiple workers, a healthy worker can answer a probe for a stalled worker.

Uvicorn and Gunicorn supervise their worker processes. Local subprocess tests
started two workers, terminated one worker with ``SIGKILL``, and verified that
a replacement worker served requests. This verifies recovery from a terminated
worker. It does not verify every deployment failure or detect every worker
that remains alive but cannot serve requests.

An in-flight request can fail when its worker stops. The process manager does
not replay that request. Set graceful shutdown periods long enough for the
expected workload. Use a service manager or container orchestrator to restart
the parent process if it exits.

Gunicorn's ``timeout`` measures worker silence for the async worker class. Use
``REQUEST_TIMEOUT`` to limit one API request. ``graceful_timeout`` limits the
time allowed to drain requests after a restart signal. Workers that remain
after that limit can be stopped forcibly. See the
`Gunicorn settings <https://gunicorn.org/reference/settings/>`_.

The toolkit image currently sets ``graceful_timeout`` to 30 seconds, while the
API request limit defaults to 300 seconds. This does not let every permitted
request finish during deployment. Set the worker drain period from the maximum
supported request duration plus cleanup time. Set the container termination
grace period above that drain period. A Docker health check alone does not set
this budget. Verify the chosen limits during a rolling deployment.

Use the probe endpoints for their separate purposes:

- ``/health/startup`` confirms that initialization has completed.
- ``/health/ready`` checks initialization and live checkpoint/lock database connections.
  It returns 503 during a database outage and recovers when connections work again.
- ``/health/live`` checks whether the worker can respond and cancellation cleanup
  is progressing. Stalled cleanup returns 503 for supervisor recovery.
- ``/health/db`` reports checkpoint-pool state.

A listening port or a response from ``/health`` alone does not prove that agents
are ready. Configure a request timeout at the ingress and monitor worker exit,
startup failure, request failure, and database connection metrics.

PostgreSQL Connection Budget
----------------------------

Each worker creates its own checkpoint pool and conversation-lock pool. With
the default settings, one worker can use up to 20 checkpoint connections and
10 lock connections. A service with four workers can therefore use up to
120 connections before other applications or separately created stores are
counted.

Set these limits together:

- ``POSTGRES_POOL_SIZE``: maximum checkpoint connections per worker; default 20.
- ``POSTGRES_LOCK_POOL_SIZE``: maximum lock connections per worker; default 10.
- ``POSTGRES_POOL_MAX_WAITING``: maximum queued pool requests; default 100.

Leave database capacity for administration, monitoring, migrations, and other
services. Increasing the worker count does not increase PostgreSQL's connection
limit. It also multiplies model concurrency and each worker's local queue
capacity.

Conversation Ordering
---------------------

Operations on one conversation wait for a lock. The lock covers invocation,
streaming, and history changes. Different conversations can run concurrently.

These locks protect short-term checkpoint state. Two conversations for the same
``user_id`` can still run concurrently and access the same long-term store.
Use atomic writes or a separate conflict strategy when custom memory logic reads
and then replaces a shared user record. A thread lock does not protect that update
from another thread. Clearing a thread does not clear the user's long-term store.

The local queue is bounded. ``THREAD_QUEUE_MAX_WAITERS`` defaults to 32 waiting
operations for one conversation in each worker. ``THREAD_QUEUE_TIMEOUT`` limits
lock waiting and defaults to 60 seconds. Queue admission or wait failures return
an error so the caller can retry. These are per-worker admission limits, not a
single durable queue shared by all replicas.

PostgreSQL uses session advisory locks to coordinate workers. A dedicated pool
holds these sessions. While a conversation lock is held, checkpoint SQL uses
the same database session as that lock. It does not switch to another pooled
connection after losing the lock. Checkpoint operations outside a locked run
use the normal checkpoint pool.

A heartbeat checks the lock connection. If that session
is lost, the service cancels the operation instead of continuing without its
lock. Cancellation, failure, and stream closure release lock resources.

The service injects its checkpointer only when a graph has no explicit checkpointer.
Built-in agents leave this choice to the service. An explicit custom checkpointer
keeps its own persistence behavior. Custom savers for another database need
matching coordination; the service does not redirect them to its default database.
With no persistent backend, histories remain local to one process and are lost
on restart. Do not use that mode for shared multi-worker conversations.

SQLite uses operating-system file locks. Workers must share the same database
file and the same local filesystem. Do not use separate container-local SQLite
files for one shared service. Network filesystems can have different locking
semantics; use PostgreSQL for deployments across hosts.

Failure and Retry Boundaries
----------------------------

Checkpoint persistence and locks do not provide exactly-once tool execution.
A tool can complete an external action before a worker loses its connection or
stops. A retry can repeat that action. Give external actions their own
idempotency keys when repetition is unacceptable.

Database failover can interrupt a checkpoint write or release an advisory lock.
After recovery, a caller can retry from the available checkpoint, but the
service does not automatically replay incomplete HTTP requests. Review the
application's recovery rules for interrupted runs and external side effects.

No Redis queue or background job broker is added by this change. Requests still
run in the worker that accepts them. Use a durable job system if work must
survive the loss of that worker independently of a connected client.

Nested Calls and Shutdown
-------------------------

Do not call a coordinated executor or history operation from inside another
active coordinated operation. The inner call can wait for its own parent or
form a cycle with another thread. These calls now raise ``NestedConversationError``
immediately. The HTTP API returns 409. Retrying the same nested call does not
resolve the programming error. Compose agents with LangGraph subgraphs instead;
subgraphs use the enclosing run's checkpoint session.

PostgreSQL checkpoint-history iteration releases its cursor before yielding
results. This permits another checkpoint query while consuming the iterator.
An ambiguous or cancelled advisory-lock acquisition discards its database
connection. The pool cannot reuse a session with an unknown lock state.

``OBSERVABILITY_SHUTDOWN_TIMEOUT`` limits the wait for a synchronous telemetry
flush. Its default is 10 seconds. If the deadline expires, shutdown continues
and pending telemetry can be lost. A daemon thread performs the flush so a stuck
SDK cannot make the event loop wait for its default executor during worker exit.
Set the process manager's graceful shutdown period above this deadline and the
expected database cleanup time.

Client and Azure Resource Cleanup
---------------------------------

Use ``with AgentClient(...)`` for owned synchronous connections. Use
``async with AgentClient(...)`` or ``await client.aclose()`` for asynchronous
connections. Close the async client before its event loop stops. A client pool
must not move between event loops. The caller retains ownership of injected
HTTPX clients.

The Azure Functions runner uses the Azure SDK ASGI adapter. It starts the
application lifespan before serving requests. Call ``await runner.aclose()``
before stopping that runner's event loop so the application can release its
checkpoint and observability resources.
