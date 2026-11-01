Background Execution
====================

Use optional background jobs when work must outlive its HTTP request. The API
stores a job and returns HTTP 202. A separate worker executes the graph and
saves its result. PostgreSQL stores the queue and job records. The toolkit
uses `Procrastinate <https://procrastinate.readthedocs.io/>`_ for worker
coordination, heartbeats, cancellation, and queue state transitions. A supervised
toolkit task recovers jobs from lost workers.

``BACKGROUND_ENABLED`` defaults to ``false``. Existing invoke, streaming,
history, authentication, and UI behavior stay unchanged. The new job routes
exist only when background execution is enabled. Ordinary ``/invoke`` calls
do not become queued jobs. Redis and RabbitMQ are not required.

.. _choose-execution-path:

Choose the execution path
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Path
     - Use it for
     - Response and execution lifetime
   * - ``POST /invoke``
     - Interactive requests that need one complete answer
     - The API executes the graph and returns one ``ChatMessage``. The client
       waits for the response.
   * - ``POST /stream``
     - Interactive chat that displays messages and tokens as they arrive
     - The API executes the graph while the HTTP connection carries SSE
       events. Token output depends on the agent and model.
   * - ``POST /jobs``
     - Work that must survive API restarts, run in separate workers, or wait
       for later approval
     - The API commits the job before returning HTTP 202. A separate worker
       executes it. Read status and saved results with ``GET /jobs/{job_id}``.
   * - ``agent_runtime()`` and ``executor.execute()``
     - Embedded applications and an existing application worker
     - The caller owns execution and recovery.

The three HTTP submission paths use the default agent. Add ``/{agent_id}``
before the path to select another agent. Use :doc:`usage` for invoke and
streaming examples. ``/stream/jsonl`` provides the same execution path with
JSON Lines instead of SSE.

Choose ``/invoke`` or ``/stream`` when the caller can keep the request open.
Both run in the API process. Waiting for an HTTP response does not mean that
the server uses blocking model or database calls. The asynchronous client method
``ainvoke()`` still waits for this HTTP result; it does not submit a background
job. A checkpoint can preserve completed steps, but it does not schedule
automatic recovery after an API process crash. A lost connection is
not proof that no tool action occurred. Do not blindly repeat the input.

Choose ``/jobs`` when execution must continue independently of the submitting
connection. It requires the ``background`` extra, PostgreSQL, an enabled jobs
API, and a separate worker. Use shared durable checkpoints for recovery of
unfinished graph execution. HTTP 202 confirms that the job was stored; it does
not confirm completion or worker availability. Jobs return saved status and
results, not a live token stream. ``/stream`` starts a new request; it does not
attach to a submitted job.

The request contracts differ. Invoke and streaming use the chat envelope, such
as ``{"input":{"message":"Hello"}}``. Jobs require an explicit ``thread_id``
and an ``Idempotency-Key`` header. Their ``input`` contains native graph input,
such as ``{"messages":[{"role":"user","content":"Hello"}]}`` for a chat
graph. See the submission example below before changing an existing client.

The application still owns event polling, schedules, approval policy, external
message delivery, and account mapping. Submit a stable operation key for each
source event. Keep source cursors and outbound delivery records in the
application database. A graph checkpoint is not an outbound message receipt.

Why use a durable queue
-----------------------

FastAPI's `BackgroundTasks
<https://fastapi.tiangolo.com/tutorial/background-tasks/>`_ runs functions
after sending an HTTP response. The functions run inside the API process.
It is useful for short, best-effort work, such as refreshing a disposable
cache when a missed update is acceptable.

``BackgroundTasks`` provides no persistent queue, automatic crash recovery,
submission deduplication, or job status API. An abrupt process termination can
lose pending work or interrupt work already in progress. Graceful shutdown
can wait for tasks, but it does not make them durable. Tasks attached to one
response run in order; an unhandled exception prevents later tasks from
running. See the `Starlette background task documentation
<https://starlette.dev/background/>`_.

Background tasks also share API capacity. Asynchronous functions use the API
event loop. Synchronous functions use its shared thread pool. Moving a slow
operation into ``BackgroundTasks`` does not isolate it from HTTP handling.
See the `Starlette thread pool documentation
<https://starlette.dev/threadpool/>`_.

The toolkit uses PostgreSQL and Procrastinate to store accepted jobs and run
them in separate workers with bounded execution concurrency. Submission saves
the job record and queue entry in one transaction before HTTP 202. Keep this
database write in the request handler. Deferring it with ``BackgroundTasks``
would allow the API to acknowledge a job before storage succeeds. A storage
failure after the response could no longer produce HTTP 503 for that request.

The durable queue supports duplicate-submission checks, saved results,
cancellation requests, and recovery after worker loss. Checkpoints preserve
graph progress; the queue schedules execution. These are separate roles.
Neither guarantees that an external tool action occurs exactly once. Give
external writes their own idempotency keys. The reliability limits below
describe recovery and cancellation boundaries.

Install and configure
---------------------

Install the ``background`` extra in both API and worker environments:

.. code-block:: bash

   uv pip install "langgraph-agent-toolkit[background]"

For a source checkout, use ``uv sync --frozen --no-install-project --extra
background``. Add the model and observability extras that the application uses.
The ``all`` extra includes background support.

Set the same agent paths, checkpoint database, and queue schema in every API
and background worker. This example uses the generic approval blueprint and
does not call a model:

.. code-block:: ini

   BACKGROUND_ENABLED=true
   BACKGROUND_POSTGRES_SCHEMA=lat_background
   BACKGROUND_POSTGRES_POOL_SIZE=5
   MEMORY_BACKEND=postgres
   POSTGRES_HOST=127.0.0.1
   POSTGRES_PORT=5432
   POSTGRES_DB=agents
   POSTGRES_USER=app_user
   POSTGRES_PASSWORD=replace-me
   POSTGRES_SCHEMA=checkpoints
   CHECKPOINT_DURABILITY=sync
   AGENT_PATHS=["langgraph_agent_toolkit.agents.blueprints.background_example.agent:agent"]
   DEFAULT_AGENT=background-example
   AUTH_SECRET=replace-with-a-private-shared-secret

Memory and jobs use the same ``POSTGRES_*`` connection settings by default.
They retain separate schemas and pools. Set ``BACKGROUND_POSTGRES_DSN`` only
when jobs need another database or connection identity. The queue uses the
shared PostgreSQL timeout, connection health, and reconnect controls. See
:doc:`environment_setup` for precedence and configuration aliases.

Create the checkpoint schema through the normal database setup. Initialize the
dedicated queue schema once with deployment credentials that can create tables:

.. code-block:: bash

   uv run --no-sync python -m langgraph_agent_toolkit.run_worker initialize

Run the API and worker as separate processes:

.. code-block:: bash

   uv run --no-sync python -m langgraph_agent_toolkit.run_api
   # In a separate terminal or supervised service:
   uv run --no-sync python -m langgraph_agent_toolkit.run_worker run

The initializer is idempotent for the current schema and Procrastinate version.
It also supports additive toolkit schema upgrades from versions 1 and 2 to 3.
These upgrades add recovery and active-conversation indexes. They preserve jobs and duplicate-submission
keys. Stop API and worker writers, back up the queue, then run ``initialize``
before restarting them. Index creation has bounded statement and lock waits.
For large databases, plan an explicit maintenance window.

Normal startup only checks the queue schema. It does not run queue migrations.
An unsupported version mismatch fails startup. Before upgrading Procrastinate,
review its schema migration instructions separately. Do not remove the version
record or recreate a queue that contains unfinished work.

Docker worker service
---------------------

The API Dockerfile accepts ``--build-arg INSTALL_BACKGROUND=true``. Its default
is ``false``. The same image can run the API or the worker. The optional Compose
overlay installs the extra and adds a worker service:

.. code-block:: bash

   docker compose -f docker-compose.yaml -f docker-compose.background.yaml build
   docker compose -f docker-compose.yaml -f docker-compose.background.yaml up -d --wait postgres
   docker compose -f docker-compose.yaml -f docker-compose.background.yaml run --rm \
     --no-deps lat-background-worker python -m langgraph_agent_toolkit.run_worker initialize
   docker compose -f docker-compose.yaml -f docker-compose.background.yaml up -d

Set the queue DSN host and ``POSTGRES_HOST`` to ``postgres`` inside this network.
The overlay uses the root ``.env`` and the existing optional Compose services.
Custom agents must be available in the image or through a mounted package.
The worker exposes no HTTP port. The API keeps its existing curl health probe.
When jobs are enabled, API readiness also checks the queue database connection.
The worker uses process supervision and queue heartbeats; it does not inherit
the API HTTP probe. Monitor pending job age and stalled-worker counts separately.

Submit, read, and cancel
------------------------

Use native graph input in ``input``. For a chat graph, this usually means
``{"messages":[{"role":"user","content":"Hello"}]}``. The existing
``/invoke`` message envelope still works on the existing chat endpoints.

.. code-block:: bash

   curl --fail-with-body http://localhost:8080/background-example/jobs \
     -H "Authorization: Bearer $AUTH_SECRET" \
     -H 'Content-Type: application/json' \
     -H 'Idempotency-Key: example-prepare-1' \
     -d '{"thread_id":"example-1","input":{"value":21}}'

The response includes ``job_id``, ``thread_id``, ``status``, attempt count,
timestamps, and a ``Location`` header. Submission and queue insertion commit in
one transaction. HTTP 202 confirms storage, not execution success or worker
availability. ``/jobs`` is the default-agent submission alias.

.. code-block:: bash

   curl --fail-with-body "http://localhost:8080/jobs/$JOB_ID" \
     -H "Authorization: Bearer $AUTH_SECRET"
   curl --fail-with-body -X POST "http://localhost:8080/jobs/$JOB_ID/cancel" \
     -H "Authorization: Bearer $AUTH_SECRET"

``user_id`` remains optional. A shared trusted token uses the service identity
when it is absent. If a trusted caller submits a different ``user_id``, it must
send that same value as a query parameter when reading or cancelling the job.
Token-bound users cannot select another owner. Another owner's job returns 404.
See :doc:`environment_setup` for authentication modes.

Reuse an ``Idempotency-Key`` only for an identical request to the same agent
and owner. A duplicate returns the existing job. A different request with that
key returns 409. A new key represents a new operation, even on the same thread.
Keep the key when retrying a submission whose HTTP response was lost.
Keys must contain 1–128 ASCII letters, digits, dots, underscores, colons, or
hyphens. Stored records retain keys until the operator removes those records.

Job statuses are ``queued``, ``running``, ``cancelling``, ``cancelled``,
``completed``, ``interrupted``, and ``failed``. An interrupted job has finished
its execution attempt and releases the worker slot. It waits for a new resume
job, not a worker held open until a person replies. Cancellation is cooperative.
A completed result can win a race with cancellation. Cancellation does not undo
external writes or checkpoints that already committed.
Synchronous tools can continue after cancellation or timeout. Give those tools
their own timeouts and idempotency controls. See :doc:`reliability`.

Explicit resume and recovery
----------------------------

An interrupted result contains ``checkpoint_id`` and ``interrupts``. Resume
requires that checkpoint and answers keyed by the pending interrupt IDs. Use
a new idempotency key for the answer:

.. code-block:: json

   {
     "operation": "resume",
     "thread_id": "example-1",
     "expected_checkpoint_id": "checkpoint-from-job-result",
     "resume": {"interrupt-id-from-job-result": true}
   }

The generic example completes with ``result: 42`` after approval. It returns
``result: null`` after rejection. A stale checkpoint or an incorrect set of
interrupt IDs fails the job without resuming the graph. The API acknowledges
storage before these graph-state checks run. A new input job cannot implicitly
approve a paused graph. These rules apply to the new execution contract; the
existing chat API retains its current resume behavior.

``operation="recover"`` with ``expected_checkpoint_id`` continues unfinished
work using LangGraph's ``input=None``. It does not append another input or answer
an approval. Inspect the saved state first. A completed state has nothing to
recover, and a pending interrupt requires ``resume`` instead.

Use the client without a polling thread or a long HTTP request:

.. code-block:: python

   from langgraph_agent_toolkit.client import AgentClient
   from langgraph_agent_toolkit.schema import ExecutionRequest

   async with AgentClient(
       base_url="http://localhost:8080", agent="background-example", get_info=False
   ) as client:
       job = await client.asubmit_job(
           ExecutionRequest(thread_id="example-1", input={"value": 21}),
           idempotency_key="example-prepare-1",
       )
       current = await client.aget_job(job.job_id)
       print(current.status)

The client also provides synchronous ``submit_job``, ``get_job``, and
``cancel_job`` methods. It does not automatically retry or poll submissions.
An application can poll with a bounded interval and its own overall deadline.

Run without HTTP or a queue
---------------------------

``agent_runtime()`` owns the same model transports, MCP sessions, observability,
checkpointer, and conversation coordinator as the API. Keep it open for the
worker lifetime and await all work before closing it:

.. code-block:: python

   from langgraph_agent_toolkit.core.runtime import agent_runtime
   from langgraph_agent_toolkit.schema import ExecutionRequest

   async def run_example():
       async with agent_runtime(
           "langgraph_agent_toolkit.agents.blueprints.background_example.agent:agent"
       ) as runtime:
           result = await runtime.executor.execute(
               "background-example",
               ExecutionRequest(thread_id="embedded-1", input={"value": 21}),
               owner_id="application-service",
           )
           snapshot = await runtime.executor.inspect_execution(
               "background-example", "embedded-1", owner_id="application-service"
           )
           return result, snapshot

This path does not require the ``background`` extra. The calling application
authenticates users and chooses the trusted ``owner_id``. Never accept an
arbitrary external owner without checking access. Explicit execution returns
JSON-compatible graph output rather than requiring a chat message.

Recovery guarantees and limits
------------------------------

* The queue recovers work from lost workers. The default worker heartbeat is
  10 seconds, the stalled threshold is 60 seconds, and recovery is scheduled
  every 10 seconds. Recovery runs outside execution slots and creates no queue
  maintenance records. Database pressure can still delay it. Recovery or final
  queue-acknowledgement failures stop the worker. Run it under a supervisor that
  replaces failed processes.
* Execution records its operation ID in checkpoint metadata. A recovered job
  continues its own unfinished checkpoint or returns its saved result. It does
  not add the original input again after finding that operation's checkpoint.
  If another operation changed the thread, recovery fails instead of guessing.
  Saved task writes can exist before their enclosing checkpoint commits.
  Recovery completes that step before it reports success, and reuses the saved
  writes instead of repeating completed nodes.
* Keep required node names and compatible state while unfinished work exists.
  Recovery rejects a missing pending node instead of reporting false completion.
  This check cannot detect changes to tool behavior or application semantics.
  Drain unfinished work with its original graph before an incompatible upgrade.
  Prerelease Functional API checkpoints can lack the scheduling metadata needed
  for this check. Keep their original entrypoint until unfinished work is drained.
* History clearing returns HTTP 409 while that conversation has queued, running,
  or cancelling background jobs. Wait for completion or cancellation before
  clearing. Do not delete checkpoints directly while recovery is possible.
  External pruning and database restores can remove operation markers and make
  duplicate detection unreliable.
* Configure shared durable checkpoints for cross-process recovery. The queue
  database alone does not make graph state durable. Explicit ``NoOpSaver`` and
  volatile savers remain valid for stateless work. A lost worker cannot safely
  recover their unsaved progress; an uncertain retry fails.
* Queued execution forces ``sync`` checkpoint durability. It awaits asynchronous
  checkpoint writes before the next step; it does not perform blocking database
  I/O. This makes the operation marker available before the first graph step.
  Ordinary HTTP and direct execution without an ``operation_id`` retain
  ``CHECKPOINT_DURABILITY`` behavior.
* Delivery and tool execution are not exactly once. A tool can commit an
  external action before its checkpoint commits. Give such actions their own
  idempotency keys. Do not put an irreversible action before ``interrupt()``
  in the same node, because LangGraph reruns that node on resume.
* Normal execution errors become failed jobs. They are not automatically
  replayed. ``BACKGROUND_MAX_ATTEMPTS`` bounds worker-loss recovery attempts.
  Inspect a failure and its checkpoints before submitting recovery work.
* Queued jobs for the same owner, agent, and thread serialize. The runtime also
  uses the existing conversation coordinator with ordinary API calls. This is
  not a global FIFO guarantee or a per-user rate limiter.
* There is no automatic retention policy, job-list endpoint, recurring business
  scheduler, or outbound delivery guarantee. Job payloads and results contain
  application data. Control database access, backups, and retention. Removing
  a record also removes its duplicate-submission protection. Do not independently
  prune queue rows while their toolkit job records still reference them.

Capacity and tuning
-------------------

The defaults allow four simultaneous executions per worker process and a
300-second execution deadline. The deadline includes waiting for the runtime's
conversation lock. It starts when a worker executes the job, not at submission.
Queue waiting can therefore exceed it. The API accepts a shorter
``timeout_seconds`` but rejects a value above the configured limit.

``BACKGROUND_POSTGRES_POOL_SIZE=5`` adds at most five queue connections per API or
worker process. The queue listener can need an additional connection. Existing
checkpoint, conversation-lock, long-term store, and model pools are separate.
Budget all processes together before increasing concurrency. Do not hold a
database transaction open while a model or tool runs.

There is no queue-length admission cap. Monitor queue depth, oldest pending
job age, failed jobs, worker heartbeats, and database capacity. Apply ingress
rate limits and scale workers within model-provider and database budgets.
API request admission still applies to submission requests. It does not count
background executions. See :doc:`reliability` for shared connection limits.

Four execution slots provide approximately 8 to 20 jobs per minute when jobs
take 12 to 30 seconds, before overhead. Sustained higher arrival rates grow the
backlog. Set ingress limits and alert on queue age. Execution deadlines do not
expire queued jobs. Choose an application retention policy before production.

Generated results have no toolkit size limit. Status reads include the saved
result. Return small graph outputs or artifact references for large documents.
Avoid frequent polling of large results. A synchronous tool can continue after
its awaiting task is cancelled; use tool-native deadlines for external calls.

Locks and database growth
-------------------------

Procrastinate claims jobs with PostgreSQL row locks. Toolkit recovery also uses
``SELECT ... FOR UPDATE SKIP LOCKED``. It locks the job record before the queue
record and rechecks eligibility inside that transaction. A busy row does not
hold up other recovery candidates. Lock and statement timeouts remain enabled.
These controls do not remove all possible database contention.

Recovery uses partial indexes for active job records and running queue records.
Completed history does not require a full-table scan on each recovery pass.
The candidate query remains sensitive to active backlog size. Each pass handles
at most 100 candidates. Keep autovacuum enabled and monitor table and index size.

Application records still need a retention policy. Recovery creates no new
maintenance rows. If a prerelease deployment used queued recovery tasks, its
old ``lat-maintenance`` rows no longer block recovery. After stopping old workers,
use the library's cleanup method for finished historical maintenance jobs:

.. code-block:: python

   from langgraph_agent_toolkit.core.background import queue_from_settings

   async def clean_maintenance_history():
       async with queue_from_settings() as queue:
           await queue.app.job_manager.delete_old_jobs(
               nb_hours=24,
               queue="lat-maintenance",
               include_failed=True,
               include_cancelled=True,
               include_aborted=True,
           )

Schedule this operation through the deployment's maintenance system. It removes
old maintenance jobs and their events. It does not remove application results
or idempotency records. See Procrastinate's
`finished-job cleanup guide <https://procrastinate.readthedocs.io/en/stable/howto/production/delete_finished_jobs.html>`_.

Choose application-job retention separately. It defines how long clients can
read results and safely retry their original submission keys. Archive required
results before deletion. Remove expired terminal toolkit records and their
terminal queue records in small transactions, in ledger-before-queue lock order.
Keep queued, running, cancelling, and approval-waiting records. Do not enable
automatic deletion on all queue jobs: toolkit records reference those rows.
