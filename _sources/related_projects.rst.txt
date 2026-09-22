Related projects and source review
==================================

Reviewed on 2026-09-13. The review compares source code and selected regression
cases. It does not compare production latency, throughput, or model quality.
The package and the projects below have different deployment goals.

Projects reviewed
-----------------

* `Agent Service Toolkit
  <https://github.com/JoshuaC215/agent-service-toolkit/tree/18d76e919b357e448f60b97c50e704db79456096>`_
  is a complete application template. Its current source has AG-UI support,
  voice interaction, conversation navigation, and an operator model allowlist.
  This package provides more explicit controls for database connections,
  admission limits, cancellation, conversation isolation, and Langfuse versions.
  Those controls do not establish a performance advantage without comparable
  load tests. The reviewed commit is ``18d76e919b357e448f60b97c50e704db79456096``.
* `LangServe <https://github.com/langchain-ai/langserve>`_ is a Runnable HTTP
  adapter. Its schema discovery, typed client errors, and feedback tokens are
  useful design references. The project is deprecated and was archived on
  2026-05-05. This review does not recommend adding it as a dependency.
  The reviewed commit is ``27e57afeda13007a7f4e007c5d1f5e8489963aa4``.
* `Chat LangChain
  <https://github.com/langchain-ai/chat-langchain/tree/f90060d55de772f543564696b56ffee7be3945b8>`_
  is a documentation application. Its current design uses Managed Deep Agents,
  managed identity and checkpoints, MCP documentation tools, and a Next.js UI.
  Its search, page-reading, and citation patterns fit a specialized agent.
  They do not require replacing this package's FastAPI service or Streamlit UI.
  The reviewed commit is ``f90060d55de772f543564696b56ffee7be3945b8``.

Requested Agent Service Toolkit commits
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Commit
     - Change
     - Decision for this package
   * - `0aaf73f <https://github.com/JoshuaC215/agent-service-toolkit/commit/0aaf73f72a7982128d4f5478f6cc893fded7a56f>`_
     - Add thread listing and Previous Chats.
     - Useful next feature. Build an authenticated conversation catalog with
       public thread IDs and update timestamps. Do not copy checkpoint scans.
   * - `54d5e89 <https://github.com/JoshuaC215/agent-service-toolkit/commit/54d5e89b6717f41c67b024c730c1512250404823>`_
     - Check stored thread ownership.
     - Do not port. The check was ineffective and upstream reverted it.
       Keep this package's authenticated storage identity.
   * - `5544508 <https://github.com/JoshuaC215/agent-service-toolkit/commit/5544508a16bdd867b2b9f9cbdf7edc1a9174d1aa>`_
     - Restore the URL-selected agent and improve resume links.
     - Adapted. Validate the agent before loading history. Update the URL for
       new chats and agent changes. Keep user identity out of the link.
   * - `cfa9e16 <https://github.com/JoshuaC215/agent-service-toolkit/commit/cfa9e16bc06aef73ed7c4ad2fa436fbfd607e544>`_
     - Make history requests agent-aware.
     - Already supported by the API and Python client. Corrected the UI call
       order. Keep the existing GET interface and history pagination.
   * - `c69b888 <https://github.com/JoshuaC215/agent-service-toolkit/commit/c69b888a500bd68550c1513d3bad19c748a943fe>`_
     - Use the public Streamlit URL API.
     - Adapted. Use ``st.context.url``, preserve the path and scheme, and encode
       query values. Report an unavailable URL without inspecting other sessions.
   * - `186b491 <https://github.com/JoshuaC215/agent-service-toolkit/commit/186b491640a435442a89dc6d3f9420418f87a5cf>`_
     - Add Python 3.14 and remove a direct ``grpcio`` constraint.
     - Added in a follow-up after compatibility tests. The package needs
       conditional Pydantic and Studio requirements. See :doc:`dependency_updates`.
       It has no equivalent direct ``grpcio`` constraint to remove.

The ownership change needs particular care. It read ``user_id`` from checkpoint
configuration. A real graph reproduction showed that the configuration contains
checkpoint identifiers, while user metadata can change on later writes.
Upstream confirmed the failed check and reverted it in
`a561c88 <https://github.com/JoshuaC215/agent-service-toolkit/commit/a561c88fbac4b1914ea08c6ff113b32437fbcd9c>`_.
This package computes storage keys from authenticated user, agent, and public
thread ID. It does not use caller-written checkpoint metadata as proof of access.

The current upstream
`thread listing implementation
<https://github.com/JoshuaC215/agent-service-toolkit/blob/18d76e919b357e448f60b97c50e704db79456096/src/service/threads.py#L19-L109>`_
selects creation checkpoints, then reads each candidate's latest checkpoint.
An old but recently active conversation can fall outside the candidate scan.
Sequential reads also increase database work. A catalog should query indexed
ownership and update fields directly. The package's hashed storage keys cannot
be converted back into public thread IDs.

Corrections from this review
----------------------------

* **Feedback ownership.** The API previously accepted a supplied run ID after
  authenticating the caller. That did not prove run ownership. Token-user
  feedback now requires server-signed proof for the user, agent, and run.
  The server uses a separate signing secret. Trusted backend feedback keeps its
  existing flow. See :doc:`migration`. LangServe documents
  `scoped feedback tokens
  <https://github.com/langchain-ai/langserve/blob/27e57afeda13007a7f4e007c5d1f5e8489963aa4/langserve/server.py#L316-L324>`_.
* **Conversation navigation.** Streamlit now loads the selected agent's history,
  handles another URL in the same session, and starts a new conversation after
  an agent change. Resume links preserve deployment paths and do not grant
  another user's identity. See :doc:`usage`.
* **Client failure details.** ``AgentClientError`` retains optional HTTP status,
  service error code, and ``Retry-After``. Existing exception text stays
  compatible. Reads of rejected stream bodies have size and time limits.
  LangServe preserves
  `HTTP status exceptions
  <https://github.com/langchain-ai/langserve/blob/27e57afeda13007a7f4e007c5d1f5e8489963aa4/langserve/client.py#L157-L180>`_.
* **SSE proxy behavior.** SSE responses now request no caching and no proxy
  buffering. This follows the headers in the
  `response implementation used by LangServe
  <https://github.com/sysid/sse-starlette/blob/v1.3.0/sse_starlette/sse.py>`_.
  Proxy configuration can override these headers. Idle heartbeats are a separate
  change because stream cancellation must preserve graph cleanup.
* **Knowledge-base inputs.** Content-block messages previously reached a text
  retriever as lists. The retriever raised an error, which became an empty
  result. Retrieval now uses text blocks only and skips media-only questions.
  Chat LangChain also separates
  `text from other content
  <https://github.com/langchain-ai/chat-langchain/blob/f90060d55de772f543564696b56ffee7be3945b8/src/middleware/guardrails_middleware.py#L257>`_.
* **Knowledge-base sources.** Bedrock source locations and nested metadata now
  reach the augmented prompt. Existing custom sources and titles keep their
  precedence. The mapping follows the
  `Bedrock location contract
  <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_RetrievalResultLocation.html>`_.
  Chat LangChain similarly preserves source information in
  `retrieved articles
  <https://github.com/langchain-ai/chat-langchain/blob/f90060d55de772f543564696b56ffee7be3945b8/src/tools/pylon_tools.py#L296>`_.

Recommended next work
---------------------

1. Add the authenticated conversation catalog and Previous Chats UI. Use indexed
   pagination and test updates, deletion, restart, and isolation on real SQLite
   and PostgreSQL stores.
2. Add an optional operator model allowlist. Restrict per-request model/provider
   overrides when callers are less trusted. Upstream enforces its
   `available-model selection
   <https://github.com/JoshuaC215/agent-service-toolkit/blob/18d76e919b357e448f60b97c50e704db79456096/src/service/utils.py#L19-L26>`_.
   Preserve unrestricted selection for deployments that explicitly require it.
3. Add early stream metadata and configurable idle heartbeats with a documented
   protocol version. Today, a client that needs the thread ID after an early
   disconnect should supply it before streaming. LangServe emits
   `run metadata before data
   <https://github.com/langchain-ai/langserve/blob/27e57afeda13007a7f4e007c5d1f5e8489963aa4/langserve/api_handler.py#L1188-L1207>`_.
   Test slow readers and same-task graph cleanup before changing the producer.
4. Add a separate documentation research blueprint if applications need it.
   Bound search/read rounds, retain source IDs, and test source-supported answers.
   Chat LangChain provides useful
   `research instructions
   <https://github.com/langchain-ai/chat-langchain/blob/f90060d55de772f543564696b56ffee7be3945b8/instructions.md#L270>`_
   and explicit
   `summarization settings
   <https://github.com/langchain-ai/chat-langchain/blob/f90060d55de772f543564696b56ffee7be3945b8/agent.py#L40>`_.
   Network link checks need bounded caches, host restrictions, and redirect rules.
5. Add AG-UI or per-agent schema discovery when a consumer needs those interfaces.
   Keep authentication, admission limits, and persistence in the existing service.
   Review upstream
   `AG-UI event filtering
   <https://github.com/JoshuaC215/agent-service-toolkit/blob/18d76e919b357e448f60b97c50e704db79456096/src/service/agui.py#L74-L115>`_
   and LangServe's
   `schema endpoints
   <https://github.com/langchain-ai/langserve/blob/27e57afeda13007a7f4e007c5d1f5e8489963aa4/langserve/server.py#L630-L704>`_.

The fixes use deterministic graphs, Streamlit AppTest, synthetic Bedrock
responses, and local HTTP transports. No competitor application was executed.
No real model calls were needed. See :doc:`testing` for broader test boundaries.
The full local run with ``--run-e2e`` completed with 938 tests passed and 36
skipped. Live provider checks were not part of this review.
