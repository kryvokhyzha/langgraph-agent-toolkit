Choose an agent integration
===========================

Choose the agent flow from the task. Then choose its tools, memory, and
observability. These choices are independent. For example, both a native
``create_agent`` graph and a Deep Agent can use MCP tools.

Start with the least complex flow that meets the requirement. Add delegation
or file-based context when a measured task needs it.

Agent patterns
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Pattern
     - Use it for
     - Main tradeoff
   * - ``chatbot``
     - Conversation, rewriting, classification, or a response that needs no
       external tool.
     - A small execution path. Add explicit retrieval or tools when the model
       needs current or private information.
   * - ``create_agent``
     - Support agents that choose among a small set of business tools.
     - The model selects the tool order. Tool schemas and server permissions
       must enforce the business boundaries.
   * - ``create_agent_structured``
     - Extraction or decisions that another program consumes.
     - The response schema validates shape. The application must still validate
       business meaning, such as an allowed order status or account ID.
   * - Custom ``StateGraph``
     - A fixed validation, retrieval, approval, or transaction sequence.
     - You define state and transitions. This requires more code but makes
       required steps explicit.
   * - ``knowledge_base_agent``
     - Answers grounded in documents from Amazon Bedrock Knowledge Bases.
     - Configure the ``aws`` extra, ``AWS_KB_ID``, and AWS access. Retrieval
       quality and document access control remain application concerns.
   * - ``supervisor_agent``
     - Routing to a known set of specialist agents with separate tools.
     - Delegation adds model calls and handoff state. The example has research
       and arithmetic specialists; adapt those roles to the actual task.
   * - Deep Agents
     - Research, analysis, and document work that need intermediate files,
       several steps, or delegated subtasks.
     - More built-in behavior and context management. Measure latency, model
       calls, storage growth, and result quality on representative tasks.

The toolkit also retains its custom ``create_react_agent`` creator. Keep it
when an existing agent depends on its hooks and routing. Prefer the native
``create_agent`` path for a new tool agent unless those custom features are
required.

Deep Agents builds on ``create_agent`` and LangGraph. A custom graph can also
become a Deep Agents subagent. See the `official ecosystem overview
<https://docs.langchain.com/oss/python/concepts/products>`_ and
:doc:`deepagents` for the optional integration.

Concrete use cases
------------------

Customer support
~~~~~~~~~~~~~~~~

Use ``create_agent`` with narrow tools such as order lookup and shipment
status. Add ``create_agent_structured`` when the result feeds a ticket router.
Pass the authenticated user's scope to the tool implementation. Do not let a
model-selected account ID grant access to another customer's records.

Policy and product questions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a retrieval graph when every answer must consult an approved document
collection. The existing knowledge-base blueprint retrieves documents before
the model runs. Replace its retriever in a custom graph if your source is not
Amazon Bedrock. Return source identifiers and test missing, stale, and
contradictory documents.

An agent with a search tool can choose whether to search. A retrieval node in a
fixed graph makes that step mandatory. Choose the latter when this is a
business requirement.

Research and report preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Deep Agents when the task needs a working document, independent research
tasks, and a final synthesis. Give search tools to a research subagent and
writing tools to the agent that produces the report. Keep intermediate files
in thread state unless they must be shared across conversations.

This pattern can keep large tool results out of the main prompt. It also adds
delegation and summarization work. Compare it with a single ``create_agent``
on the same input set before selecting it for a latency-sensitive endpoint.
See `Deep Agents context management
<https://docs.langchain.com/oss/python/deepagents/context-engineering>`_.

Approvals and updates to business systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a custom graph or human-in-the-loop middleware when a person must approve
an update. Place approval before the external write. Use an idempotency key in
the target API when repeating a request could create another order, payment,
or message. A checkpoint records graph progress; it does not make an external
side effect atomic with the database.

MCP and direct tools
--------------------

Use a direct LangChain tool when the application owns a small API client or a
database operation. This keeps validation, authorization, and connection
ownership in one service.

Use MCP when a service already exposes tools through that protocol, or when
several applications need the same tool service. Configure ``MCP_SERVERS`` and
assign servers with ``MCP_AGENT_SERVERS``. The toolkit discovers tools at worker
startup and injects them through each supported agent's graph factory.
See :doc:`mcp` for HTTP, stdio, credentials, allowlists, and elicitation.

MCP is a tool transport. It does not add task planning, document retrieval, or
authorization by itself. The remote server must enforce access. Toolkit API
credentials and MCP credentials serve different connections.

Memory and persistence
----------------------

``thread_id`` identifies short-term conversation state. Checkpoints can hold
messages, pending interrupts, and Deep Agents state files. Use the configured
SQLite or PostgreSQL checkpointer when this state must survive restart.

``user_id`` identifies the scope for long-term user memory. A store or a custom
memory service can share selected facts across that user's threads. Passing
``user_id`` alone does not create a store or make a backend user-specific.
Set the namespace explicitly and manage its connections in application
startup and shutdown.

A thread lock protects one conversation update. It does not serialize writes
to the same long-term record from different threads. Use atomic store
operations or a separate concurrency rule for such records.

Clearing conversation history removes checkpoints for that conversation.
Long-term memory and external files need their own retention and deletion
rules. See :doc:`migration` and :doc:`deployment` for the service contract.

Observe and test the selected combination
-----------------------------------------

Record tool failures, model-call counts, elapsed time, and successful task
outcomes. Check whether delegated runs preserve the parent trace and user
scope. Use the configured Langfuse or LangSmith backend. See
:doc:`langfuse_compatibility` for the tested Langfuse SDK and server boundaries.

Test the whole combination that you deploy: agent builder, model, tools,
database, and observability SDK. The deterministic examples verify execution
and state behavior. They do not establish the quality or cost of a real model.
Use a small representative evaluation set before changing an existing service
to a more complex agent flow. See :doc:`testing` for the test layers.
