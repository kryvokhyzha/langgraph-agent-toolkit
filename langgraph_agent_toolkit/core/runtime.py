"""Own agent resources independently of an HTTP server."""

import asyncio
import threading
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

import aiosqlite
from langgraph.checkpoint.memory import MemorySaver
from psycopg_pool import AsyncConnectionPool

from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.mcp import MCPTransportManager, configure_mcp_agents
from langgraph_agent_toolkit.core.memory.concurrency import (
    ConversationCoordinator,
    PostgresConversationCoordinator,
    SQLiteConversationCoordinator,
)
from langgraph_agent_toolkit.core.memory.factory import MemoryFactory
from langgraph_agent_toolkit.core.memory.types import MemoryBackends
from langgraph_agent_toolkit.core.models.transport import LLMTransportManager
from langgraph_agent_toolkit.core.observability.factory import ObservabilityFactory
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.logging import logger


@dataclass(frozen=True)
class AgentRuntime:
    """Resources valid for the lifetime of an ``agent_runtime`` context.

    Run and await work inside the context. Direct graph calls still need explicit
    identity, concurrency, and cancellation controls.
    """

    executor: AgentExecutor
    llm_transport_manager: LLMTransportManager
    mcp_transport_manager: MCPTransportManager
    initialized_agents: tuple[str, ...]
    db_pool: AsyncConnectionPool | None = None
    lock_pool: AsyncConnectionPool | None = None
    sqlite_connection: aiosqlite.Connection | None = None


async def shutdown_observability(observability) -> None:
    """Give synchronous telemetry flush a deadline without blocking worker exit."""
    loop = asyncio.get_running_loop()
    completed = loop.create_future()

    def finish(error: Exception | None) -> None:
        if completed.done():
            return
        if error is None:
            completed.set_result(None)
        else:
            completed.set_exception(error)

    def flush() -> None:
        error = None
        try:
            observability.before_shutdown()
        except Exception as exc:
            error = exc
        try:
            loop.call_soon_threadsafe(finish, error)
        except RuntimeError:
            pass  # The worker event loop has already stopped.

    # A stuck default-executor thread would also block asyncio.run() shutdown.
    threading.Thread(target=flush, name="observability-shutdown", daemon=True).start()
    try:
        await asyncio.wait_for(completed, settings.OBSERVABILITY_SHUTDOWN_TIMEOUT)
    except TimeoutError:
        logger.warning("Observability flush exceeded its shutdown time limit; pending telemetry may be lost")
    except Exception:
        logger.opt(exception=True).warning("Observability flush failed during shutdown")


@asynccontextmanager
async def agent_runtime(*agent_paths: str) -> AsyncGenerator[AgentRuntime, None]:
    """Initialize and close resources for API or worker execution.

    Agent paths default to ``settings.AGENT_PATHS``. Authentication belongs to
    the caller. Startup failures close all resources opened before the failure.
    """
    async with AsyncExitStack() as resources:
        manager = await resources.enter_async_context(LLMTransportManager.from_settings(settings))
        resources.enter_context(manager.bind())
        mcp_manager = await resources.enter_async_context(MCPTransportManager())
        observability = ObservabilityFactory.create(settings.OBSERVABILITY_BACKEND or ObservabilityBackend.EMPTY)
        telemetry_resources = await resources.enter_async_context(AsyncExitStack())
        telemetry_resources.push_async_callback(shutdown_observability, observability)
        telemetry_ids = {id(observability)}
        saver = None
        db_pool = None
        lock_pool = None
        sqlite_connection = None
        concurrency = ConversationCoordinator()
        if settings.MEMORY_BACKEND:
            backend = MemoryFactory.create(settings.MEMORY_BACKEND)
            saver = await resources.enter_async_context(backend.get_checkpoint_saver())
            await saver.setup()
            if settings.MEMORY_BACKEND == MemoryBackends.POSTGRES:
                db_pool = saver.conn
                lock_pool = await resources.enter_async_context(backend.get_lock_pool())
                concurrency = PostgresConversationCoordinator(lock_pool)
            else:
                sqlite_connection = saver.conn
                if settings.SQLITE_DB_PATH != ":memory:":
                    concurrency = SQLiteConversationCoordinator(settings.SQLITE_DB_PATH)
                else:
                    logger.warning(
                        "SQLite checkpoints use :memory: and are not durable. "
                        "Workers do not share this database. Worker shutdown deletes its checkpoints."
                    )

        executor = AgentExecutor(*(agent_paths or settings.AGENT_PATHS))
        executor.concurrency = concurrency
        agents = executor.get_all_agent_info()
        if not agents:
            raise RuntimeError("No agents were initialized")
        # Register custom providers before graph setup can fail.
        for info in agents:
            agent = executor.get_agent(info.key)
            if agent.observability is None:
                agent.observability = observability
            if id(agent.observability) not in telemetry_ids:
                telemetry_ids.add(id(agent.observability))
                telemetry_resources.push_async_callback(shutdown_observability, agent.observability)
        await configure_mcp_agents(executor, settings, rebuild_all=True, manager=mcp_manager)
        for info in agents:
            agent = executor.get_agent(info.key)
            if agent.graph.checkpointer is None:
                agent.graph.checkpointer = saver if saver is not None else MemorySaver()
            if isinstance(agent.graph.checkpointer, MemorySaver):
                logger.warning(
                    f"Agent {info.key!r} uses MemorySaver. Checkpoints are not durable. "
                    "Workers do not share these checkpoints. Worker shutdown deletes them."
                )

        yield AgentRuntime(
            executor=executor,
            llm_transport_manager=manager,
            mcp_transport_manager=mcp_manager,
            initialized_agents=tuple(info.key for info in agents),
            db_pool=db_pool,
            lock_pool=lock_pool,
            sqlite_connection=sqlite_connection,
        )
