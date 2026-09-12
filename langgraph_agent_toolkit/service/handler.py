import asyncio
import threading
import warnings
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from langchain_core._api import LangChainBetaWarning
from langgraph.checkpoint.memory import MemorySaver

from langgraph_agent_toolkit import __version__
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.mcp import configure_mcp_agents
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
from langgraph_agent_toolkit.service.admission import RequestAdmissionMiddleware
from langgraph_agent_toolkit.service.auth import validate_auth_configuration
from langgraph_agent_toolkit.service.blocking import BoundedBlockingExecutor
from langgraph_agent_toolkit.service.exception_handlers import register_exception_handlers
from langgraph_agent_toolkit.service.middleware import LoggingMiddleware, RequestSizeLimitMiddleware
from langgraph_agent_toolkit.service.routes import COMMON_ERROR_RESPONSES, private_router, public_router
from langgraph_agent_toolkit.service.utils import verify_bearer


warnings.filterwarnings("ignore", category=LangChainBetaWarning)


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
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize all required resources or fail worker startup."""
    app.state.ready = False
    app.state.startup_complete = False
    app.state.initialized_agents = []
    app.state.db_pool = None
    app.state.lock_pool = None
    app.state.sqlite_connection = None
    app.state.llm_transport_manager = None
    try:
        validate_auth_configuration()
        async with AsyncExitStack() as resources:
            manager = await resources.enter_async_context(LLMTransportManager.from_settings(settings))
            resources.enter_context(manager.bind())
            app.state.llm_transport_manager = manager
            observability = ObservabilityFactory.create(settings.OBSERVABILITY_BACKEND or ObservabilityBackend.EMPTY)
            resources.push_async_callback(shutdown_observability, observability)
            app.state.blocking_executor = BoundedBlockingExecutor(settings.REQUEST_MAX_CONCURRENT)
            resources.push_async_callback(app.state.blocking_executor.aclose, settings.OBSERVABILITY_SHUTDOWN_TIMEOUT)
            saver = None
            concurrency = ConversationCoordinator()
            if settings.MEMORY_BACKEND:
                backend = MemoryFactory.create(settings.MEMORY_BACKEND)
                saver = await resources.enter_async_context(backend.get_checkpoint_saver())
                await saver.setup()
                if settings.MEMORY_BACKEND == MemoryBackends.POSTGRES:
                    app.state.db_pool = saver.conn
                    lock_pool = await resources.enter_async_context(backend.get_lock_pool())
                    app.state.lock_pool = lock_pool
                    concurrency = PostgresConversationCoordinator(lock_pool)
                else:
                    app.state.sqlite_connection = saver.conn
                    if settings.SQLITE_DB_PATH != ":memory:":
                        concurrency = SQLiteConversationCoordinator(settings.SQLITE_DB_PATH)

            executor = AgentExecutor(*settings.AGENT_PATHS)
            executor.concurrency = concurrency
            await configure_mcp_agents(executor, settings, rebuild_all=True)
            agents = executor.get_all_agent_info()
            if not agents:
                raise RuntimeError("No agents were initialized")
            for info in agents:
                agent = executor.get_agent(info.key)
                if agent.graph.checkpointer is None:
                    agent.graph.checkpointer = saver if saver is not None else MemorySaver()
                if agent.observability is None:
                    agent.observability = observability
                app.state.initialized_agents.append(info.key)

            app.state.agent_executor = executor
            app.state.startup_complete = True
            app.state.ready = True
            logger.info(f"Initialized {len(agents)} agents")
            yield
    finally:
        app.state.ready = False
        app.state.db_pool = None
        app.state.lock_pool = None
        app.state.sqlite_connection = None
        app.state.llm_transport_manager = None
        if hasattr(app.state, "agent_executor"):
            del app.state.agent_executor


def custom_generate_unique_id(route: APIRoute) -> str:
    """Use the route function name as its OpenAPI `operationId`.

    This produces client-friendly IDs, such as `invoke` instead of `invoke_invoke_post`.
    Routes that share a function set `operation_id` on the `/{agent_id}/...` decorator.
    """
    return route.name


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    logger.info(f"Initializing API service v{__version__}")

    app = FastAPI(
        lifespan=lifespan,
        title="LangGraph Agent API",
        description="API for interacting with LangGraph agents",
        version=__version__,
        generate_unique_id_function=custom_generate_unique_id,
        openapi_tags=[
            {"name": "agent", "description": "Invoke and stream agent responses (SSE and JSON Lines)."},
            {"name": "info", "description": "Service and agent metadata."},
            {"name": "history", "description": "Conversation history management."},
            {"name": "feedback", "description": "Record feedback to the configured observability platform."},
            {"name": "healthcheck", "description": "Liveness, readiness, startup, and database-pool probes."},
            {"name": "public", "description": "Unauthenticated endpoints (home / docs redirect)."},
        ],
    )
    app.state.blocking_executor = BoundedBlockingExecutor(settings.REQUEST_MAX_CONCURRENT)

    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware)
    app.add_middleware(RequestAdmissionMiddleware)

    # CORS also applies to admission and request-size errors.
    if settings.CORS_ENABLED:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_credentials=settings.CORS_CREDENTIALS,
            allow_methods=settings.CORS_METHODS,
            allow_headers=settings.CORS_HEADERS,
            max_age=settings.CORS_MAX_AGE,
        )
        logger.info(
            f"CORS enabled with origins: {settings.CORS_ORIGINS}, "
            f"credentials: {settings.CORS_CREDENTIALS}, "
            f"methods: {settings.CORS_METHODS}"
        )

    # Register exception handlers.
    register_exception_handlers(app)

    # Include the public router without authentication.
    app.include_router(public_router)

    # Include the authenticated router with shared OpenAPI error responses.
    app.include_router(private_router, dependencies=[Depends(verify_bearer)], responses=COMMON_ERROR_RESPONSES)

    return app
