import warnings
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from langchain_core._api import LangChainBetaWarning

from langgraph_agent_toolkit import __version__
from langgraph_agent_toolkit.core.runtime import agent_runtime
from langgraph_agent_toolkit.core.runtime import shutdown_observability as shutdown_observability
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
    app.state.mcp_transport_manager = None
    app.state.agent_runtime = None
    app.state.background_jobs = None
    try:
        validate_auth_configuration()
        blocking = BoundedBlockingExecutor(settings.REQUEST_MAX_CONCURRENT)
        app.state.blocking_executor = blocking
        try:
            async with agent_runtime() as runtime, AsyncExitStack() as resources:
                app.state.agent_runtime = runtime
                app.state.db_pool = runtime.db_pool
                app.state.lock_pool = runtime.lock_pool
                app.state.sqlite_connection = runtime.sqlite_connection
                app.state.llm_transport_manager = runtime.llm_transport_manager
                app.state.mcp_transport_manager = runtime.mcp_transport_manager
                app.state.agent_executor = runtime.executor
                app.state.initialized_agents = list(runtime.initialized_agents)
                if settings.BACKGROUND_ENABLED:
                    from langgraph_agent_toolkit.core.background import queue_from_settings

                    app.state.background_jobs = await resources.enter_async_context(queue_from_settings())
                app.state.startup_complete = True
                app.state.ready = True
                logger.info(f"Initialized {len(runtime.initialized_agents)} agents")
                try:
                    yield
                finally:
                    # Feedback threads can still use the observability providers.
                    await blocking.aclose(settings.OBSERVABILITY_SHUTDOWN_TIMEOUT)
        finally:
            # Runtime startup can fail before the service accepts any requests.
            if not app.state.startup_complete:
                await blocking.aclose(settings.OBSERVABILITY_SHUTDOWN_TIMEOUT)
    finally:
        app.state.ready = False
        app.state.db_pool = None
        app.state.lock_pool = None
        app.state.sqlite_connection = None
        app.state.llm_transport_manager = None
        app.state.mcp_transport_manager = None
        app.state.agent_runtime = None
        app.state.background_jobs = None
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

    if settings.BACKGROUND_ENABLED:
        from langgraph_agent_toolkit.service.jobs import jobs_router

        app.include_router(jobs_router, dependencies=[Depends(verify_bearer)], responses=COMMON_ERROR_RESPONSES)

    return app
