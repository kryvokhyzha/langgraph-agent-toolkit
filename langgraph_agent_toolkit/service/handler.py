import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core._api import LangChainBetaWarning

from langgraph_agent_toolkit import __version__
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.memory.factory import MemoryFactory
from langgraph_agent_toolkit.core.observability.empty import BaseObservabilityPlatform, EmptyObservability
from langgraph_agent_toolkit.core.observability.factory import ObservabilityFactory
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.logging import logger
from langgraph_agent_toolkit.service.exception_handlers import register_exception_handlers
from langgraph_agent_toolkit.service.middleware import LoggingMiddleware
from langgraph_agent_toolkit.service.routes import private_router, public_router
from langgraph_agent_toolkit.service.utils import verify_bearer


warnings.filterwarnings("ignore", category=LangChainBetaWarning)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create a lifespan context manager for the FastAPI app."""
    observability = None
    initialized_agents = []

    # Initialize readiness state - service is not ready until agents are loaded
    app.state.ready = False
    app.state.startup_complete = False
    app.state.initialized_agents = []

    def initialize_agents(
        executor: AgentExecutor,
        observability: BaseObservabilityPlatform,
        checkpointer: Optional[Any] = None,
    ):
        agents = executor.get_all_agent_info()
        if not agents:
            logger.warning("No agents found in the executor.")
        for a in agents:
            try:
                agent = executor.get_agent(a.key)

                if checkpointer and not agent.graph.checkpointer:
                    agent.graph.checkpointer = checkpointer

                if not agent.observability:
                    agent.observability = observability

                initialized_agents.append(a.key)
                logger.info(f"Successfully initialized agent: {a.key}")
            except Exception as e:
                logger.error(f"Error setting up agent {a.key}: {e}")

        # Update app state with initialized agents
        app.state.initialized_agents = initialized_agents.copy()

        if initialized_agents:
            logger.info(f"Successfully initialized {len(initialized_agents)} agents")
            # Mark service as ready only after agents are initialized
            app.state.ready = True
            app.state.startup_complete = True
            logger.info("Service is now ready to accept traffic")
        else:
            logger.warning("No agents were successfully initialized")

    try:
        # Initialize observability platform
        try:
            observability = ObservabilityFactory.create(settings.OBSERVABILITY_BACKEND or ObservabilityBackend.EMPTY)
            logger.info(f"Initialized observability backend: {settings.OBSERVABILITY_BACKEND}")
        except Exception as e:
            logger.error(f"Failed to initialize observability backend: {e}")
            observability = EmptyObservability()

        # Initialize memory backend
        try:
            memory_backend = MemoryFactory.create(settings.MEMORY_BACKEND) if settings.MEMORY_BACKEND else None

            if memory_backend:
                logger.info(f"Initialized memory backend: {settings.MEMORY_BACKEND}")
            else:
                logger.warning("No memory backend configured.")
        except Exception as e:
            logger.error(f"Failed to initialize memory backend: {e}")
            yield
            return

        # Initialize agent executor
        try:
            executor = AgentExecutor(*settings.AGENT_PATHS)
            logger.info(f"Initialized AgentExecutor: {settings.AGENT_PATHS}")
            app.state.agent_executor = executor
        except Exception as e:
            logger.error(f"Failed to initialize AgentExecutor: {e}")
            yield
            return

        if memory_backend:
            checkpoint = memory_backend.get_checkpoint_saver()
            async with checkpoint as saver:
                try:
                    if saver is not None:
                        await saver.setup()
                        # Store pool reference for health monitoring
                        if hasattr(saver, "conn") and saver.conn is not None:
                            app.state.db_pool = saver.conn
                    initialize_agents(executor, observability, checkpointer=saver)
                    yield
                except Exception as e:
                    logger.error(f"Error during database setup: {e}")
                    yield
        else:
            initialize_agents(executor, observability)
            yield
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        yield
    finally:
        if observability:
            try:
                logger.info("Closing observability platform...")
                observability.before_shutdown()
            except Exception as e:
                logger.error(f"Error closing observability: {e}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    logger.info(f"Initializing API service v{__version__}")

    app = FastAPI(
        lifespan=lifespan,
        title="LangGraph Agent API",
        description="API for interacting with LangGraph agents",
        version=__version__,
    )

    # Add CORS middleware if explicitly enabled
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

    # add middleware
    app.add_middleware(LoggingMiddleware)

    # Register exception handlers
    register_exception_handlers(app)

    # Include public router without authentication
    app.include_router(public_router)

    # Include private router with authentication
    app.include_router(private_router, dependencies=[Depends(verify_bearer)])

    return app
