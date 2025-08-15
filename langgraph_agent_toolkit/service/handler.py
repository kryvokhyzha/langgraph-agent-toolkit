import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from langchain_core._api import LangChainBetaWarning

from langgraph_agent_toolkit import __version__
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.memory.factory import MemoryFactory
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
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
            memory_backend = MemoryFactory.create(settings.MEMORY_BACKEND)
            logger.info(f"Initialized memory backend: {settings.MEMORY_BACKEND}")
        except Exception as e:
            logger.error(f"Failed to initialize memory backend: {e}")
            yield
            return

        # Initialize agent executor
        try:
            executor = AgentExecutor(*settings.AGENT_PATHS)
            logger.info(f"Initialized AgentExecutor: {settings.AGENT_PATHS}")
            # Store the executor in app.state for access in routes
            app.state.agent_executor = executor
        except Exception as e:
            logger.error(f"Failed to initialize AgentExecutor: {e}")
            yield
            return

        checkpoint = memory_backend.get_checkpoint_saver()
        async with checkpoint as saver:
            try:
                await saver.setup()
                agents = executor.get_all_agent_info()

                if not agents:
                    logger.warning("No agents found in the executor.")

                for a in agents:
                    try:
                        agent = executor.get_agent(a.key)
                        agent.graph.checkpointer = saver

                        # do not set up observability if it's already set
                        if not agent.observability:
                            agent.observability = observability

                        initialized_agents.append(a.key)
                        logger.info(f"Successfully initialized agent: {a.key}")
                    except Exception as e:
                        logger.error(f"Error setting up agent {a.key}: {e}")

                if initialized_agents:
                    logger.info(f"Successfully initialized {len(initialized_agents)} agents")
                else:
                    logger.warning("No agents were successfully initialized")

                yield
            except Exception as e:
                logger.error(f"Error during database setup: {e}")
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

    # add middleware
    app.add_middleware(LoggingMiddleware)

    # Register exception handlers
    register_exception_handlers(app)

    # Include public router without authentication
    app.include_router(public_router)

    # Include private router with authentication
    app.include_router(private_router, dependencies=[Depends(verify_bearer)])

    return app
