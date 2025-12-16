from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TypeVar

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from langgraph_agent_toolkit.core.memory.base import BaseMemoryBackend
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.logging import logger


T = TypeVar("T")


class PostgresMemoryBackend(BaseMemoryBackend):
    """PostgreSQL implementation of memory backend."""

    def validate_config(self) -> bool:
        """Validate that all required PostgreSQL configuration is present."""
        required_vars = [
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DB",
        ]

        missing = [var for var in required_vars if not getattr(settings, var, None)]
        if missing:
            raise ValueError(
                f"Missing required PostgreSQL configuration: {', '.join(missing)}. "
                "These environment variables must be set to use PostgreSQL persistence."
            )

        if settings.POSTGRES_MIN_SIZE > settings.POSTGRES_POOL_SIZE:
            raise ValueError(
                f"POSTGRES_MIN_SIZE ({settings.POSTGRES_MIN_SIZE}) must be less than or equal to "
                f"POSTGRES_POOL_SIZE ({settings.POSTGRES_POOL_SIZE})"
            )

        return True

    @staticmethod
    def get_connection_string() -> str:
        """Build and return the PostgreSQL connection string from settings."""
        return (
            f"postgresql://{settings.POSTGRES_USER}:"
            f"{settings.POSTGRES_PASSWORD.get_secret_value()}@"
            f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/"
            f"{settings.POSTGRES_DB}"
        )

    @asynccontextmanager
    async def _get_connection_context(
        self,
        factory_func: Callable[[AsyncConnectionPool], T],
        app_prefix: str,
    ) -> AsyncGenerator[T, None]:
        """Yield the result of the factory function.

        Args:
            factory_func: Function that creates the appropriate object from the connection pool
            app_prefix: Prefix for the application name in the connection pool

        Yields:
            The object created by the factory_func

        """
        application_name = f"{settings.POSTGRES_APPLICATION_NAME}-{app_prefix}"

        logger.info(
            f"Creating PostgreSQL connection pool: min_size={settings.POSTGRES_MIN_SIZE}, "
            f"max_size={settings.POSTGRES_POOL_SIZE}, max_idle={settings.POSTGRES_MAX_IDLE}, "
            f"timeout={settings.POSTGRES_POOL_TIMEOUT}s, reconnect_timeout={settings.POSTGRES_RECONNECT_TIMEOUT}s, "
            f"max_lifetime={settings.POSTGRES_MAX_LIFETIME}s, "
            f"statement_timeout={settings.POSTGRES_STATEMENT_TIMEOUT}ms, "
            f"lock_timeout={settings.POSTGRES_LOCK_TIMEOUT}ms, "
            f"idle_in_transaction_timeout={settings.POSTGRES_IDLE_IN_TRANSACTION_SESSION_TIMEOUT}ms, "
            f"schema={settings.POSTGRES_SCHEMA}, application_name={application_name}"
        )

        # Prepare connection kwargs with schema setting
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "application_name": application_name,
        }

        # Build PostgreSQL options string with timeouts and schema
        pg_options = []

        # Set search_path if schema is specified and not default
        if settings.POSTGRES_SCHEMA and settings.POSTGRES_SCHEMA != "public":
            pg_options.append(f"-c search_path={settings.POSTGRES_SCHEMA}")

        # Set statement timeout - kills queries running too long
        if settings.POSTGRES_STATEMENT_TIMEOUT > 0:
            pg_options.append(f"-c statement_timeout={settings.POSTGRES_STATEMENT_TIMEOUT}")

        # Set lock timeout - prevents waiting forever for locks
        if settings.POSTGRES_LOCK_TIMEOUT > 0:
            pg_options.append(f"-c lock_timeout={settings.POSTGRES_LOCK_TIMEOUT}")

        # Set idle_in_transaction_session_timeout - kills idle transactions
        if settings.POSTGRES_IDLE_IN_TRANSACTION_SESSION_TIMEOUT > 0:
            pg_options.append(
                f"-c idle_in_transaction_session_timeout={settings.POSTGRES_IDLE_IN_TRANSACTION_SESSION_TIMEOUT}"
            )

        if pg_options:
            connection_kwargs["options"] = " ".join(pg_options)

        # Callback for logging reconnection attempts
        def on_reconnect_failed(pool: AsyncConnectionPool) -> None:
            logger.error(
                f"Failed to reconnect to PostgreSQL. Pool stats: "
                f"pool_size={pool.get_stats().get('pool_size', 'N/A')}, "
                f"pool_available={pool.get_stats().get('pool_available', 'N/A')}"
            )

        # Use AsyncConnectionPool as an async context manager
        async with AsyncConnectionPool(
            self.get_connection_string(),
            min_size=settings.POSTGRES_MIN_SIZE,
            max_size=settings.POSTGRES_POOL_SIZE,
            max_idle=settings.POSTGRES_MAX_IDLE,
            timeout=settings.POSTGRES_POOL_TIMEOUT,
            reconnect_timeout=settings.POSTGRES_RECONNECT_TIMEOUT,
            # Maximum time a connection can live before being recycled (prevents stale connections)
            max_lifetime=settings.POSTGRES_MAX_LIFETIME,
            # Number of background workers for connection maintenance
            num_workers=settings.POSTGRES_NUM_WORKERS,
            # Automatically check connection health before returning from pool
            check=AsyncConnectionPool.check_connection,
            # Callback when reconnection fails
            reconnect_failed=on_reconnect_failed,
            # Connection configuration
            kwargs=connection_kwargs,
            # Don't open immediately, we'll open manually after setup
            open=False,
        ) as pool:
            # Open the pool and wait for min_size connections
            await pool.open(wait=True, timeout=settings.POSTGRES_POOL_TIMEOUT)
            logger.info(
                f"PostgreSQL connection pool opened successfully. "
                f"Initial stats: pool_size={pool.get_stats().get('pool_size', 'N/A')}, "
                f"pool_available={pool.get_stats().get('pool_available', 'N/A')}"
            )

            try:
                yield factory_func(pool)
            except PoolTimeout:
                # Log pool statistics for debugging
                stats = pool.get_stats()
                logger.error(
                    f"Pool timeout occurred. Pool stats: "
                    f"pool_size={stats.get('pool_size', 'N/A')}, "
                    f"pool_available={stats.get('pool_available', 'N/A')}, "
                    f"requests_waiting={stats.get('requests_waiting', 'N/A')}, "
                    f"requests_num={stats.get('requests_num', 'N/A')}"
                )
                raise
            finally:
                logger.info("PostgreSQL connection pool will be closed automatically")

    @asynccontextmanager
    async def get_saver(self) -> AsyncGenerator[AsyncPostgresSaver, None]:
        """Asynchronous context manager for acquiring a PostgreSQL saver.

        Yields:
            AsyncPostgresSaver: The database saver instance

        """
        async with self._get_connection_context(
            lambda pool: AsyncPostgresSaver(conn=pool), app_prefix="saver"
        ) as saver:
            yield saver

    @asynccontextmanager
    async def get_store(self) -> AsyncGenerator[AsyncPostgresStore, None]:
        """Asynchronous context manager for acquiring a PostgreSQL store.

        Yields:
            AsyncPostgresStore: The database store instance

        """
        async with self._get_connection_context(
            lambda pool: AsyncPostgresStore(conn=pool, app_prefix="store")
        ) as store:
            yield store

    def get_checkpoint_saver(self) -> AbstractAsyncContextManager[AsyncPostgresSaver]:
        """Initialize and return a PostgreSQL saver instance."""
        self.validate_config()
        return self.get_saver()

    def get_memory_store(self) -> AbstractAsyncContextManager[AsyncPostgresStore]:
        """Initialize and return a PostgreSQL store instance."""
        self.validate_config()
        return self.get_store()
