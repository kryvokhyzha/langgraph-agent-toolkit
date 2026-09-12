import hashlib
import json
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TypeVar

from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import PoolTimeout

from langgraph_agent_toolkit.core.memory.base import BaseMemoryBackend
from langgraph_agent_toolkit.core.memory.coordinated_saver import CoordinatedPostgresSaver as AsyncPostgresSaver
from langgraph_agent_toolkit.core.memory.coordinated_store import CoordinatedPostgresStore as AsyncPostgresStore
from langgraph_agent_toolkit.core.memory.pool import CheckedAsyncConnectionPool as AsyncConnectionPool
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.logging import logger


T = TypeVar("T")


class PostgresMemoryBackend(BaseMemoryBackend):
    """PostgreSQL memory backend."""

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
        """Build the PostgreSQL connection string from settings."""
        return make_conninfo(
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD.get_secret_value(),
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname=settings.POSTGRES_DB,
            connect_timeout=settings.POSTGRES_CONNECT_TIMEOUT,
            keepalives=1,
            keepalives_idle=settings.POSTGRES_KEEPALIVES_IDLE,
            keepalives_interval=settings.POSTGRES_KEEPALIVES_INTERVAL,
            keepalives_count=settings.POSTGRES_KEEPALIVES_COUNT,
            tcp_user_timeout=settings.POSTGRES_TCP_USER_TIMEOUT,
        )

    @asynccontextmanager
    async def _get_connection_context(
        self,
        factory_func: Callable[[AsyncConnectionPool], T],
        app_prefix: str,
    ) -> AsyncGenerator[T, None]:
        """Yield the object from the factory function.

        Args:
            factory_func: Function that creates an object from the connection pool.
            app_prefix: Application name prefix for the connection pool.

        Yields:
            Object from `factory_func`.

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

        # Prepare connection arguments with the schema setting.
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "application_name": application_name,
        }

        # Build PostgreSQL options for timeouts and schema.
        pg_options = []

        # Set `search_path` for a non-default schema.
        if settings.POSTGRES_SCHEMA and settings.POSTGRES_SCHEMA != "public":
            pg_options.append(f"-c search_path={settings.POSTGRES_SCHEMA}")

        # Set a timeout for long-running statements.
        if settings.POSTGRES_STATEMENT_TIMEOUT > 0:
            pg_options.append(f"-c statement_timeout={settings.POSTGRES_STATEMENT_TIMEOUT}")

        # Set a timeout while waiting for locks.
        if settings.POSTGRES_LOCK_TIMEOUT > 0:
            pg_options.append(f"-c lock_timeout={settings.POSTGRES_LOCK_TIMEOUT}")

        # Set a timeout for idle transactions.
        if settings.POSTGRES_IDLE_IN_TRANSACTION_SESSION_TIMEOUT > 0:
            pg_options.append(
                f"-c idle_in_transaction_session_timeout={settings.POSTGRES_IDLE_IN_TRANSACTION_SESSION_TIMEOUT}"
            )

        if pg_options:
            connection_kwargs["options"] = " ".join(pg_options)

        # Log failed reconnection attempts.
        def on_reconnect_failed(pool: AsyncConnectionPool) -> None:
            logger.error(
                f"Failed to reconnect to PostgreSQL. Pool stats: "
                f"pool_size={pool.get_stats().get('pool_size', 'N/A')}, "
                f"pool_available={pool.get_stats().get('pool_available', 'N/A')}"
            )

        # Use `AsyncConnectionPool` as an asynchronous context manager.
        # Each gunicorn or uvicorn worker has its own pool.
        # Size `POSTGRES_POOL_SIZE` with worker count below server `max_connections`.
        conninfo = self.get_connection_string()
        scope = hashlib.sha256(
            json.dumps([conninfo_to_dict(conninfo), settings.POSTGRES_SCHEMA], sort_keys=True).encode()
        ).hexdigest()
        async with AsyncConnectionPool(
            conninfo,
            min_size=1 if app_prefix == "locks" else settings.POSTGRES_MIN_SIZE,
            max_size=settings.POSTGRES_LOCK_POOL_SIZE if app_prefix == "locks" else settings.POSTGRES_POOL_SIZE,
            max_waiting=settings.POSTGRES_POOL_MAX_WAITING,
            max_idle=settings.POSTGRES_MAX_IDLE,
            timeout=settings.POSTGRES_POOL_TIMEOUT,
            reconnect_timeout=settings.POSTGRES_RECONNECT_TIMEOUT,
            # Recycle connections after this time.
            max_lifetime=settings.POSTGRES_MAX_LIFETIME,
            # Use this number of background workers for connection maintenance.
            num_workers=settings.POSTGRES_NUM_WORKERS,
            # Check connection health before returning it from the pool.
            health_check_timeout=settings.POSTGRES_HEALTH_CHECK_TIMEOUT,
            # Handle reconnection failure.
            reconnect_failed=on_reconnect_failed,
            # Set connection configuration.
            kwargs=connection_kwargs,
            # Open the pool manually after setup.
            open=False,
        ) as pool:
            pool._lat_checkpoint_scope = scope
            # Open the pool and wait for `min_size` connections.
            await pool.open(wait=True, timeout=settings.POSTGRES_POOL_TIMEOUT)
            logger.info(
                f"PostgreSQL connection pool opened successfully. "
                f"Initial stats: pool_size={pool.get_stats().get('pool_size', 'N/A')}, "
                f"pool_available={pool.get_stats().get('pool_available', 'N/A')}"
            )

            try:
                yield factory_func(pool)
            except PoolTimeout:
                # Log pool statistics.
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
        """Yield a PostgreSQL saver in an asynchronous context.

        Yields:
            AsyncPostgresSaver: Database saver.

        """
        async with self._get_connection_context(
            lambda pool: AsyncPostgresSaver(conn=pool), app_prefix="saver"
        ) as saver:
            yield saver

    @asynccontextmanager
    async def get_store(self) -> AsyncGenerator[AsyncPostgresStore, None]:
        """Yield a PostgreSQL store in an asynchronous context.

        Yields:
            AsyncPostgresStore: Database store.

        """
        async with self._get_connection_context(
            lambda pool: AsyncPostgresStore(conn=pool), app_prefix="store"
        ) as store:
            yield store

    def get_checkpoint_saver(self) -> AbstractAsyncContextManager[AsyncPostgresSaver]:
        """Initialize and return a PostgreSQL saver."""
        self.validate_config()
        return self.get_saver()

    def get_lock_pool(self):
        """Create a separate pool for long-running conversation locks."""
        self.validate_config()
        return self._get_connection_context(lambda pool: pool, app_prefix="locks")

    def get_memory_store(self) -> AbstractAsyncContextManager[AsyncPostgresStore]:
        """Initialize and return a PostgreSQL store."""
        self.validate_config()
        return self.get_store()
