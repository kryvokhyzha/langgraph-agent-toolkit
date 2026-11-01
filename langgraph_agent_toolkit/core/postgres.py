"""Share PostgreSQL connection and pool controls across toolkit backends."""

from typing import TYPE_CHECKING, Any

from psycopg import ProgrammingError
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row

from langgraph_agent_toolkit.helper.logging import logger


if TYPE_CHECKING:
    from langgraph_agent_toolkit.core._base_settings import Settings
    from langgraph_agent_toolkit.core.memory.pool import CheckedAsyncConnectionPool


def validate_postgres_credentials(config: "Settings") -> None:
    """Require the shared connection fields without including their values."""
    required = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB")
    missing = [name for name in required if not getattr(config, name, None)]
    if missing:
        raise ValueError(f"Missing required PostgreSQL configuration: {', '.join(missing)}")


def postgres_connection_string(config: "Settings", *, dsn: str | None = None) -> str:
    """Use an explicit DSN or the shared credentials and network defaults.

    Explicit DSN parameters take precedence over network defaults. TLS and
    other libpq parameters remain in the DSN. Managed session options are
    applied separately by ``postgres_connection_kwargs``.
    """
    defaults = {
        "connect_timeout": config.POSTGRES_CONNECT_TIMEOUT,
        "keepalives": 1,
        "keepalives_idle": config.POSTGRES_KEEPALIVES_IDLE,
        "keepalives_interval": config.POSTGRES_KEEPALIVES_INTERVAL,
        "keepalives_count": config.POSTGRES_KEEPALIVES_COUNT,
        "tcp_user_timeout": config.POSTGRES_TCP_USER_TIMEOUT,
    }
    if dsn is None:
        validate_postgres_credentials(config)
        parameters = {
            "user": config.POSTGRES_USER,
            "password": config.POSTGRES_PASSWORD.get_secret_value(),
            "host": config.POSTGRES_HOST,
            "port": config.POSTGRES_PORT,
            "dbname": config.POSTGRES_DB,
        }
    else:
        if not dsn.strip():
            raise ValueError("Invalid PostgreSQL connection string")
        try:
            parameters = conninfo_to_dict(dsn)
        except (ProgrammingError, ValueError):
            raise ValueError("Invalid PostgreSQL connection string") from None
    return make_conninfo(**(defaults | parameters))


def postgres_connection_kwargs(
    config: "Settings", *, schema: str, app_prefix: str, conninfo: str = "", force_schema: bool = False
) -> dict[str, Any]:
    """Apply the same session controls to each PostgreSQL backend.

    Keep extra DSN options. Append managed schema and timeout options so that
    they take precedence. For memory's default public schema, preserve the
    existing role search path.
    """
    options = []
    inherited = conninfo_to_dict(conninfo).get("options")
    if inherited:
        options.append(inherited)
    if schema and (schema != "public" or force_schema):
        options.append(f"-c search_path={schema}")
    options.extend(
        (
            f"-c statement_timeout={config.POSTGRES_STATEMENT_TIMEOUT}",
            f"-c lock_timeout={config.POSTGRES_LOCK_TIMEOUT}",
            f"-c idle_in_transaction_session_timeout={config.POSTGRES_IDLE_IN_TRANSACTION_SESSION_TIMEOUT}",
        )
    )
    return {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
        "application_name": f"{config.POSTGRES_APPLICATION_NAME}-{app_prefix}",
        "options": " ".join(options),
    }


def _reconnect_failed(pool: "CheckedAsyncConnectionPool") -> None:
    stats = pool.get_stats()
    logger.error(
        "Failed to reconnect to PostgreSQL. Pool stats: pool_size={}, pool_available={}",
        stats.get("pool_size", "N/A"),
        stats.get("pool_available", "N/A"),
    )


def postgres_pool_options(
    config: "Settings",
    *,
    min_size: int,
    max_size: int,
    schema: str,
    app_prefix: str,
    conninfo: str = "",
    force_schema: bool = False,
) -> dict[str, Any]:
    """Build options for ``CheckedAsyncConnectionPool`` without opening it."""
    if not 0 <= min_size <= max_size or max_size < 1:
        raise ValueError("PostgreSQL pool minimum must be between zero and its positive maximum")
    return {
        "min_size": min_size,
        "max_size": max_size,
        "max_waiting": config.POSTGRES_POOL_MAX_WAITING,
        "max_idle": config.POSTGRES_MAX_IDLE,
        "timeout": config.POSTGRES_POOL_TIMEOUT,
        "reconnect_timeout": config.POSTGRES_RECONNECT_TIMEOUT,
        "max_lifetime": config.POSTGRES_MAX_LIFETIME,
        "num_workers": config.POSTGRES_NUM_WORKERS,
        "health_check_timeout": config.POSTGRES_HEALTH_CHECK_TIMEOUT,
        "reconnect_failed": _reconnect_failed,
        "kwargs": postgres_connection_kwargs(
            config, schema=schema, app_prefix=app_prefix, conninfo=conninfo, force_schema=force_schema
        ),
        "open": False,
    }
