"""Check shared connection options on a real PostgreSQL session."""

import os

import pytest
from psycopg.conninfo import make_conninfo

from langgraph_agent_toolkit.core._base_settings import Settings
from langgraph_agent_toolkit.core.memory.pool import CheckedAsyncConnectionPool
from langgraph_agent_toolkit.core.postgres import postgres_connection_string, postgres_pool_options


@pytest.mark.postgres
async def test_shared_pool_overrides_managed_options_and_preserves_extra_options():
    dsn = os.environ.get("LAT_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("Set LAT_TEST_POSTGRES_DSN to use a disposable PostgreSQL database")
    config = Settings(
        _env_file=None,
        POSTGRES_APPLICATION_NAME="lat-config-test",
        POSTGRES_STATEMENT_TIMEOUT=1234,
        POSTGRES_LOCK_TIMEOUT=234,
        POSTGRES_IDLE_IN_TRANSACTION_SESSION_TIMEOUT=3456,
        POSTGRES_POOL_TIMEOUT=5,
    )
    conninfo = postgres_connection_string(
        config,
        dsn=make_conninfo(dsn, options="-c search_path=public -c statement_timeout=9000 -c timezone=UTC"),
    )
    options = postgres_pool_options(
        config, min_size=1, max_size=2, schema="lat_config_test", app_prefix="background", conninfo=conninfo
    )
    async with CheckedAsyncConnectionPool(conninfo, **options) as pool:
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT current_setting('search_path') AS schema, "
                    "current_setting('statement_timeout') AS statement_timeout, "
                    "current_setting('lock_timeout') AS lock_timeout, "
                    "current_setting('idle_in_transaction_session_timeout') AS idle_timeout, "
                    "current_setting('timezone') AS timezone, "
                    "current_setting('application_name') AS application_name"
                )
            ).fetchone()
            assert row == {
                "schema": "lat_config_test",
                "statement_timeout": "1234ms",
                "lock_timeout": "234ms",
                "idle_timeout": "3456ms",
                "timezone": "UTC",
                "application_name": "lat-config-test-background",
            }
            assert conn.autocommit is True
