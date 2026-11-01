"""Test connection overrides and background configuration compatibility."""

import os
from unittest.mock import patch

import pytest
from psycopg.conninfo import conninfo_to_dict
from pydantic import SecretStr, ValidationError

from langgraph_agent_toolkit.core._base_settings import Settings
from langgraph_agent_toolkit.core.postgres import postgres_connection_string


@pytest.mark.parametrize("prefix", ["", "LANGGRAPH_"])
@pytest.mark.parametrize("legacy", [False, True])
def test_background_postgres_setting_names(prefix, legacy):
    names = (
        ("BACKGROUND_SCHEMA", "BACKGROUND_POOL_MAX_SIZE")
        if legacy
        else ("BACKGROUND_POSTGRES_SCHEMA", "BACKGROUND_POSTGRES_POOL_SIZE")
    )
    with patch.dict(os.environ, {prefix + names[0]: "jobs_test", prefix + names[1]: "7"}, clear=True):
        config = Settings(_env_file=None)
        config._apply_langgraph_env_overrides()
    assert config.BACKGROUND_POSTGRES_SCHEMA == "jobs_test"
    assert config.BACKGROUND_POSTGRES_POOL_SIZE == 7


def test_background_canonical_names_take_precedence():
    with patch.dict(
        os.environ,
        {
            "BACKGROUND_POSTGRES_SCHEMA": "canonical",
            "BACKGROUND_SCHEMA": "legacy",
            "LANGGRAPH_BACKGROUND_POSTGRES_POOL_SIZE": "6",
            "LANGGRAPH_BACKGROUND_POOL_MAX_SIZE": "9",
        },
        clear=True,
    ):
        config = Settings(_env_file=None)
        config._apply_langgraph_env_overrides()
    assert config.BACKGROUND_POSTGRES_SCHEMA == "canonical"
    assert config.BACKGROUND_POSTGRES_POOL_SIZE == 6
    config.apply_overrides({"BACKGROUND_SCHEMA": "old", "BACKGROUND_POSTGRES_SCHEMA": "new"})
    assert config.BACKGROUND_POSTGRES_SCHEMA == "new"


def test_constructor_settings_are_not_replaced_by_unprefixed_alias_environment():
    with patch.dict(os.environ, {"BACKGROUND_SCHEMA": "environment"}, clear=True):
        config = Settings(_env_file=None, BACKGROUND_POSTGRES_SCHEMA="constructor")
        config._apply_langgraph_env_overrides()
    assert config.BACKGROUND_POSTGRES_SCHEMA == "constructor"


def test_legacy_background_constructor_and_invalid_override():
    with patch.dict(os.environ, {}, clear=True):
        config = Settings(_env_file=None, BACKGROUND_SCHEMA="legacy", BACKGROUND_POOL_MAX_SIZE=3)
    assert config.BACKGROUND_POSTGRES_SCHEMA == "legacy"
    assert config.BACKGROUND_POSTGRES_POOL_SIZE == 3
    with pytest.raises(ValidationError):
        config.apply_overrides({"BACKGROUND_POOL_MAX_SIZE": "1", "BACKGROUND_SCHEMA": "changed"})
    assert config.BACKGROUND_POSTGRES_SCHEMA == "legacy"
    assert config.BACKGROUND_POSTGRES_POOL_SIZE == 3


def test_explicit_dsn_preserves_tls_and_network_overrides_without_shared_credentials():
    config = Settings(_env_file=None, POSTGRES_PASSWORD=None, POSTGRES_HOST=None)
    conninfo = postgres_connection_string(
        config,
        dsn="postgresql://test-user:p%40ss%20word@db.example:5433/jobs?sslmode=verify-full&connect_timeout=4",
    )
    parameters = conninfo_to_dict(conninfo)
    assert parameters["password"] == "p@ss word"
    assert parameters["dbname"] == "jobs"
    assert parameters["sslmode"] == "verify-full"
    assert parameters["connect_timeout"] == "4"
    assert parameters["keepalives_idle"] == str(config.POSTGRES_KEEPALIVES_IDLE)


def test_shared_credentials_are_required_only_without_dsn():
    config = Settings(_env_file=None, POSTGRES_PASSWORD=SecretStr("private-password"), POSTGRES_HOST=None)
    with pytest.raises(ValueError, match="POSTGRES_HOST") as error:
        postgres_connection_string(config)
    assert "private-password" not in str(error.value)
    assert "private-password" not in repr(config)


@pytest.mark.parametrize("dsn", ["", "  ", "postgresql://user:private-password@host/db?unsupported_parameter=yes"])
def test_invalid_dsn_does_not_expose_credentials(dsn):
    with pytest.raises(ValueError, match="Invalid PostgreSQL connection string") as error:
        postgres_connection_string(Settings(_env_file=None), dsn=dsn)
    assert "private-password" not in str(error.value)
