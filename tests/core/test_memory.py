import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from langgraph_agent_toolkit.core.memory.factory import MemoryFactory
from langgraph_agent_toolkit.core.memory.postgres import PostgresMemoryBackend
from langgraph_agent_toolkit.core.memory.sqlite import SQLiteMemoryBackend
from langgraph_agent_toolkit.core.memory.types import MemoryBackends


class TestMemoryFactory(unittest.TestCase):
    """Test the MemoryFactory class."""

    def test_factory_creates_postgres_backend(self):
        """Test that the factory creates a PostgreSQL backend."""
        backend = MemoryFactory.create(MemoryBackends.POSTGRES)
        self.assertIsInstance(backend, PostgresMemoryBackend)

    def test_factory_creates_sqlite_backend(self):
        """Test that the factory creates a SQLite backend."""
        backend = MemoryFactory.create(MemoryBackends.SQLITE)
        self.assertIsInstance(backend, SQLiteMemoryBackend)

    def test_factory_raises_on_unsupported_backend(self):
        """Test that the factory raises ValueError for unsupported backends."""
        # Create a mock enum value that doesn't exist
        with pytest.raises(ValueError, match=r"Unsupported memory backend:"):
            MemoryFactory.create("UNSUPPORTED_BACKEND")


class TestSQLiteMemoryBackend(unittest.TestCase):
    """Test the SQLiteMemoryBackend class."""

    def setUp(self):
        """Set up the test environment."""
        self.backend = SQLiteMemoryBackend()

    @patch("langgraph_agent_toolkit.core.memory.sqlite.settings")
    def test_validate_config_success(self, mock_settings):
        """Test that validation succeeds when configuration is present."""
        mock_settings.SQLITE_DB_PATH = "sqlite:///test.db"
        self.assertTrue(self.backend.validate_config())

    @patch("langgraph_agent_toolkit.core.memory.sqlite.settings")
    def test_validate_config_failure(self, mock_settings):
        """Test that validation fails when configuration is missing."""
        mock_settings.SQLITE_DB_PATH = None
        with pytest.raises(ValueError, match=r"Missing SQLITE_DB_PATH configuration"):
            self.backend.validate_config()

    @patch("langgraph_agent_toolkit.core.memory.sqlite.AsyncSqliteSaver")
    @patch("langgraph_agent_toolkit.core.memory.sqlite.settings")
    def test_get_checkpoint_saver(self, mock_settings, mock_saver):
        """Test that get_checkpoint_saver returns a correct saver."""
        mock_settings.SQLITE_DB_PATH = "sqlite:///test.db"
        mock_saver.from_conn_string.return_value = MagicMock()

        saver = self.backend.get_checkpoint_saver()

        mock_saver.from_conn_string.assert_called_once_with("sqlite:///test.db")
        self.assertEqual(saver, mock_saver.from_conn_string.return_value)


class TestPostgresMemoryBackend(unittest.TestCase):
    """Test the PostgresMemoryBackend class."""

    def setUp(self):
        """Set up the test environment."""
        self.backend = PostgresMemoryBackend()

    @patch("langgraph_agent_toolkit.core.memory.postgres.settings")
    def test_validate_config_success(self, mock_settings):
        """Test that validation succeeds when all required configuration is present."""
        mock_settings.POSTGRES_USER = "user"
        mock_settings.POSTGRES_PASSWORD = SecretStr("password")
        mock_settings.POSTGRES_HOST = "localhost"
        mock_settings.POSTGRES_PORT = "5432"
        mock_settings.POSTGRES_DB = "testdb"

        self.assertTrue(self.backend.validate_config())

    @patch("langgraph_agent_toolkit.core.memory.postgres.settings")
    def test_validate_config_failure(self, mock_settings):
        """Test that validation fails when configuration is missing."""
        mock_settings.POSTGRES_USER = "user"
        mock_settings.POSTGRES_PASSWORD = None  # Missing password
        mock_settings.POSTGRES_HOST = "localhost"
        mock_settings.POSTGRES_PORT = "5432"
        mock_settings.POSTGRES_DB = "testdb"

        with pytest.raises(ValueError, match=r"Missing required PostgreSQL configuration"):
            self.backend.validate_config()

    @patch("langgraph_agent_toolkit.core.memory.postgres.settings")
    def test_get_connection_string(self, mock_settings):
        """Test that get_connection_string returns the correct connection string."""
        mock_settings.POSTGRES_USER = "user"
        mock_settings.POSTGRES_PASSWORD = SecretStr("password")
        mock_settings.POSTGRES_HOST = "localhost"
        mock_settings.POSTGRES_PORT = "5432"
        mock_settings.POSTGRES_DB = "testdb"

        expected = "postgresql://user:password@localhost:5432/testdb"
        self.assertEqual(self.backend.get_connection_string(), expected)


@pytest.mark.asyncio
class TestPostgresAsyncFunctionality:
    """Test the async functionality of PostgresMemoryBackend."""

    @pytest.fixture
    def backend(self):
        """Create a PostgresMemoryBackend instance."""
        return PostgresMemoryBackend()

    @patch("langgraph_agent_toolkit.core.memory.postgres.AsyncConnectionPool")
    @patch("langgraph_agent_toolkit.core.memory.postgres.AsyncPostgresSaver")
    @patch("langgraph_agent_toolkit.core.memory.postgres.settings")
    async def test_get_saver(self, mock_settings, mock_saver, mock_pool, backend):
        """Test that get_saver returns a correct saver."""
        # Configure mocks
        mock_settings.POSTGRES_USER = "user"
        mock_settings.POSTGRES_PASSWORD = SecretStr("password")
        mock_settings.POSTGRES_HOST = "localhost"
        mock_settings.POSTGRES_PORT = "5432"
        mock_settings.POSTGRES_DB = "testdb"
        mock_settings.POSTGRES_MIN_SIZE = 1
        mock_settings.POSTGRES_POOL_SIZE = 5
        mock_settings.POSTGRES_MAX_IDLE = 10

        # Setup AsyncContextManager mock for connection pool
        mock_pool_instance = AsyncMock()
        mock_pool.return_value.__aenter__.return_value = mock_pool_instance

        # Setup saver mock
        mock_saver_instance = MagicMock()
        mock_saver.return_value = mock_saver_instance

        # Test the context manager
        async with backend.get_saver() as saver:
            # Check that we got the expected saver
            assert saver == mock_saver_instance
            mock_saver.assert_called_once_with(conn=mock_pool_instance)

        # Verify that the pool was created with correct parameters
        mock_pool.assert_called_once()
        call_args = mock_pool.call_args[0][0]
        assert "postgresql://user:password@localhost:5432/testdb" == call_args

        # Verify that __aexit__ was called (pool closed)
        mock_pool.return_value.__aexit__.assert_called_once()

    @patch("langgraph_agent_toolkit.core.memory.postgres.PostgresMemoryBackend.get_saver")
    @patch("langgraph_agent_toolkit.core.memory.postgres.PostgresMemoryBackend.validate_config")
    async def test_get_checkpoint_saver(self, mock_validate, mock_get_saver, backend):
        """Test that get_checkpoint_saver calls validate_config and get_saver."""
        mock_validate.return_value = True
        mock_context_manager = MagicMock()
        mock_get_saver.return_value = mock_context_manager

        result = backend.get_checkpoint_saver()

        mock_validate.assert_called_once()
        assert result == mock_context_manager
