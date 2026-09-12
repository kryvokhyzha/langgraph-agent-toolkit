import os
from unittest.mock import patch

import dotenv
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-docker", action="store_true", default=False, help="run docker integration tests")
    parser.addoption("--run-e2e", action="store_true", default=False, help="run local service process journeys")
    parser.addoption("--run-postgres", action="store_true", default=False, help="require PostgreSQL integration tests")
    parser.addoption("--run-langfuse", action="store_true", default=False, help="run against a Langfuse test project")
    parser.addoption("--run-llm", action="store_true", default=False, help="make bounded real OpenAI model requests")


def pytest_configure(config):
    # Unit tests must not read developer credentials or database paths from .env.
    patches = pytest.MonkeyPatch()
    patches.setattr(dotenv, "find_dotenv", lambda *args, **kwargs: "")
    patches.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    # Live tests use LAT_TEST_LANGFUSE_* and must not select a normal project.
    for key in tuple(os.environ):
        if key.startswith(("LANGFUSE_", "LANGGRAPH_LANGFUSE_")):
            patches.delenv(key)
    config.add_cleanup(patches.undo)
    for marker, description in {
        "integration": "run real package components or dependency protocols together",
        "e2e": "run a local API process and call its HTTP interface",
        "docker": "require running service and UI containers",
        "postgres": "require a disposable PostgreSQL database",
        "langfuse": "require an explicitly configured Langfuse test project",
        "llm": "make bounded paid requests to an explicitly configured OpenAI model",
    }.items():
        config.addinivalue_line("markers", f"{marker}: {description}")
    if config.getoption("--run-postgres") and not os.environ.get("LAT_TEST_POSTGRES_DSN"):
        raise pytest.UsageError("--run-postgres requires LAT_TEST_POSTGRES_DSN for a disposable test database.")
    if config.getoption("--run-llm"):
        required = ("LAT_TEST_OPENAI_API_KEY", "LAT_TEST_OPENAI_MODEL")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise pytest.UsageError("--run-llm requires explicit test configuration: " + ", ".join(missing))
        if not config.getoption("--run-e2e"):
            raise pytest.UsageError("--run-llm requires --run-e2e to exercise the real local API.")


def pytest_collection_modifyitems(config, items):
    for item in items:
        parts = item.path.relative_to(config.rootpath).parts
        if "integration" in parts:
            item.add_marker(pytest.mark.integration)
        if "e2e" in parts or item.path.name == "test_worker_recovery.py":
            item.add_marker(pytest.mark.e2e)
        for marker in ("docker", "e2e", "langfuse", "llm"):
            if item.get_closest_marker(marker) is not None and not config.getoption(f"--run-{marker}"):
                item.add_marker(pytest.mark.skip(reason=f"Use --run-{marker} to run this test."))


@pytest.fixture
def mock_env():
    """Clean the environment for each test."""
    with patch.dict(os.environ, {}, clear=True):
        yield


class MockStateSnapshot:
    """Lightweight `StateSnapshot` stand-in with `.values` and `.tasks`."""

    def __init__(self, values=None, tasks=None):
        self.values = values or {}
        self.tasks = tasks or []


@pytest.fixture
def mock_state_snapshot():
    """Return `MockStateSnapshot` for test state snapshots."""
    return MockStateSnapshot
