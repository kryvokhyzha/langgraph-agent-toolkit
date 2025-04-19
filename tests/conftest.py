import os
from unittest.mock import patch

import httpx
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-docker", action="store_true", default=False, help="run docker integration tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: mark test as requiring docker containers")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-docker"):
        skip_docker = pytest.mark.skip(reason="need --run-docker option to run")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)


@pytest.fixture
def mock_env():
    """Fixture to ensure environment is clean for each test."""
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.fixture
def check_service_available():
    """Fixture to check if a service is available at a given URL."""

    def _check(url, timeout=5):
        try:
            response = httpx.get(f"{url}/info", timeout=timeout)
            return response.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            return False

    return _check
