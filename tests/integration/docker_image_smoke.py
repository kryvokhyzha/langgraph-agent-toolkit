"""Check final image contents without external services or test dependencies.

Run this script as the image user with ``--network none``. Pass ``api`` or
``app``. For ``app``, mount ``scripts/ci/verify_ui_imports.py`` beside this file.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
import runpy
import shutil
import sqlite3
import ssl
import sys
import tempfile
from contextlib import closing
from pathlib import Path


def check_common() -> dict:
    """Check permissions, the Python environment, and installed runtime files."""
    assert os.geteuid() != 0, "The image must run as a non-root user."
    assert sys.prefix == "/opt/venv", "The image must use its copied virtual environment."
    assert sys.prefix != sys.base_prefix
    assert Path(sys.executable).is_file(), "The virtual environment must have a valid Python executable."
    assert shutil.which("curl") is not None, "The image health check requires curl."
    for command in ("uv", "uvx", "gcc", "g++", "make"):
        assert shutil.which(command) is None, f"The runtime must not contain {command}."

    distributions = {dist.metadata["Name"].lower().replace("_", "-") for dist in importlib.metadata.distributions()}
    development_packages = {"pytest", "pytest-cov", "pre-commit", "ruff", "sphinx"}
    assert not distributions & development_packages, (
        f"Development packages found: {distributions & development_packages}"
    )

    source = Path("/app")
    for name in (".env", ".git", ".venv", ".cache", "pyproject.toml", "uv.lock", "tests", "docs", "data"):
        assert not (source / name).exists(), f"Unexpected runtime file: /app/{name}"
    assert (source / "langgraph_agent_toolkit").is_dir()
    assert not list(source.rglob(".env*")), "The image must not contain a source environment file."

    trust_store = ssl.create_default_context().get_ca_certs()
    assert trust_store, "HTTPS requires a populated certificate trust store."
    return {"uid": os.geteuid(), "python_prefix": sys.prefix, "trusted_certificates": len(trust_store)}


def check_api() -> dict:
    """Check selected API features and native database libraries."""
    modules = (
        "gunicorn",
        "uvicorn",
        "fastapi",
        "fastmcp",
        "langfuse",
        "langsmith",
        "openai",
        "aiohttp",
        "httpx",
        "langgraph.checkpoint.sqlite.aio",
        "langgraph.checkpoint.postgres.aio",
        "psycopg_pool",
    )
    for name in modules:
        importlib.import_module(name)

    from psycopg import pq

    assert pq.__impl__ == "binary", "The image must load the bundled PostgreSQL client."
    assert pq.version() > 0
    with tempfile.TemporaryDirectory(prefix="toolkit-image-smoke-", dir="/app") as directory:
        path = Path(directory) / "smoke.sqlite"
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("CREATE TABLE smoke (value TEXT NOT NULL)")
            connection.execute("INSERT INTO smoke VALUES (?)", ("checkpoint",))
        with closing(sqlite3.connect(path)) as connection:
            assert connection.execute("SELECT value FROM smoke").fetchone() == ("checkpoint",)
    return {"postgres_driver": pq.__impl__, "sqlite_write": True, "feature_imports": list(modules)}


def check_app() -> dict:
    """Reuse the client-only import check."""
    helper = Path(__file__).with_name("verify_ui_imports.py")
    assert helper.is_file(), "Mount scripts/ci/verify_ui_imports.py beside this script."
    runpy.run_path(str(helper), run_name="__main__")
    return {"client_only_imports": True}


def main() -> None:
    """Run the checks for the selected image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", choices=("api", "app"))
    selected = parser.parse_args().image
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    results = {"image": selected, **check_common()}
    results.update(check_api() if selected == "api" else check_app())
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
