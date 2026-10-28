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
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path


def check_trust_store(label: str, **kwargs) -> int:
    """Reject the affected emSign root in a populated certificate store."""
    certificates = ssl.create_default_context(**kwargs).get_ca_certs()
    assert certificates, f"{label} requires certificate roots."
    names = {
        value
        for certificate in certificates
        for group in certificate["subject"]
        for key, value in group
        if key == "commonName"
    }
    assert "emSign ECC Root CA - G3" not in names, f"{label} still trusts the affected emSign root."
    return len(certificates)


def check_certificates() -> dict:
    """Check application, system, pip, and ensurepip certificate bundles."""
    import certifi

    assert tuple(map(int, certifi.__version__.split("."))) >= (2026, 10, 19)
    system_count = check_trust_store("System TLS")
    certifi_count = check_trust_store("Application certifi", cafile=certifi.where())
    # pip belongs to the base interpreter, outside the application environment.
    script = """
import ensurepip
import json
import runpy
import sys
from pathlib import Path
from zipfile import ZipFile

import pip
from pip._vendor import certifi

check = runpy.run_path(sys.argv[1])["check_trust_store"]
assert tuple(map(int, pip.__version__.split("."))) >= (26, 2, 2)
assert tuple(map(int, ensurepip.version().split("."))) >= (26, 2, 2)
check("pip certifi", cafile=certifi.where())
wheels = list(Path(ensurepip.__file__).with_name("_bundled").glob("pip-*.whl"))
assert len(wheels) == 1, "Expected one bundled pip wheel."
with ZipFile(wheels[0]) as wheel:
    check("ensurepip certifi", cadata=wheel.read("pip/_vendor/certifi/cacert.pem").decode())
print(json.dumps({"pip": pip.__version__, "ensurepip": ensurepip.version()}))
"""
    output = subprocess.check_output(
        ["/usr/local/bin/python", "-c", script, str(Path(__file__).resolve())], text=True, timeout=15
    )
    return {
        "trusted_certificates": system_count,
        "certifi": certifi.__version__,
        "certifi_roots": certifi_count,
        **json.loads(output),
    }


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

    return {"uid": os.geteuid(), "python_prefix": sys.prefix, **check_certificates()}


def check_api() -> dict:
    """Check selected API features and native database libraries."""
    modules = (
        "gunicorn",
        "uvicorn",
        "fastapi",
        "fastmcp",
        "langfuse",
        "langsmith",
        "phoenix.client",
        "openinference.instrumentation.langchain",
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
