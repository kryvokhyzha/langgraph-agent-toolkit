"""Manage an isolated Phoenix server for explicit integration tests."""

import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import fire
import httpx


ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "tests/integration/phoenix.compose.yaml"


def _directory(directory: str) -> Path:
    path = Path(directory).resolve()
    if not path.is_relative_to(ROOT / "data"):
        raise ValueError("Keep generated test files under the repository's ignored data directory.")
    return path


def prepare(directory: str = "data/phoenix-test", server_version: str = "20.26.1") -> dict[str, str]:
    """Create a new local test stack without reading application settings."""
    if not re.fullmatch(r"\d+\.\d+\.\d+", server_version):
        raise ValueError("Select an exact stable Phoenix server version.")
    path = _directory(directory)
    path.mkdir(parents=True, exist_ok=True)
    environment = path / "server.env"
    metadata = path / "stack.json"
    if environment.exists() or metadata.exists():
        raise FileExistsError("This test directory already has a stack. Select a new directory.")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    values = {
        "LAT_TEST_PHOENIX_SERVER_VERSION": server_version,
        "LAT_TEST_PHOENIX_PORT": str(port),
        "LAT_TEST_PHOENIX_BASE_URL": f"http://127.0.0.1:{port}",
        "LAT_TEST_PHOENIX_STACK_DIRECTORY": str(path),
    }
    with environment.open("x") as output:
        environment.chmod(0o600)
        output.write("".join(f"{key}={value}\n" for key, value in values.items()))
    project = "lat-px-test-" + uuid4().hex[:12]
    metadata.write_text(json.dumps({"project": project, "values": values}, indent=2) + "\n")
    return {"project": project, "environment_file": str(environment)}


def _command(directory: str) -> tuple[Path, dict[str, str], list[str]]:
    path = _directory(directory)
    metadata = json.loads((path / "stack.json").read_text())
    project = metadata.get("project")
    if not isinstance(project, str) or not re.fullmatch(r"lat-px-test-[0-9a-f]{12}", project):
        raise ValueError("The directory does not identify a managed test stack.")
    environment = path / "server.env"
    if not environment.is_file() or environment.is_symlink():
        raise ValueError("The managed test environment file is missing or is a symbolic link.")
    command = ["docker", "compose", "--project-name", project, "--env-file", str(environment), "-f", str(COMPOSE)]
    return path, metadata["values"], command


def _run(path: Path, command: list[str], name: str) -> None:
    log = path / name
    with log.open("w") as output:
        log.chmod(0o600)
        environment = {key: value for key, value in os.environ.items() if not key.startswith("LAT_TEST_PHOENIX_")}
        result = subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"The test stack command failed. Read {log}.")


def _ready(base_url: str) -> None:
    deadline = time.monotonic() + 90
    with httpx.Client(timeout=2, trust_env=False) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(f"{base_url}/healthz")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
    raise RuntimeError("The isolated Phoenix server did not become ready within 90 seconds.")


def start(directory: str = "data/phoenix-test") -> str:
    """Start the isolated server and wait for its HTTP health endpoint."""
    path, values, command = _command(directory)
    _run(path, [*command, "up", "--detach"], "start.log")
    _ready(values["LAT_TEST_PHOENIX_BASE_URL"])
    return f"The test server is ready. Load test variables from {path / 'server.env'}."


def restart(directory: str = "data/phoenix-test") -> str:
    """Restart the server without deleting its SQLite volume."""
    path, values, command = _command(directory)
    _run(path, [*command, "restart", "phoenix"], "restart.log")
    _ready(values["LAT_TEST_PHOENIX_BASE_URL"])
    return "The test server restarted. Its database volume remains available."


def stop(directory: str = "data/phoenix-test") -> str:
    """Collect logs and delete only the generated test stack and its volume."""
    path, _, command = _command(directory)
    _run(path, [*command, "logs", "--no-color"], "services.log")
    _run(path, [*command, "down", "--volumes", "--remove-orphans"], "stop.log")
    return f"The isolated test stack was removed. Logs remain in {path}."


if __name__ == "__main__":
    fire.Fire({"prepare": prepare, "start": start, "restart": restart, "stop": stop})
