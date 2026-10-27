"""Manage an isolated Langfuse server for explicit integration tests."""

import json
import os
import re
import secrets
import socket
import subprocess
from pathlib import Path
from uuid import uuid4

import fire


ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "tests/integration/langfuse.compose.yaml"
LEGACY_COMPOSE = ROOT / "tests/integration/langfuse-legacy.compose.yaml"


def _directory(directory: str) -> Path:
    path = Path(directory).resolve()
    if not path.is_relative_to(ROOT / "data"):
        raise ValueError("Keep generated test credentials under the repository's ignored data directory.")
    return path


def prepare(
    directory: str = "data/langfuse-test", server_version: str = "4.71.0", write_mode: str = "default"
) -> dict[str, str]:
    """Generate credentials for a new test project without reading application settings."""
    if not re.fullmatch(r"[34]\.\d+\.\d+", server_version):
        raise ValueError("Select an exact stable Langfuse server v3 or v4 version.")
    if write_mode not in ("default", "legacy"):
        raise ValueError("The managed test write_mode must be default or legacy.")
    path = _directory(directory)
    path.mkdir(parents=True, exist_ok=True)
    environment = path / "server.env"
    metadata = path / "stack.json"
    if environment.exists() or metadata.exists():
        raise FileExistsError("This test directory already has credentials. Select a new directory.")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    values = {
        "LAT_TEST_LANGFUSE_SERVER_VERSION": server_version,
        "LAT_TEST_LANGFUSE_PORT": str(port),
        "LAT_TEST_LANGFUSE_BASE_URL": f"http://127.0.0.1:{port}",
        "LAT_TEST_LANGFUSE_PUBLIC_KEY": "pk-lf-" + uuid4().hex,
        "LAT_TEST_LANGFUSE_SECRET_KEY": "sk-lf-" + secrets.token_hex(24),
        "LAT_TEST_LANGFUSE_ORG_ID": uuid4().hex,
        "LAT_TEST_LANGFUSE_PROJECT_ID": uuid4().hex,
    }
    for name in (
        "DB_PASSWORD",
        "CLICKHOUSE_PASSWORD",
        "S3_PASSWORD",
        "SALT",
        "ENCRYPTION_KEY",
        "NEXTAUTH_SECRET",
        "USER_PASSWORD",
    ):
        values[f"LAT_TEST_LANGFUSE_{name}"] = secrets.token_hex(32)
    with environment.open("x") as file:
        environment.chmod(0o600)
        file.write("".join(f"{key}={value}\n" for key, value in values.items()))
    project = "lat-lf-test-" + uuid4().hex[:12]
    metadata.write_text(json.dumps({"project": project, "write_mode": write_mode}, indent=2) + "\n")
    return {"project": project, "environment_file": str(environment)}


def _command(directory: str) -> tuple[Path, list[str]]:
    path = _directory(directory)
    metadata = json.loads((path / "stack.json").read_text())
    project = metadata.get("project")
    if not isinstance(project, str) or not re.fullmatch(r"lat-lf-test-[0-9a-f]{12}", project):
        raise ValueError("The directory does not identify a managed test stack.")
    environment = path / "server.env"
    if not environment.is_file() or environment.is_symlink():
        raise ValueError("The managed test environment file is missing or is a symbolic link.")
    command = [
        "docker",
        "compose",
        "--project-name",
        project,
        "--env-file",
        str(environment),
        "-f",
        str(COMPOSE),
    ]
    write_mode = metadata.get("write_mode", "default")
    if write_mode == "legacy":
        command.extend(["-f", str(LEGACY_COMPOSE)])
    elif write_mode != "default":
        raise ValueError("The managed test write_mode must be default or legacy.")
    return path, command


def _run(path: Path, command: list[str], name: str) -> None:
    log = path / name
    with log.open("w") as output:
        log.chmod(0o600)
        environment = {key: value for key, value in os.environ.items() if not key.startswith("LAT_TEST_LANGFUSE_")}
        result = subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"The test stack command failed. Read {log}. It can contain test credentials.")


def start(directory: str = "data/langfuse-test") -> str:
    """Start the isolated stack and wait for all service health checks."""
    path, command = _command(directory)
    _run(path, [*command, "up", "--detach", "--wait", "--wait-timeout", "600"], "start.log")
    return f"The test stack is ready. Load test variables from {path / 'server.env'}."


def restart(directory: str = "data/langfuse-test") -> str:
    """Restart web and worker containers without deleting their database volumes."""
    path, command = _command(directory)
    _run(path, [*command, "restart", "web", "worker"], "restart.log")
    _run(path, [*command, "up", "--detach", "--wait", "--wait-timeout", "300"], "restart-health.log")
    return "The test web and worker restarted. Their volumes remain available."


def stop(directory: str = "data/langfuse-test") -> str:
    """Collect logs and delete only the generated test stack and its volumes."""
    path, command = _command(directory)
    _run(path, [*command, "logs", "--no-color"], "services.log")
    _run(path, [*command, "down", "--volumes", "--remove-orphans"], "stop.log")
    return f"The isolated test stack was removed. Logs remain in {path}."


if __name__ == "__main__":
    fire.Fire({"prepare": prepare, "start": start, "restart": restart, "stop": stop})
