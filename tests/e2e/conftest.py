"""Start a real toolkit API process for each user journey."""

import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


BOOTSTRAP = """
import os
if os.environ.get("COVERAGE_PROCESS_START") or os.environ.get("COVERAGE_PROCESS_CONFIG"):
    import coverage
    if coverage.Coverage.current() is None:
        coverage.process_startup()
import dotenv
dotenv.find_dotenv = lambda *args, **kwargs: ""
dotenv.load_dotenv = lambda *args, **kwargs: False
import runpy
runpy.run_module("langgraph_agent_toolkit.run_api", run_name="__main__")
"""


class ApiProcess:
    """Own one API process and its disposable SQLite database."""

    def __init__(self, directory: Path):
        if importlib.util.find_spec("uvicorn") is None:
            pytest.fail("Real API tests require the uvicorn-backend extra.")
        with socket.socket() as listener:
            try:
                listener.bind(("127.0.0.1", 0))
            except OSError as exc:
                pytest.fail(f"Cannot bind localhost for the real API tests: {exc}")
            self.port = listener.getsockname()[1]
        self.directory = directory
        self.database = directory / "checkpoints.sqlite"
        self.logfile = directory / "api.log"
        self.url = f"http://127.0.0.1:{self.port}"
        self.secret = "test-client-credential"
        self.process = None
        self.log = None
        repository = Path(__file__).resolve().parents[2]
        retained = {"PATH", "HOME", "USER", "LANG", "LC_ALL", "SYSTEMROOT", "SystemRoot", "TMPDIR", "TEMP", "TMP"}
        self.env = {key: value for key, value in os.environ.items() if key in retained}
        self.env.update(
            PYTHONPATH=os.pathsep.join([str(repository), str(Path(__file__).resolve().parent)]),
            PYTHONUNBUFFERED="1",
            PYTHON_DOTENV_DISABLED="1",
            ENV_MODE="production",
            AUTH_MODE="trusted",
            AUTH_SECRET=self.secret,
            AUTH_USERS="{}",
            USE_FAKE_MODEL="true",
            AGENT_PATHS=json.dumps(["support_agent:journey_agent"]),
            DEFAULT_AGENT="journey-agent",
            MEMORY_BACKEND="sqlite",
            SQLITE_DB_PATH=str(self.database),
            OBSERVABILITY_BACKEND="empty",
            LOG_LEVEL="WARNING",
            JSON_LOGS="false",
            COLORIZE="false",
        )
        if os.environ.get("COVERAGE_PROCESS_START"):
            self.env["COVERAGE_PROCESS_START"] = str(Path(os.environ["COVERAGE_PROCESS_START"]).resolve())
        if os.environ.get("COVERAGE_PROCESS_CONFIG"):
            self.env["COVERAGE_PROCESS_CONFIG"] = os.environ["COVERAGE_PROCESS_CONFIG"]
        if any(key in self.env for key in ("COVERAGE_PROCESS_START", "COVERAGE_PROCESS_CONFIG")):
            self.env["COVERAGE_FILE"] = str(Path(os.environ.get("COVERAGE_FILE", repository / ".coverage")).resolve())

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.secret}"}

    def start(self):
        """Run the public API entry point and wait for database readiness."""
        self.log = self.logfile.open("ab")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                BOOTSTRAP,
                "--runner_type=uvicorn",
                "--host=127.0.0.1",
                f"--port={self.port}",
                "--workers=1",
                "--reload=False",
            ],
            cwd=self.directory,
            env=self.env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        deadline = time.monotonic() + 30
        with httpx.Client(timeout=0.5, trust_env=False) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    pytest.fail(f"The API exited before readiness.\n{self.logfile.read_text()}")
                try:
                    response = client.get(f"{self.url}/health/ready")
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.05)
        pytest.fail(f"The API did not become ready.\n{self.logfile.read_text()}")

    def stop(self, *, crash=False):
        """Stop the process and close its log within a fixed deadline."""
        try:
            if self.process is None:
                return
            if self.process.poll() is None:
                if os.name == "posix":
                    os.killpg(self.process.pid, signal.SIGKILL if crash else signal.SIGTERM)
                elif crash:
                    self.process.kill()
                else:
                    self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(self.process.pid, signal.SIGKILL)
                else:
                    self.process.kill()
                self.process.wait(timeout=5)
        finally:
            if self.log is not None:
                self.log.close()

    def restart(self):
        """Kill the worker and start a new process with the same SQLite file."""
        old_pid = self.process.pid
        self.stop(crash=True)
        self.start()
        assert self.process.pid != old_pid


@pytest.fixture
def api_service(tmp_path, request):
    service = ApiProcess(tmp_path)
    service.env.update(getattr(request, "param", {}))
    try:
        service.start()
        yield service
    finally:
        service.stop()
