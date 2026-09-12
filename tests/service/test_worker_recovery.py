import importlib
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import httpx
import pytest


@pytest.mark.skipif(os.name != "posix", reason="This test uses POSIX process signals.")
@pytest.mark.parametrize("runner_name", ["uvicorn", "gunicorn"])
def test_service_runner_replaces_a_terminated_worker(tmp_path, runner_name):
    """Run two local workers and verify recovery after one worker exits."""
    importlib.import_module(runner_name)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    launcher = tmp_path / "worker_probe.py"
    launcher.write_text(
        textwrap.dedent(
            """
            import os
            import dotenv

            dotenv.find_dotenv = lambda *args, **kwargs: ""
            dotenv.load_dotenv = lambda *args, **kwargs: False

            from fastapi import FastAPI
            from langgraph_agent_toolkit.service import factory, handler

            def create_probe_app():
                app = FastAPI()

                @app.get("/worker")
                async def worker():
                    return {"pid": os.getpid()}

                return app

            factory.create_app = create_probe_app
            handler.create_app = create_probe_app

            if __name__ == "__main__":
                import sys
                runner = factory.ServiceRunner()
                if sys.argv[1] == "uvicorn":
                    # Coverage can make imports slower than Uvicorn's first heartbeat.
                    runner.run_uvicorn(host="127.0.0.1", port=int(sys.argv[2]), workers=2,
                                       reload=False, timeout_worker_healthcheck=10)
                else:
                    runner.run_gunicorn(bind="127.0.0.1:" + sys.argv[2], workers=2,
                                        timeout=10, graceful_timeout=2)
            """
        )
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["USE_FAKE_MODEL"] = "true"
    env["LANGGRAPH_USE_FAKE_MODEL"] = "true"
    logfile = tmp_path / "worker.log"
    with logfile.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(launcher), runner_name, str(port)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            with httpx.Client(timeout=0.5, trust_env=False, limits=httpx.Limits(max_keepalive_connections=0)) as client:

                def collect_workers(expected, deadline, excluded=frozenset()):
                    pids = set()
                    while time.monotonic() < deadline:
                        assert process.poll() is None, logfile.read_text()
                        try:
                            response = client.get(f"http://127.0.0.1:{port}/worker")
                            response.raise_for_status()
                            pid = response.json()["pid"]
                            if pid not in excluded:
                                pids.add(pid)
                            if len(pids) >= expected:
                                return pids
                        except httpx.HTTPError:
                            pass
                        time.sleep(0.05)
                    pytest.fail(f"Expected {expected} live workers, found {pids}.\n{logfile.read_text()}")

                before = collect_workers(2, time.monotonic() + 35)
                victim = min(before)
                os.kill(victim, signal.SIGKILL)
                after = collect_workers(2, time.monotonic() + 25, excluded={victim})
                assert victim not in after
                assert len(after - before) == 1
                for _ in range(10):
                    response = client.get(f"http://127.0.0.1:{port}/worker")
                    assert response.status_code == 200
                    assert response.json()["pid"] in after
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
