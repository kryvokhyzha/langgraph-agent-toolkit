"""Keep agent runtime dependencies out of client and UI imports."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", ["off", "llm", "provider"])
def test_ui_imports_without_agent_runtime_dependencies(mode, tmp_path):
    """Run the image import check with agent libraries unavailable."""
    root = Path(__file__).resolve().parents[2]
    program = """
import runpy
import sys

import dotenv

dotenv.find_dotenv = lambda *args, **kwargs: ""
for name in ("langchain", "langchain_core", "langgraph", "deepagents", "fastapi"):
    sys.modules[name] = None
runpy.run_path(sys.argv[1], run_name="__main__")
from langgraph_agent_toolkit.core.settings import settings
assert settings.TOOL_SELECTION.mode == sys.argv[2]
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(root / "scripts/ci/verify_ui_imports.py"), mode],
        cwd=tmp_path,
        env={
            "PYTHONPATH": str(root),
            "PYTHON_DOTENV_DISABLED": "1",
            "TOOL_SELECTION": json.dumps({"mode": mode}),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
