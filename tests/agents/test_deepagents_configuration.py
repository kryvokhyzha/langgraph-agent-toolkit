"""Check Deep Agents startup configuration and tool naming contracts."""

import importlib

import pytest
from langchain_core.tools import tool

from langgraph_agent_toolkit.core.settings import settings


pytest.importorskip("deepagents")

build_graph = importlib.import_module("langgraph_agent_toolkit.agents.blueprints.deep_agent.agent").build_graph


@pytest.mark.parametrize("duplicate", [False, True])
def test_supplied_tools_cannot_replace_existing_tools(duplicate):
    @tool("external_read" if duplicate else "read_file")
    def external_read(path: str) -> str:
        """Read one external document."""
        raise AssertionError("A rejected tool must not run.")

    with pytest.raises(ValueError, match="Duplicate agent tool name|cannot replace Deep Agents built-ins"):
        build_graph([external_read] * (2 if duplicate else 1))


def test_startup_uses_the_named_model_configuration(monkeypatch):
    monkeypatch.setattr(settings, "USE_FAKE_MODEL", False)
    monkeypatch.setattr(settings, "MODEL_CONFIGS", {"deep_agent": {"provider": "fake", "name": "configured-demo"}})
    monkeypatch.setattr(settings, "OPENAI_MODEL_NAME", None)
    result = build_graph().invoke({"messages": [{"role": "user", "content": "Check the configured model."}]})
    assert result["messages"][-1].content == "This is a test response from the fake model."
