from langgraph_agent_toolkit.agents.components.tools import add, multiply


def test_add():
    assert add.invoke({"a": 2, "b": 3}) == 5


def test_multiply():
    assert multiply.invoke({"a": 2, "b": 3}) == 6


def test_tool_metadata():
    assert add.name == "add"
    assert "Add two numbers" in add.description
    assert multiply.name == "multiply"
    assert "Multiply two numbers" in multiply.description
