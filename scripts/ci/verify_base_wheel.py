"""Verify the base wheel in an isolated Python environment."""

from importlib import import_module
from importlib.util import find_spec

from langgraph_agent_toolkit.agents.blueprints.create_agent.agent import react_agent
from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.service.handler import create_app


OPTIONAL_MODULES = ("langchain_openai", "streamlit", "fastmcp", "deepagents")
EXPECTED_APP_TITLE = "LangGraph Agent API"
EXPECTED_FAKE_RESPONSE = "This is a test response from the fake model."


def verify_optional_dependencies_are_absent() -> None:
    """Verify that the base wheel does not install optional dependencies."""
    installed_modules = [module_name for module_name in OPTIONAL_MODULES if find_spec(module_name) is not None]
    if installed_modules:
        names = ", ".join(installed_modules)
        raise AssertionError(f"The base wheel installed optional modules: {names}")


def verify_deep_agents_error() -> None:
    """Verify that the Deep Agents import gives the required extra name."""
    try:
        import_module("langgraph_agent_toolkit.agents.blueprints.deep_agent.agent")
    except ImportError as error:
        if "[deepagents]" not in str(error):
            raise AssertionError("The Deep Agents import error must name the [deepagents] extra.") from error
    else:
        raise AssertionError("The Deep Agents blueprint imported without the deepagents dependency.")


def verify_base_features() -> None:
    """Verify the client, service, and fake model."""
    with AgentClient(get_info=False) as client:
        if client.agent is not None:
            raise AssertionError("AgentClient must start without a selected agent.")

    app = create_app()
    if app.title != EXPECTED_APP_TITLE:
        raise AssertionError(f"The service title is {app.title!r}. Expected {EXPECTED_APP_TITLE!r}.")

    result = react_agent.graph.invoke({"messages": [{"role": "user", "content": "hello"}]})
    response = result["messages"][-1].content
    if response != EXPECTED_FAKE_RESPONSE:
        raise AssertionError(f"The fake model returned {response!r}. Expected {EXPECTED_FAKE_RESPONSE!r}.")


def main() -> None:
    """Run the base wheel checks."""
    verify_optional_dependencies_are_absent()
    verify_deep_agents_error()
    verify_base_features()


if __name__ == "__main__":
    main()
