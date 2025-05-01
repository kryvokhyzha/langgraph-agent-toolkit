import fire
from dotenv import load_dotenv

from langgraph_agent_toolkit.service.factory import RunnerType, ServiceRunner


load_dotenv(override=True)


def run_service(
    runner_type: str = "uvicorn",
    **kwargs,
):
    """Run the service with the specified runner type.

    Args:
        runner_type (str): The type of runner to use.
        **kwargs: Additional arguments to pass to the service runner.

    """
    runner_type = RunnerType(runner_type)

    service = ServiceRunner(
        custom_settings=dict(
            AGENT_PATHS=[
                "langgraph_agent_toolkit.agents.blueprints.react.agent:react_agent",
                "langgraph_agent_toolkit.agents.blueprints.supervisor_agent.agent:supervisor_agent",
                "langgraph_agent_toolkit.agents.blueprints.chatbot.agent:chatbot_agent",
            ]
        ),
    )
    _ = service.run(runner_type=runner_type, **kwargs)


if __name__ == "__main__":
    fire.Fire(run_service)
