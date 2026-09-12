import fire
from dotenv import load_dotenv


def run_api(
    runner_type: str = "uvicorn",
    **kwargs,
):
    """Run the service with the selected runner type.

    Args:
        runner_type (str): Runner type.
        **kwargs: Arguments for the service runner.

    """
    from langgraph_agent_toolkit.service.utils import setup_logging

    setup_logging()

    from langgraph_agent_toolkit.service.factory import RunnerType, ServiceRunner

    runner_type = RunnerType(runner_type)

    service = ServiceRunner()
    _ = service.run(runner_type=runner_type, **kwargs)


if __name__ == "__main__":
    load_dotenv(override=False)

    fire.Fire(run_api)
