from dotenv import load_dotenv

from langgraph_agent_toolkit.service.factory import RunnerType, ServiceRunner


load_dotenv(override=True)

if __name__ == "__main__":
    # Create and run the service with the ServiceRunner
    service = ServiceRunner(
        custom_settings=dict(
            AGENT_PATHS=[
                "langgraph_agent_toolkit.agents.blueprints.react.agent:react_agent",
                "langgraph_agent_toolkit.agents.blueprints.supervisor_agent.agent:supervisor_agent",
                "langgraph_agent_toolkit.agents.blueprints.chatbot.agent:chatbot_agent",
            ]
        ),
    )
    handler = service.run(runner_type=RunnerType.UVICORN)
