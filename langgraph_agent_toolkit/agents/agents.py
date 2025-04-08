from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from langgraph_agent_toolkit.agents.bg_task_agent.bg_task_agent import bg_task_agent
from langgraph_agent_toolkit.agents.chatbot import chatbot
from langgraph_agent_toolkit.agents.command_agent import command_agent
from langgraph_agent_toolkit.agents.interrupt_agent import interrupt_agent
from langgraph_agent_toolkit.agents.langgraph_supervisor_agent import langgraph_supervisor_agent
from langgraph_agent_toolkit.schema import AgentInfo

DEFAULT_AGENT = "langgraph-supervisor-agent"


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "chatbot": Agent(description="A simple chatbot.", graph=chatbot),
    "command-agent": Agent(description="A command agent.", graph=command_agent),
    "bg-task-agent": Agent(description="A background task agent.", graph=bg_task_agent),
    "langgraph-supervisor-agent": Agent(description="A langgraph supervisor agent", graph=langgraph_supervisor_agent),
    "interrupt-agent": Agent(description="An agent the uses interrupts.", graph=interrupt_agent),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()]
