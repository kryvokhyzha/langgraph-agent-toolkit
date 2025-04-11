import importlib
from typing import List, Dict, Optional
import os
import joblib

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.schema import AgentInfo
from langgraph_agent_toolkit.helper.constants import DEFAULT_AGENT


class AgentExecutor:
    """
    Handles the loading, execution and saving logic for different LangGraph agents.
    """

    def __init__(self, *args):
        """
        Initializes the AgentExecutor by importing agents.

        Args:
            *args: Variable length strings specifying the agents to import,
                  e.g., "langgraph_agent_toolkit.agents.blueprints.react.agent:react_agent".

        Raises:
            ValueError: If no agents are provided.
        """
        self.agents: Dict[str, Agent] = {}

        # Check if args is empty and raise an error
        if not args:
            raise ValueError("At least one agent must be provided to AgentExecutor.")

        # Load agents from import strings
        self.load_agents_from_imports(args)
        self._validate_default_agent_loaded()

    def load_agents_from_imports(self, args: tuple) -> None:
        """
        Dynamically imports agents based on the provided import strings.
        """
        for import_str in args:
            try:
                module_path, object_name = import_str.split(":")
                module = importlib.import_module(module_path)
                agent_obj = getattr(module, object_name)

                # Check if it's a raw graph or already an Agent instance
                if isinstance(agent_obj, (CompiledStateGraph, Pregel)):
                    agent = Agent(name=object_name, description=f"Dynamically loaded {object_name}", graph=agent_obj)
                    self.agents[agent.name] = agent
                elif isinstance(agent_obj, Agent):
                    self.agents[agent_obj.name] = agent_obj
                else:
                    print(f"Warning: Object '{object_name}' is neither a graph nor an Agent instance")
            except (ImportError, AttributeError, ValueError) as e:
                print(f"Error loading agent from '{import_str}': {e}")

    def _validate_default_agent_loaded(self) -> None:
        """
        Validates that the default agent is loaded.

        Raises:
            ValueError: If the default agent is not loaded
        """
        if not self.agents or DEFAULT_AGENT not in self.agents:
            raise ValueError(
                f"Default agent '{DEFAULT_AGENT}' was not imported. Make sure to include it in your agent imports."
            )

    def get_agent(self, agent_id: str) -> Agent:
        """
        Gets an agent by its ID.

        Args:
            agent_id: The ID of the agent to retrieve

        Returns:
            The requested Agent instance

        Raises:
            KeyError: If the agent_id is not found
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent '{agent_id}' not found")
        return self.agents[agent_id]

    def get_all_agent_info(self) -> list[AgentInfo]:
        """
        Gets information about all available agents.

        Returns:
            A list of AgentInfo objects containing agent IDs and descriptions
        """
        return [AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in self.agents.items()]

    def add_agent(self, agent_id: str, agent: Agent) -> None:
        """
        Adds a new agent to the executor.

        Args:
            agent_id: The ID to assign to the agent
            agent: The Agent instance to add
        """
        self.agents[agent_id] = agent

    def save(self, path: str, agent_ids: Optional[List[str]] = None) -> None:
        """
        Saves agents to disk using joblib.

        Args:
            path: Directory path where to save agents
            agent_ids: List of agent IDs to save. If None, saves all agents.
        """
        os.makedirs(path, exist_ok=True)

        agents_to_save = self.agents
        if agent_ids:
            agents_to_save = {k: v for k, v in self.agents.items() if k in agent_ids}

        for agent_id, agent in agents_to_save.items():
            joblib.dump(agent, os.path.join(path, f"{agent_id}.joblib"))

    def load_saved_agents(self, path: str) -> None:
        """
        Loads saved agents from disk using joblib.

        Args:
            path: Directory path from which to load agents
        """
        for filename in os.listdir(path):
            if filename.endswith(".joblib"):
                agent = joblib.load(os.path.join(path, filename))
                self.agents[agent.name] = agent

        self._validate_default_agent_loaded()
