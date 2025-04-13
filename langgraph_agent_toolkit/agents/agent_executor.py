import importlib
from typing import List, Dict, Optional, Any, Callable, TypeVar, AsyncGenerator, Tuple
import os
import joblib
import functools
import asyncio
from uuid import UUID, uuid4

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel
from langgraph.types import Command, Interrupt
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.schema import AgentInfo, ChatMessage
from langgraph_agent_toolkit.helper.constants import DEFAULT_AGENT, DEFAULT_RECURSION_LIMIT
from langgraph_agent_toolkit.helper.logging import logger
from langgraph_agent_toolkit.helper.utils import (
    langchain_to_chat_message,
    remove_tool_calls,
    convert_message_content_to_string,
)


# Type variable for the decorator
T = TypeVar("T")


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

    def handle_agent_errors(func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to handle errors occurring during agent execution.
        Specifically handles GraphRecursionError and other exceptions.

        Args:
            func: The function to decorate

        Returns:
            The decorated function
        """

        def _handle_error(e: Exception):
            """Internal helper to handle and re-raise errors with logging."""
            if isinstance(e, GraphRecursionError):
                logger.error(f"GraphRecursionError occurred: {e}")
            else:
                logger.error(f"Error during agent execution: {e}")
            # Re-raise the exception to be handled by the service
            raise e

        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                return _handle_error(e)

        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                return _handle_error(e)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    async def _setup_agent_execution(
        self,
        agent_id: str,
        message: str,
        thread_id: Optional[str] = None,
        model: Optional[str] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        recursion_limit: Optional[int] = None,
    ) -> Tuple[Agent, Any, Any, UUID]:
        """
        Common setup for agent execution that both invoke and stream methods share.

        Args:
            agent_id: ID of the agent to invoke
            message: User message to send to the agent
            thread_id: Optional thread ID for conversation history
            model: Optional model name to override the default
            agent_config: Optional additional configuration for the agent
            recursion_limit: Optional recursion limit for the agent

        Returns:
            Tuple containing:
                - agent: The Agent instance
                - input_data: The properly formatted input for the agent
                - config: The RunnableConfig for the agent
                - run_id: The UUID for this run
        """
        agent = self.get_agent(agent_id)
        agent_graph = agent.graph

        run_id = uuid4()
        thread_id = thread_id or str(uuid4())

        recursion_limit = recursion_limit or DEFAULT_RECURSION_LIMIT

        configurable = {
            "thread_id": thread_id,
        }

        if model:
            configurable["model"] = model

        if agent_config:
            configurable.update(agent_config)

        callback = agent.observability.get_callback_handler(session_id=thread_id)

        config = RunnableConfig(
            configurable=configurable,
            run_id=run_id,
            callbacks=[callback] if callback else None,
            recursion_limit=recursion_limit,
        )

        # Check if there are any interrupts that need to be resumed
        state = await agent_graph.aget_state(config=config)
        interrupted_tasks = [task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts]

        if interrupted_tasks:
            # User input is a response to resume agent execution from interrupt
            input_data = Command(resume=message)
        else:
            input_data = {"messages": [HumanMessage(content=message)]}

        return agent, input_data, config, run_id

    @handle_agent_errors
    async def invoke(
        self,
        agent_id: str,
        message: str,
        thread_id: Optional[str] = None,
        model: Optional[str] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        recursion_limit: Optional[int] = None,
    ) -> ChatMessage:
        """
        Invoke an agent with a message and return the response.

        Args:
            agent_id: ID of the agent to invoke
            message: User message to send to the agent
            thread_id: Optional thread ID for conversation history
            model: Optional model name to override the default
            agent_config: Optional additional configuration for the agent
            recursion_limit: Optional recursion limit for the agent

        Returns:
            ChatMessage: The agent's response
        """
        agent, input_data, config, run_id = await self._setup_agent_execution(
            agent_id=agent_id,
            message=message,
            thread_id=thread_id,
            model=model,
            agent_config=agent_config,
            recursion_limit=recursion_limit,
        )

        # Invoke the agent
        response_events = await agent.graph.ainvoke(input=input_data, config=config, stream_mode=["updates", "values"])

        response_type, response = response_events[-1]
        if response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
        elif response_type == "updates" and "__interrupt__" in response:
            # The last thing to occur was an interrupt
            # Return the value of the first interrupt as an AIMessage
            output = langchain_to_chat_message(AIMessage(content=response["__interrupt__"][0].value))
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        return output

    @handle_agent_errors
    async def stream(
        self,
        agent_id: str,
        message: str,
        thread_id: Optional[str] = None,
        model: Optional[str] = None,
        stream_tokens: bool = True,
        agent_config: Optional[Dict[str, Any]] = None,
        recursion_limit: Optional[int] = None,
    ) -> AsyncGenerator[str | ChatMessage, None]:
        """
        Stream an agent's response to a message, yielding either tokens or messages.

        Args:
            agent_id: ID of the agent to invoke
            message: User message to send to the agent
            thread_id: Optional thread ID for conversation history
            model: Optional model name to override the default
            stream_tokens: Whether to stream individual tokens
            agent_config: Optional additional configuration for the agent
            recursion_limit: Optional recursion limit for the agent

        Yields:
            Either ChatMessage objects for full messages or strings for token chunks
        """
        agent, input_data, config, run_id = await self._setup_agent_execution(
            agent_id=agent_id,
            message=message,
            thread_id=thread_id,
            model=model,
            agent_config=agent_config,
            recursion_limit=recursion_limit,
        )

        # Stream from the agent with appropriate modes
        stream_mode = ["updates", "messages", "custom"] if stream_tokens else ["updates"]

        async for stream_event in agent.graph.astream(input=input_data, config=config, stream_mode=stream_mode):
            if not isinstance(stream_event, tuple):
                continue

            stream_mode, event = stream_event

            if stream_mode == "updates":
                for node, updates in event.items():
                    # Handle agent interrupts
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            chat_message = langchain_to_chat_message(AIMessage(content=interrupt.value))
                            chat_message.run_id = str(run_id)
                            yield chat_message
                        continue

                    update_messages = updates.get("messages", [])

                    # Special case for supervisor agent
                    if node == "supervisor":
                        # Get only the last AIMessage since supervisor includes all previous messages
                        ai_messages = [msg for msg in update_messages if isinstance(msg, AIMessage)]
                        if ai_messages:
                            update_messages = [ai_messages[-1]]

                    # Special case for expert agents
                    if node in ("research_expert", "math_expert"):
                        # Convert to ToolMessage so it displays in the UI as a tool response
                        msg = ToolMessage(
                            content=update_messages[0].content,
                            name=node,
                            tool_call_id="",
                        )
                        update_messages = [msg]

                    for message in update_messages:
                        chat_message = langchain_to_chat_message(message)
                        chat_message.run_id = str(run_id)
                        yield chat_message

            elif stream_mode == "custom":
                chat_message = langchain_to_chat_message(event)
                chat_message.run_id = str(run_id)
                yield chat_message

            elif stream_mode == "messages" and stream_tokens:
                msg, metadata = event

                if "skip_stream" in metadata.get("tags", []):
                    continue

                # Skip non-LLM nodes that might send messages
                if not isinstance(msg, AIMessageChunk):
                    continue

                content = remove_tool_calls(msg.content)

                if content:
                    # Empty content in OpenAI context usually means the model is asking for a tool to be invoked
                    yield convert_message_content_to_string(content)

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
