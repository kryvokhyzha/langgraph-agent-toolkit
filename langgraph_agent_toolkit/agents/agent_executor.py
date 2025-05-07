import asyncio
import functools
import importlib
import json
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, TypeVar
from uuid import UUID, uuid4

import joblib
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel
from langgraph.types import Command, Interrupt

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.constants import DEFAULT_RECURSION_LIMIT, get_default_agent, set_default_agent
from langgraph_agent_toolkit.helper.logging import logger
from langgraph_agent_toolkit.helper.utils import (
    convert_message_content_to_string,
    create_ai_message,
    langchain_to_chat_message,
    remove_tool_calls,
)
from langgraph_agent_toolkit.schema import AgentInfo, ChatMessage


T = TypeVar("T")


class AgentExecutor:
    """Handles the loading, execution and saving logic for different LangGraph agents."""

    def __init__(self, *args):
        """Initialize the AgentExecutor by importing agents.

        Args:
            *args: Variable length strings specifying the agents to import,
                  e.g., "langgraph_agent_toolkit.agents.blueprints.react.agent:react_agent".

        Raises:
            ValueError: If no agents are provided.

        """
        self.agents: Dict[str, Agent] = {}

        if not args:
            raise ValueError("At least one agent must be provided to AgentExecutor.")

        # Load agents from import strings
        self.load_agents_from_imports(args)
        self._validate_default_agent_loaded()

    def load_agents_from_imports(self, args: tuple) -> None:
        """Dynamically imports agents based on the provided import strings."""
        for import_str in args:
            try:
                module_path, object_name = import_str.split(":")
                module = importlib.import_module(module_path)
                agent_obj = getattr(module, object_name)

                if isinstance(agent_obj, (CompiledStateGraph, Pregel)):
                    agent = Agent(name=object_name, description=f"Dynamically loaded {object_name}", graph=agent_obj)
                    self.agents[agent.name] = agent
                elif isinstance(agent_obj, Agent):
                    self.agents[agent_obj.name] = agent_obj
                else:
                    logger.warning(f"Object '{object_name}' is neither a graph nor an Agent instance")
            except (ImportError, AttributeError, ValueError) as e:
                logger.error(f"Error loading agent from '{import_str}': {e}")

    def _validate_default_agent_loaded(self) -> None:
        """Validate that a default agent is available and set it if needed.

        If the default agent from constants.py is not available,
        use the first loaded agent as the default and update the global value.
        """
        if not self.agents:
            raise ValueError("No agents were loaded. Please check your imports.")

        initial_default = get_default_agent()

        if initial_default not in self.agents:
            new_default = list(self.agents.keys())[0]
            logger.warning(
                f"Default agent '{initial_default}' not found in loaded agents. Using '{new_default}' as default."
            )
            set_default_agent(new_default)

    def get_agent(self, agent_id: str) -> Agent:
        """Get an agent by its ID.

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
        """Get information about all available agents.

        Returns:
            A list of AgentInfo objects containing agent IDs and descriptions

        """
        return [AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in self.agents.items()]

    def add_agent(self, agent_id: str, agent: Agent) -> None:
        """Add a new agent to the executor.

        Args:
            agent_id: The ID to assign to the agent
            agent: The Agent instance to add

        """
        self.agents[agent_id] = agent

    @staticmethod
    def handle_agent_errors(func: Callable[..., T]) -> Callable[..., T]:
        """Handle errors occurring during agent execution.

        Specifically handles GraphRecursionError and other exceptions.

        Args:
            func: The function to decorate

        Returns:
            The decorated function

        """

        def _handle_error(e: Exception):
            """Handle and re-raise errors with logging."""
            if isinstance(e, GraphRecursionError):
                logger.error(f"GraphRecursionError occurred: {e}")
            else:
                logger.error(f"Error during agent execution: {e}")

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
        user_id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_provider: Optional[str] = None,
        model_config_key: Optional[str] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        recursion_limit: Optional[int] = None,
    ) -> Tuple[Agent, Any, Any, UUID]:
        """Apply common setup for agent execution that both invoke and stream methods share.

        Args:
            agent_id: ID of the agent to invoke
            message: User message to send to the agent
            thread_id: Optional thread ID for conversation history
            user_id: Optional user ID for the agent
            model_name: Optional model name to override the default
            model_provider: Optional model provider to override the default
            model_config_key: Optional model config key to override the default
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
            "user_id": user_id,
        }

        # Handle model_config_key if provided (takes precedence over individual model settings)
        if model_config_key and model_config_key in settings.MODEL_CONFIGS:
            # Store the model_config_key so agents can use it if needed
            configurable["model_config_key"] = model_config_key

            # Extract basic model info for backward compatibility with agents that
            # don't explicitly check for model_config_key
            model_config = settings.MODEL_CONFIGS[model_config_key]
            if "provider" in model_config:
                configurable["model_provider"] = model_config["provider"]
            if "name" in model_config:
                configurable["model_name"] = model_config["name"]
        else:
            # Fall back to individual parameters
            if model_name:
                configurable["model_name"] = model_name

            if model_provider:
                configurable["model_provider"] = model_provider

        if agent_config:
            configurable.update(agent_config)

        callback = agent.observability.get_callback_handler(session_id=thread_id, user_id=user_id)

        config = RunnableConfig(
            configurable=configurable,
            run_id=run_id,
            callbacks=[callback] if callback else None,
            recursion_limit=recursion_limit,
        )

        # Check if there are any interrupts that need to be resumed
        state = await agent_graph.aget_state(config=config)
        interrupted_tasks = [task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts]

        input_data: Command | dict[str, Any]
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
        user_id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_provider: Optional[str] = None,
        model_config_key: Optional[str] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        recursion_limit: Optional[int] = None,
    ) -> ChatMessage:
        """Invoke an agent with a message and return the response.

        Args:
            agent_id: ID of the agent to invoke
            message: User message to send to the agent
            thread_id: Optional thread ID for conversation history
            user_id: Optional user ID for the agent
            model_name: Optional model name to override the default
            model_provider: Optional model provider to override the default
            model_config_key: Optional model config key to override the default
            agent_config: Optional additional configuration for the agent
            recursion_limit: Optional recursion limit for the agent

        Returns:
            ChatMessage: The agent's response

        """
        agent, input_data, config, run_id = await self._setup_agent_execution(
            agent_id=agent_id,
            message=message,
            thread_id=thread_id,
            user_id=user_id,
            model_name=model_name,
            model_provider=model_provider,
            model_config_key=model_config_key,  # Pass this parameter
            agent_config=agent_config,
            recursion_limit=recursion_limit,
        )

        # Invoke the agent
        response_events: list[tuple[str, Any]] = await agent.graph.ainvoke(
            input=input_data,
            config=config,
            stream_mode=["updates", "values"],
        )

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
        user_id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_provider: Optional[str] = None,
        model_config_key: Optional[str] = None,
        stream_tokens: bool = True,
        agent_config: Optional[Dict[str, Any]] = None,
        recursion_limit: Optional[int] = None,
    ) -> AsyncGenerator[str | ChatMessage, None]:
        """Stream an agent's response to a message, yielding either tokens or messages.

        Args:
            agent_id: ID of the agent to invoke
            message: User message to send to the agent
            thread_id: Optional thread ID for conversation history
            user_id: Optional user ID for the agent
            model_name: Optional model name to override the default
            model_provider: Optional model provider to override the default
            model_config_key: Optional model config key to override the default
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
            user_id=user_id,
            model_name=model_name,
            model_provider=model_provider,
            model_config_key=model_config_key,
            agent_config=agent_config,
            recursion_limit=recursion_limit,
        )

        # Stream from the agent with appropriate modes
        stream_mode = ["updates", "messages", "custom"] if stream_tokens else ["updates"]

        async for stream_event in agent.graph.astream(input=input_data, config=config, stream_mode=stream_mode):
            if not isinstance(stream_event, tuple):
                continue

            stream_mode, event = stream_event
            new_messages = []

            if stream_mode == "updates":
                for node, updates in event.items():
                    # A simple approach to handle agent interrupts.
                    # In a more sophisticated implementation, we could add
                    # some structured ChatMessage type to return the interrupt value.
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue

                    update_messages = (updates or {}).get("messages", [])

                    # Special case for supervisor agent
                    if node == "supervisor":
                        # Get only the last AIMessage since supervisor includes all previous messages
                        ai_messages = [msg for msg in update_messages if isinstance(msg, AIMessage)]
                        if ai_messages:
                            update_messages = [ai_messages[-1]]

                    # Special case for expert agents
                    if node in ("research_expert", "math_expert"):
                        # Convert to ToolMessage so it displays in the UI as a tool response
                        if update_messages:
                            msg = ToolMessage(
                                content=update_messages[0].content,
                                name=node,
                                tool_call_id="",
                            )
                            update_messages = [msg]
                    new_messages.extend(update_messages)

            elif stream_mode == "custom":
                new_messages = [event]

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

            # LangGraph streaming may emit tuples: (field_name, field_value)
            # e.g. ('content', <str>), ('tool_calls', [ToolCall,...]), ('additional_kwargs', {...}), etc.
            # We accumulate only supported fields into `parts` and skip unsupported metadata.
            # More info at: https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/
            processed_messages = []
            current_message: dict[str, Any] = {}
            for msg in new_messages:
                if isinstance(msg, tuple):
                    key, value = msg
                    # Store parts in temporary dict
                    current_message[key] = value
                else:
                    # Add complete message if we have one in progress
                    if current_message:
                        processed_messages.append(create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(msg)

            # Add any remaining message parts
            if current_message:
                processed_messages.append(create_ai_message(current_message))

            for msg in new_messages:
                try:
                    chat_message = langchain_to_chat_message(msg)
                    chat_message.run_id = str(run_id)
                    # Skip the input message if it's repeated by LangGraph
                    if chat_message.type == "human" and chat_message.content == msg:
                        continue
                    yield chat_message
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    continue

    def save(self, path: str, agent_ids: Optional[List[str]] = None) -> None:
        """Save agents to disk using joblib.

        Args:
            path: Directory path where to save agents
            agent_ids: List of agent IDs to save. If None, saves all agents.

        """
        _path = Path(path)
        _path.mkdir(exist_ok=True, parents=True)

        agents_to_save = self.agents
        if agent_ids:
            agents_to_save = {k: v for k, v in self.agents.items() if k in agent_ids}

        for agent_id, agent in agents_to_save.items():
            joblib.dump(agent, _path / f"{agent_id}.joblib")

    def load_saved_agents(self, path: str) -> None:
        """Load saved agents from disk using joblib.

        Args:
            path: Directory path from which to load agents

        """
        for filename in os.listdir(path):
            if filename.endswith(".joblib"):
                agent = joblib.load(os.path.join(path, filename))
                self.agents[agent.name] = agent

        self._validate_default_agent_loaded()
