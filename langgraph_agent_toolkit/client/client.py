import json
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

import httpx

from langgraph_agent_toolkit.schema import (
    AddMessagesInput,
    AddMessagesResponse,
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    ClearHistoryInput,
    ClearHistoryResponse,
    Feedback,
    FeedbackResponse,
    MessageInput,
    ServiceMetadata,
    StreamInput,
    UserInput,
)


class AgentClientError(Exception):
    pass


class AgentClient:
    """Client for interacting with the agent service."""

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | None = None,
        get_info: bool = True,
        verify: bool = False,
    ) -> None:
        """Initialize the client.

        Args:
            base_url (str): The base URL of the agent service.
            agent (str): The name of the default agent to use.
            timeout (float, optional): The timeout for requests.
            get_info (bool, optional): Whether to fetch agent information on init.
                Default: True
            verify (bool, optional): Whether to verify the agent information.
                Default: False

        """
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout
        self.info: ServiceMetadata | None = None
        self.agent: str | None = None
        if get_info:
            self.retrieve_info()
        if agent:
            self.update_agent(agent, verify=verify)

    @property
    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    def retrieve_info(self) -> None:
        try:
            response = httpx.get(
                f"{self.base_url}/info",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error getting service info: {e}")

        self.info = ServiceMetadata.model_validate(response.json())
        if not self.agent or self.agent not in [a.key for a in self.info.agents]:
            self.agent = self.info.default_agent

    def update_agent(self, agent: str, verify: bool = True) -> None:
        if verify:
            if not self.info:
                self.retrieve_info()
            agent_keys = [a.key for a in self.info.agents]
            if agent not in agent_keys:
                raise AgentClientError(f"Agent {agent} not found in available agents: {', '.join(agent_keys)}")
        self.agent = agent

    async def ainvoke(
        self,
        message: str,
        model_name: str | None = None,
        model_provider: str | None = None,
        model_config_key: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        recursion_limit: int | None = None,
    ) -> ChatMessage:
        """Invoke the agent asynchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            model_name (str, optional): LLM model to use for the agent
            model_provider (str, optional): LLM model provider to use for the agent
            model_config_key (str, optional): Key for predefined model configuration
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for identifying the user
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            recursion_limit (int, optional): Recursion limit for the agent

        Returns:
            ChatMessage: The response from the agent

        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")

        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model_name:
            request.model_name = model_name
        if model_provider:
            request.model_provider = model_provider
        if model_config_key:
            request.model_config_key = model_config_key
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        if recursion_limit is not None:
            request.recursion_limit = recursion_limit

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/{self.agent}/invoke",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def invoke(
        self,
        message: str,
        model_name: str | None = None,
        model_provider: str | None = None,
        model_config_key: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        recursion_limit: int | None = None,
    ) -> ChatMessage:
        """Invoke the agent synchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            model_name (str, optional): LLM model to use for the agent
            model_provider (str, optional): LLM model provider to use for the agent
            model_config_key (str, optional): Key for predefined model configuration
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for identifying the user
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            recursion_limit (int, optional): Recursion limit for the agent

        Returns:
            ChatMessage: The response from the agent

        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")

        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model_name:
            request.model_name = model_name
        if model_provider:
            request.model_provider = model_provider
        if model_config_key:
            request.model_config_key = model_config_key
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        if recursion_limit is not None:
            request.recursion_limit = recursion_limit

        try:
            response = httpx.post(
                f"{self.base_url}/{self.agent}/invoke",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def _parse_stream_line(self, line: str) -> ChatMessage | str | None:
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise Exception(f"Error JSON parsing message from server: {e}")
            match parsed["type"]:
                case "message":
                    # Convert the JSON formatted message to an AnyMessage
                    try:
                        return ChatMessage.model_validate(parsed["content"])
                    except Exception as e:
                        raise Exception(f"Server returned invalid message: {e}")
                case "token":
                    # Yield the str token directly
                    return parsed["content"]
                case "error":
                    error_msg = "Error: " + parsed["content"]
                    return ChatMessage(type="ai", content=error_msg)
        return None

    def stream(
        self,
        message: str,
        model_name: str | None = None,
        model_provider: str | None = None,
        model_config_key: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        recursion_limit: int | None = None,
        stream_tokens: bool = True,
    ) -> Generator[ChatMessage | str, None, None]:
        """Stream the agent's response synchronously.

        Each intermediate message of the agent process is yielded as a ChatMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming models as they are generated.

        Args:
            message (str): The message to send to the agent
            model_name (str, optional): LLM model to use for the agent
            model_provider (str, optional): LLM model provider to use for the agent
            model_config_key (str, optional): Key for predefined model configuration
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for identifying the user
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            recursion_limit (int, optional): Recursion limit for the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            Generator[ChatMessage | str, None, None]: The response from the agent

        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")

        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model_name:
            request.model_name = model_name
        if model_provider:
            request.model_provider = model_provider
        if model_config_key:
            request.model_config_key = model_config_key
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        if recursion_limit is not None:
            request.recursion_limit = recursion_limit

        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/{self.agent}/stream",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        parsed = self._parse_stream_line(line)
                        if parsed is None:
                            break
                        yield parsed
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

    async def astream(
        self,
        message: str,
        model_name: str | None = None,
        model_provider: str | None = None,
        model_config_key: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        recursion_limit: int | None = None,
        stream_tokens: bool = True,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """Stream the agent's response asynchronously.

        Each intermediate message of the agent process is yielded as a ChatMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming models as they are generated.

        Args:
            message (str): The message to send to the agent
            model_name (str, optional): LLM model to use for the agent
            model_provider (str, optional): LLM model provider to use for the agent
            model_config_key (str, optional): Key for predefined model configuration
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for identifying the user
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            recursion_limit (int, optional): Recursion limit for the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            AsyncGenerator[ChatMessage | str, None]: The response from the agent

        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")

        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model_name:
            request.model_name = model_name
        if model_provider:
            request.model_provider = model_provider
        if model_config_key:
            request.model_config_key = model_config_key
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        if recursion_limit is not None:
            request.recursion_limit = recursion_limit

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/{self.agent}/stream",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                parsed = self._parse_stream_line(line)
                                if parsed is None:
                                    break
                                yield parsed
                            except GeneratorExit:
                                # Handle GeneratorExit properly to close the stream gracefully
                                break
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    async def acreate_feedback(
        self,
        run_id: str,
        key: str,
        score: float,
        kwargs: dict[str, Any] = {},
        user_id: str | None = None,
    ) -> None:
        """Create a feedback record for a run.

        Args:
            run_id (str): The ID of the run to provide feedback for
            key (str): The key for the feedback
            score (float): The score for the feedback
            kwargs (dict[str, Any], optional): Additional metadata for the feedback
            user_id (str, optional): User ID for identifying the user

        """
        request = Feedback(run_id=run_id, key=key, score=score, user_id=user_id, kwargs=kwargs)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/feedback",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response.json()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    def get_history(
        self,
        thread_id: str,
        user_id: str | None = None,
    ) -> ChatHistory:
        """Get chat history.

        Args:
            thread_id (str, optional): Thread ID for identifying a conversation
            user_id (str, optional): User ID for identifying the user

        """
        request = ChatHistoryInput(thread_id=thread_id, user_id=user_id)
        try:
            response = httpx.get(
                f"{self.base_url}/{self.agent}/history" if self.agent else f"{self.base_url}/history",
                params=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatHistory.model_validate(response.json())

    async def aget_history(
        self,
        thread_id: str,
        user_id: str | None = None,
    ) -> ChatHistory:
        """Get chat history asynchronously.

        Args:
            thread_id (str, optional): Thread ID for identifying a conversation
            user_id (str, optional): User ID for identifying the user

        """
        request = ChatHistoryInput(thread_id=thread_id, user_id=user_id)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/{self.agent}/history" if self.agent else f"{self.base_url}/history",
                    params=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return ChatHistory.model_validate(response.json())

    def clear_history(
        self,
        thread_id: str | None = None,
        user_id: str | None = None,
    ) -> ClearHistoryResponse:
        """Clear chat history.

        Args:
            thread_id (str, optional): Thread ID for identifying a conversation
            user_id (str, optional): User ID for identifying the user

        """
        if not thread_id and not user_id:
            raise AgentClientError("At least one of thread_id or user_id must be provided")

        request = ClearHistoryInput(thread_id=thread_id, user_id=user_id)
        try:
            response = httpx.delete(
                f"{self.base_url}/{self.agent}/history/clear" if self.agent else f"{self.base_url}/history/clear",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ClearHistoryResponse.model_validate(response.json())

    async def aclear_history(
        self,
        thread_id: str | None = None,
        user_id: str | None = None,
    ) -> ClearHistoryResponse:
        """Clear chat history asynchronously.

        Args:
            thread_id (str, optional): Thread ID for identifying a conversation
            user_id (str, optional): User ID for identifying the user

        """
        if not thread_id and not user_id:
            raise AgentClientError("At least one of thread_id or user_id must be provided")

        request = ClearHistoryInput(thread_id=thread_id, user_id=user_id)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.delete(
                    f"{self.base_url}/{self.agent}/history/clear" if self.agent else f"{self.base_url}/history/clear",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return ClearHistoryResponse.model_validate(response.json())

    def add_messages(
        self,
        messages: list[dict[str, str]] | list[MessageInput],
        thread_id: str | None = None,
        user_id: str | None = None,
    ) -> AddMessagesResponse:
        """Add messages to chat history.

        Args:
            messages (list[dict[str, str]] | list[MessageInput]): Messages to add
            thread_id (str, optional): Thread ID for identifying a conversation
            user_id (str, optional): User ID for identifying the user

        """
        if not thread_id and not user_id:
            raise AgentClientError("At least one of thread_id or user_id must be provided")

        # Convert dict messages to MessageInput if needed
        message_inputs = [
            m if isinstance(m, MessageInput) else MessageInput(type=m["type"], content=m["content"]) for m in messages
        ]

        request = AddMessagesInput(thread_id=thread_id, user_id=user_id, messages=message_inputs)
        try:
            response = httpx.post(
                f"{self.base_url}/{self.agent}/history/add_messages"
                if self.agent
                else f"{self.base_url}/history/add_messages",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return AddMessagesResponse.model_validate(response.json())

    async def aadd_messages(
        self,
        messages: list[dict[str, str]] | list[MessageInput],
        thread_id: str | None = None,
        user_id: str | None = None,
    ) -> AddMessagesResponse:
        """Add messages to chat history asynchronously.

        Args:
            messages (list[dict[str, str]] | list[MessageInput]): Messages to add
            thread_id (str, optional): Thread ID for identifying a conversation
            user_id (str, optional): User ID for identifying the user

        """
        if not thread_id and not user_id:
            raise AgentClientError("At least one of thread_id or user_id must be provided")

        # Convert dict messages to MessageInput if needed
        message_inputs = [
            m if isinstance(m, MessageInput) else MessageInput(type=m["type"], content=m["content"]) for m in messages
        ]

        request = AddMessagesInput(thread_id=thread_id, user_id=user_id, messages=message_inputs)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/{self.agent}/history/add_messages"
                    if self.agent
                    else f"{self.base_url}/history/add_messages",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return AddMessagesResponse.model_validate(response.json())

    def create_feedback(
        self,
        run_id: str,
        key: str,
        score: float,
        kwargs: dict[str, Any] = {},
        user_id: str | None = None,
    ) -> FeedbackResponse:
        """Create a feedback record for a run.

        Args:
            run_id (str): The ID of the run to provide feedback for
            key (str): The key for the feedback
            score (float): The score for the feedback
            kwargs (dict[str, Any], optional): Additional metadata for the feedback
            user_id (str, optional): User ID for identifying the user

        """
        request = Feedback(run_id=run_id, key=key, score=score, user_id=user_id, kwargs=kwargs)
        try:
            response = httpx.post(
                f"{self.base_url}/{self.agent}/feedback" if self.agent else f"{self.base_url}/feedback",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return FeedbackResponse.model_validate(response.json())
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")
