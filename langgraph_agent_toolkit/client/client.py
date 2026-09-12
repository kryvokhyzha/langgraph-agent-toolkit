import asyncio
import json
import os
import re
from collections.abc import AsyncGenerator, Generator
from time import monotonic
from typing import Any, Dict

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
    UserComplexInput,
    UserInput,
)
from langgraph_agent_toolkit.schema.models import ModelProvider


class AgentClientError(Exception):
    """Report a client failure with optional HTTP retry information."""

    def __init__(
        self,
        *args: Any,
        status_code: int | None = None,
        error_code: str | None = None,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(*args)
        self.status_code = status_code
        self.error_code = error_code
        # HTTP permits a delay in seconds or an HTTP date.
        self.retry_after = retry_after


_MAX_ERROR_BODY_BYTES = 16 * 1024
_ERROR_BODY_TIMEOUT = 1.0


def _http_error(
    error: httpx.HTTPError,
    *,
    prefix: str = "Error",
    body: bytes | None = None,
) -> AgentClientError:
    """Keep status and safe error fields without adding body text to the message."""
    status_code = None
    error_code = None
    retry_after = None
    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        status_code = response.status_code
        retry_after = response.headers.get("Retry-After")
        if body is None:
            try:
                body = response.content
            except httpx.ResponseNotRead:
                pass
        if body is not None and len(body) <= _MAX_ERROR_BODY_BYTES:
            try:
                value = json.loads(body)
            except (ValueError, UnicodeError, RecursionError):
                value = None
            code = value.get("error_code") if isinstance(value, dict) else None
            if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code):
                error_code = code
    return AgentClientError(
        f"{prefix}: {error}", status_code=status_code, error_code=error_code, retry_after=retry_after
    )


def _raise_stream_status(response: httpx.Response) -> None:
    """Read a bounded error body before closing a rejected stream response."""
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        body = bytearray()
        extensions = response.request.extensions
        previous_timeouts = extensions.get("timeout")
        timeouts = dict(previous_timeouts or {})
        read_timeout = timeouts.get("read")
        timeouts["read"] = min(read_timeout, _ERROR_BODY_TIMEOUT) if read_timeout is not None else _ERROR_BODY_TIMEOUT
        # HTTPX passes this extension to HTTPcore before response-body reads start.
        extensions["timeout"] = timeouts
        deadline = monotonic() + _ERROR_BODY_TIMEOUT
        try:
            for chunk in response.iter_bytes():
                if monotonic() >= deadline or len(body) + len(chunk) > _MAX_ERROR_BODY_BYTES:
                    body.clear()
                    break
                body.extend(chunk)
        except httpx.HTTPError:
            body.clear()
        finally:
            if previous_timeouts is None:
                extensions.pop("timeout", None)
            else:
                extensions["timeout"] = previous_timeouts
        raise _http_error(error, body=bytes(body)) from error


async def _araise_stream_status(response: httpx.Response) -> None:
    """Read a bounded async error body before closing the response."""
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        body = bytearray()
        try:
            async with asyncio.timeout(_ERROR_BODY_TIMEOUT):
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > _MAX_ERROR_BODY_BYTES:
                        body.clear()
                        break
                    body.extend(chunk)
        except (httpx.HTTPError, TimeoutError):
            body.clear()
        raise _http_error(error, body=bytes(body)) from error


class AgentClient:
    """Client for the agent service.

    Use a context manager to close owned HTTP connections. An async client must
    stay in one event loop until aclose() completes. The caller owns injected clients.
    """

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | httpx.Timeout | None = httpx.Timeout(60.0, connect=10.0, write=30.0, pool=10.0),
        get_info: bool = True,
        verify: bool = False,
        *,
        stream_timeout: float | httpx.Timeout | None = httpx.Timeout(120.0, connect=10.0, write=30.0, pool=10.0),
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
        auth_secret: str | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            base_url (str): Base URL of the agent service.
            agent (str): Default agent name.
            timeout (float, optional): Request timeout.
            get_info (bool, optional): Fetch agent information during initialization.
            verify (bool, optional): Verify agent information.
            stream_timeout: Stream timeout. The read timeout limits idle time between chunks.
            http_client: Optional shared sync client. The caller must close this client.
            async_http_client: Optional shared async client. The caller must close this client.
            auth_secret: Optional bearer token. Defaults to AUTH_SECRET from the environment.

        """
        self.base_url = base_url
        self.auth_secret = auth_secret if auth_secret is not None else os.getenv("AUTH_SECRET")
        self.timeout = timeout
        self.stream_timeout = stream_timeout
        self._http_client = http_client
        self._async_http_client = async_http_client
        self._owns_http_client = http_client is None
        self._owns_async_http_client = async_http_client is None
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self.info: ServiceMetadata | None = None
        self.agent: str | None = None
        try:
            if get_info:
                self.retrieve_info()
            if agent:
                self.update_agent(agent, verify=verify)
        except Exception:
            self.close()
            raise

    def _get_http_client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client()
        return self._http_client

    def _get_async_http_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._async_loop is not None and self._async_loop is not loop:
            raise AgentClientError(
                "The async client belongs to another event loop. "
                "Use a separate AgentClient for each loop and close it with 'async with' or aclose()."
            )
        if self._async_http_client is None:
            self._async_http_client = httpx.AsyncClient()
        self._async_loop = loop
        return self._async_http_client

    def close(self) -> None:
        """Close the owned sync client. Use aclose() to close async resources."""
        if self._owns_http_client and self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    async def aclose(self) -> None:
        """Close owned clients. Call this method in the loop that made the requests."""
        if self._owns_async_http_client and self._async_http_client is not None:
            if self._async_loop is not asyncio.get_running_loop():
                raise AgentClientError("Close the async client in the event loop that created it.")
            await self._async_http_client.aclose()
            self._async_http_client = None
            self._async_loop = None
        self.close()

    def __enter__(self) -> "AgentClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    async def __aenter__(self) -> "AgentClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    @property
    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    def retrieve_info(self) -> None:
        try:
            response = self._get_http_client().get(
                f"{self.base_url}/info",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise _http_error(e, prefix="Error getting service info") from e

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
        input: Dict[str, Any],
        model_name: str | None = None,
        model_provider: str | ModelProvider | None = None,
        model_config_key: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        recursion_limit: int | None = None,
    ) -> ChatMessage:
        """Invoke the agent asynchronously and return its final message.

        Args:
            input (Dict[str, Any]): The input to send to the agent
            model_name (str, optional): LLM model to use for the agent
            model_provider (str | ModelProvider, optional): LLM model provider to use for the agent
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

        request = self._build_request(
            input, model_name, model_provider, model_config_key, thread_id, user_id, agent_config, recursion_limit
        )

        client = self._get_async_http_client()
        try:
            response = await client.post(
                f"{self.base_url}/{self.agent}/invoke",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise _http_error(e) from e

        return ChatMessage.model_validate(response.json())

    def invoke(
        self,
        input: Dict[str, Any],
        model_name: str | None = None,
        model_provider: str | ModelProvider | None = None,
        model_config_key: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        recursion_limit: int | None = None,
    ) -> ChatMessage:
        """Invoke the agent synchronously and return its final message.

        Args:
            input (Dict[str, Any]): The input to send to the agent
            model_name (str, optional): LLM model to use for the agent
            model_provider (str | ModelProvider, optional): LLM model provider to use for the agent
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

        request = self._build_request(
            input, model_name, model_provider, model_config_key, thread_id, user_id, agent_config, recursion_limit
        )

        try:
            response = self._get_http_client().post(
                f"{self.base_url}/{self.agent}/invoke",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise _http_error(e) from e

        return ChatMessage.model_validate(response.json())

    def _parse_stream_payload(self, data: str) -> ChatMessage | str:
        """Decode the shared SSE and JSON Lines event format."""
        try:
            parsed = json.loads(data)
        except ValueError as exc:
            raise AgentClientError(f"Invalid JSON in the server stream: {exc}") from exc
        if not isinstance(parsed, dict):
            raise AgentClientError("Server returned an invalid stream event.")
        content = parsed.get("content")
        match parsed.get("type"):
            case "message":
                try:
                    return ChatMessage.model_validate(content)
                except ValueError as exc:
                    raise AgentClientError(f"Server returned an invalid message: {exc}") from exc
            case "token" | "error":
                if not isinstance(content, str):
                    raise AgentClientError("A stream token or error must contain text.")
                if parsed["type"] == "error":
                    raise AgentClientError(content)
                return content
            case _:
                raise AgentClientError("Server returned an unsupported stream event type.")

    def _parse_stream_line(self, line: str) -> ChatMessage | str | None:
        """Read one SSE data line. Ignore comments and event metadata."""
        line = line.strip()
        if not line.startswith("data:"):
            return None
        data = line[5:].lstrip(" ")
        if data == "[DONE]":
            return None
        return self._parse_stream_payload(data)

    def _parse_jsonl_line(self, line: str) -> ChatMessage | str | None:
        """Read one JSON Lines event. Blank lines carry no data."""
        return self._parse_stream_payload(line) if line.strip() else None

    def _build_request(
        self,
        input: Dict[str, Any],
        model_name: str | None,
        model_provider: str | ModelProvider | None,
        model_config_key: str | None,
        thread_id: str | None,
        user_id: str | None,
        agent_config: dict[str, Any] | None,
        recursion_limit: int | None,
        stream_tokens: bool | None = None,
    ) -> UserInput:
        """Build one request body for invocation or either streaming protocol."""
        values = UserComplexInput(**input)
        request = (
            UserInput(input=values) if stream_tokens is None else StreamInput(input=values, stream_tokens=stream_tokens)
        )
        if thread_id:
            request.thread_id = thread_id
        if model_name:
            request.model_name = model_name
        if model_provider:
            request.model_provider = (
                model_provider.value if isinstance(model_provider, ModelProvider) else model_provider
            )
        if model_config_key:
            request.model_config_key = model_config_key
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        if recursion_limit is not None:
            request.recursion_limit = recursion_limit
        return request

    def stream(
        self,
        input: Dict[str, Any],
        model_name: str | None = None,
        model_provider: str | ModelProvider | None = None,
        model_config_key: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        recursion_limit: int | None = None,
        stream_tokens: bool = True,
    ) -> Generator[ChatMessage | str, None, None]:
        """Stream agent responses synchronously.

        Yield each intermediate `ChatMessage`.
        Yield content tokens when `stream_tokens` is `True`.

        Args:
            input (Dict[str, Any]): The input to send to the agent
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

        request = self._build_request(
            input,
            model_name,
            model_provider,
            model_config_key,
            thread_id,
            user_id,
            agent_config,
            recursion_limit,
            stream_tokens,
        )

        try:
            with self._get_http_client().stream(
                "POST",
                f"{self.base_url}/{self.agent}/stream",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.stream_timeout,
            ) as response:
                _raise_stream_status(response)
                for line in response.iter_lines():
                    if line.startswith("data:") and line[5:].strip() == "[DONE]":
                        break
                    parsed = self._parse_stream_line(line)
                    if parsed is not None and parsed != "":
                        yield parsed
                else:
                    raise AgentClientError("The server stream ended before its completion marker.")
        except httpx.HTTPError as e:
            raise _http_error(e) from e

    def stream_jsonl(
        self,
        input: Dict[str, Any],
        model_name: str | None = None,
        model_provider: str | ModelProvider | None = None,
        model_config_key: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        recursion_limit: int | None = None,
        stream_tokens: bool = True,
    ) -> Generator[ChatMessage | str, None, None]:
        """Stream agent responses synchronously through the JSON Lines endpoint.

        Yield the same `ChatMessage | str` values as `stream`.
        Use `/stream/jsonl` with media type `application/jsonl` instead of SSE.
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")

        request = self._build_request(
            input,
            model_name,
            model_provider,
            model_config_key,
            thread_id,
            user_id,
            agent_config,
            recursion_limit,
            stream_tokens,
        )

        try:
            with self._get_http_client().stream(
                "POST",
                f"{self.base_url}/{self.agent}/stream/jsonl",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.stream_timeout,
            ) as response:
                _raise_stream_status(response)
                for line in response.iter_lines():
                    parsed = self._parse_jsonl_line(line)
                    if parsed is not None and parsed != "":
                        yield parsed
        except httpx.HTTPError as e:
            raise _http_error(e) from e

    async def astream(
        self,
        input: Dict[str, Any],
        model_name: str | None = None,
        model_provider: str | ModelProvider | None = None,
        model_config_key: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        recursion_limit: int | None = None,
        stream_tokens: bool = True,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """Stream agent responses asynchronously.

        Yield each intermediate `ChatMessage`.
        Yield content tokens when `stream_tokens` is `True`.

        Args:
            input (Dict[str, Any]): The input to send to the agent
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

        request = self._build_request(
            input,
            model_name,
            model_provider,
            model_config_key,
            thread_id,
            user_id,
            agent_config,
            recursion_limit,
            stream_tokens,
        )

        client = self._get_async_http_client()
        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/{self.agent}/stream",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.stream_timeout,
            ) as response:
                await _araise_stream_status(response)
                async for line in response.aiter_lines():
                    if line.startswith("data:") and line[5:].strip() == "[DONE]":
                        break
                    parsed = self._parse_stream_line(line)
                    if parsed is not None and parsed != "":
                        yield parsed
                else:
                    raise AgentClientError("The server stream ended before its completion marker.")
        except httpx.HTTPError as e:
            raise _http_error(e) from e

    async def astream_jsonl(
        self,
        input: Dict[str, Any],
        model_name: str | None = None,
        model_provider: str | ModelProvider | None = None,
        model_config_key: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        recursion_limit: int | None = None,
        stream_tokens: bool = True,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """Stream JSON Lines (NDJSON) responses asynchronously like `stream_jsonl`."""
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")

        request = self._build_request(
            input,
            model_name,
            model_provider,
            model_config_key,
            thread_id,
            user_id,
            agent_config,
            recursion_limit,
            stream_tokens,
        )

        client = self._get_async_http_client()
        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/{self.agent}/stream/jsonl",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.stream_timeout,
            ) as response:
                await _araise_stream_status(response)
                async for line in response.aiter_lines():
                    try:
                        parsed = self._parse_jsonl_line(line)
                        if parsed is not None and parsed != "":
                            yield parsed
                    except GeneratorExit:
                        break
        except httpx.HTTPError as e:
            raise _http_error(e) from e

    async def acreate_feedback(
        self,
        run_id: str,
        key: str,
        score: float,
        kwargs: dict[str, Any] = {},
        user_id: str | None = None,
        *,
        feedback_token: str | None = None,
    ) -> FeedbackResponse:
        """Create feedback for a run.

        Args:
            run_id (str): Run ID for feedback.
            key (str): Feedback key.
            score (float): Feedback score.
            kwargs (dict[str, Any], optional): Additional feedback metadata.
            user_id (str, optional): User ID.
            feedback_token: Token from the run's returned `ChatMessage`.

        """
        request = Feedback(
            run_id=run_id, key=key, score=score, user_id=user_id, kwargs=kwargs, feedback_token=feedback_token
        )
        client = self._get_async_http_client()
        try:
            response = await client.post(
                f"{self.base_url}/{self.agent}/feedback" if self.agent else f"{self.base_url}/feedback",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return FeedbackResponse.model_validate(response.json())
        except httpx.HTTPError as e:
            raise _http_error(e) from e

    def get_history(
        self,
        thread_id: str,
        user_id: str | None = None,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> ChatHistory:
        """Get short-term chat history.

        Args:
            thread_id: Required ID for one short-term conversation.
            user_id: User who owns the conversation. This does not select long-term memory.
            offset: Number of messages to skip.
            limit: Maximum number of messages to return.

        """
        if not thread_id:
            raise AgentClientError("thread_id is required")

        request = ChatHistoryInput(thread_id=thread_id, user_id=user_id, offset=offset, limit=limit)
        try:
            response = self._get_http_client().get(
                f"{self.base_url}/{self.agent}/history" if self.agent else f"{self.base_url}/history",
                params=request.model_dump(exclude_none=True),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise _http_error(e) from e

        return ChatHistory.model_validate(response.json())

    async def aget_history(
        self,
        thread_id: str,
        user_id: str | None = None,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> ChatHistory:
        """Get short-term chat history asynchronously.

        Args:
            thread_id: Required ID for one short-term conversation.
            user_id: User who owns the conversation. This does not select long-term memory.
            offset: Number of messages to skip.
            limit: Maximum number of messages to return.

        """
        if not thread_id:
            raise AgentClientError("thread_id is required")

        request = ChatHistoryInput(thread_id=thread_id, user_id=user_id, offset=offset, limit=limit)
        client = self._get_async_http_client()
        try:
            response = await client.get(
                f"{self.base_url}/{self.agent}/history" if self.agent else f"{self.base_url}/history",
                params=request.model_dump(exclude_none=True),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise _http_error(e) from e

        return ChatHistory.model_validate(response.json())

    def clear_history(
        self,
        thread_id: str | None = None,
        user_id: str | None = None,
    ) -> ClearHistoryResponse:
        """Clear one conversation. Keep long-term memory.

        Args:
            thread_id: Required ID for one short-term conversation.
            user_id: User who owns the conversation. This does not select long-term memory.

        """
        if not thread_id:
            raise AgentClientError("thread_id is required")

        request = ClearHistoryInput(thread_id=thread_id, user_id=user_id)
        try:
            response = self._get_http_client().request(
                "DELETE",
                f"{self.base_url}/{self.agent}/history/clear" if self.agent else f"{self.base_url}/history/clear",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise _http_error(e) from e

        return ClearHistoryResponse.model_validate(response.json())

    async def aclear_history(
        self,
        thread_id: str | None = None,
        user_id: str | None = None,
    ) -> ClearHistoryResponse:
        """Clear one conversation asynchronously. Keep long-term memory.

        Args:
            thread_id: Required ID for one short-term conversation.
            user_id: User who owns the conversation. This does not select long-term memory.

        """
        if not thread_id:
            raise AgentClientError("thread_id is required")

        request = ClearHistoryInput(thread_id=thread_id, user_id=user_id)
        client = self._get_async_http_client()
        try:
            response = await client.request(
                "DELETE",
                f"{self.base_url}/{self.agent}/history/clear" if self.agent else f"{self.base_url}/history/clear",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise _http_error(e) from e

        return ClearHistoryResponse.model_validate(response.json())

    def add_messages(
        self,
        messages: list[dict[str, str]] | list[MessageInput],
        thread_id: str | None = None,
        user_id: str | None = None,
    ) -> AddMessagesResponse:
        """Add messages to one short-term conversation.

        Args:
            messages (list[dict[str, str]] | list[MessageInput]): Messages to add
            thread_id: Required ID for one short-term conversation.
            user_id: User who owns the conversation. This does not select long-term memory.

        """
        if not thread_id:
            raise AgentClientError("thread_id is required")

        # Convert dictionaries to `MessageInput` objects.
        message_inputs = [m if isinstance(m, MessageInput) else MessageInput.model_validate(m) for m in messages]

        request = AddMessagesInput(thread_id=thread_id, user_id=user_id, messages=message_inputs)
        try:
            response = self._get_http_client().post(
                f"{self.base_url}/{self.agent}/history/add_messages"
                if self.agent
                else f"{self.base_url}/history/add_messages",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise _http_error(e) from e

        return AddMessagesResponse.model_validate(response.json())

    async def aadd_messages(
        self,
        messages: list[dict[str, str]] | list[MessageInput],
        thread_id: str | None = None,
        user_id: str | None = None,
    ) -> AddMessagesResponse:
        """Add messages to one short-term conversation asynchronously.

        Args:
            messages (list[dict[str, str]] | list[MessageInput]): Messages to add
            thread_id: Required ID for one short-term conversation.
            user_id: User who owns the conversation. This does not select long-term memory.

        """
        if not thread_id:
            raise AgentClientError("thread_id is required")

        # Convert dictionaries to `MessageInput` objects.
        message_inputs = [m if isinstance(m, MessageInput) else MessageInput.model_validate(m) for m in messages]

        request = AddMessagesInput(thread_id=thread_id, user_id=user_id, messages=message_inputs)
        client = self._get_async_http_client()
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
            raise _http_error(e) from e

        return AddMessagesResponse.model_validate(response.json())

    def create_feedback(
        self,
        run_id: str,
        key: str,
        score: float,
        kwargs: dict[str, Any] = {},
        user_id: str | None = None,
        *,
        feedback_token: str | None = None,
    ) -> FeedbackResponse:
        """Create a feedback record for a run.

        Args:
            run_id (str): The ID of the run to provide feedback for
            key (str): The key for the feedback
            score (float): The score for the feedback
            kwargs (dict[str, Any], optional): Additional metadata for the feedback
            user_id (str, optional): User ID for identifying the user
            feedback_token: Token from the run's returned `ChatMessage`.

        """
        request = Feedback(
            run_id=run_id, key=key, score=score, user_id=user_id, kwargs=kwargs, feedback_token=feedback_token
        )
        try:
            response = self._get_http_client().post(
                f"{self.base_url}/{self.agent}/feedback" if self.agent else f"{self.base_url}/feedback",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return FeedbackResponse.model_validate(response.json())
        except httpx.HTTPError as e:
            raise _http_error(e) from e
