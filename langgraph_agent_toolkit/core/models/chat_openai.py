import importlib
from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing
from typing import Any, Dict, Optional, Union

import openai
from langchain_core.exceptions import ModelConnectionError, ModelError, ModelTimeoutError
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import AzureChatOpenAI, ChatOpenAI


def _stream_transport_error(error: Exception) -> Exception | None:
    """Classify raw transport errors only at the model streaming boundary."""
    if isinstance(error, ModelError):
        return None
    for name in ("httpx", "httpx2", "aiohttp"):
        try:
            library = importlib.import_module(name)
        except ImportError:
            continue
        if name == "aiohttp":
            timeouts = (library.ServerTimeoutError,)
            connections = (library.ClientConnectionError, library.ClientPayloadError)
        else:
            timeouts = (library.TimeoutException,)
            connections = (library.NetworkError, library.RemoteProtocolError, library.ProxyError)
        if isinstance(error, timeouts):
            return ModelTimeoutError("The model provider stream timed out")
        if isinstance(error, connections):
            return ModelConnectionError("The model provider stream connection failed")
    return None


class _StreamCompatibility:
    """Preserve refusal text and normalize transport failures in model streams."""

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        if generation is not None and choices:
            refusal = (choices[0].get("delta") or {}).get("refusal")
            if isinstance(refusal, str):
                # LangChain combines string fragments when it combines message chunks.
                generation.message.additional_kwargs.setdefault("refusal", refusal)
        return generation

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        try:
            yield from super()._stream(*args, **kwargs)
        except Exception as exc:
            if normalized := _stream_transport_error(exc):
                raise normalized from exc
            raise

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        try:
            async with aclosing(super()._astream(*args, **kwargs)) as stream:
                async for chunk in stream:
                    yield chunk
        except Exception as exc:
            if normalized := _stream_transport_error(exc):
                raise normalized from exc
            raise


class AzureChatOpenAIPatched(_StreamCompatibility, AzureChatOpenAI):
    """Preserve refusal text and normalize transport failures in Azure streams."""


class ChatOpenAIPatched(_StreamCompatibility, ChatOpenAI):
    def _create_chat_result(
        self,
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[Dict] = None,
    ) -> ChatResult:
        if isinstance(response, dict):
            for choice in response.get("choices") or []:
                message = choice.get("message", {})
                role = message.get("role")
                if isinstance(role, str) and role.startswith("assistant"):
                    message["role"] = "assistant"
        else:
            for choice in getattr(response, "choices", None) or []:
                role = getattr(choice.message, "role", None)
                if isinstance(role, str) and role.startswith("assistant"):
                    choice.message.role = "assistant"

        return super()._create_chat_result(response, generation_info)
