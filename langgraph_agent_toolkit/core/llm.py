from functools import cache
from typing import TypeAlias, Any, Callable, Dict, Sequence, Union, Optional, Literal

from langchain_community.chat_models import FakeListChatModel
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import BaseMessage

from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.schema.models import (
    AllModelEnum,
    FakeModelName,
    OpenAICompatibleName,
)

_MODEL_TABLE = {
    OpenAICompatibleName.OPENAI_COMPATIBLE: settings.COMPATIBLE_MODEL,
    FakeModelName.FAKE: "fake",
}


class FakeToolModel(FakeListChatModel):
    def __init__(self, responses: list[str]):
        super().__init__(responses=responses)

    def bind_tools(
        self,
        tools: Sequence[
            Union[Dict[str, Any], type, Callable, BaseTool]  # noqa: UP006
        ],
        *,
        tool_choice: Optional[Union[str, Literal["any"]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        return self


ModelT: TypeAlias = ChatOpenAI | FakeToolModel


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in OpenAICompatibleName:
        if not settings.COMPATIBLE_BASE_URL or not settings.COMPATIBLE_MODEL:
            raise ValueError("OpenAICompatible base url and endpoint must be configured")

        return ChatOpenAI(
            model_name=settings.COMPATIBLE_MODEL,
            temperature=0.5,
            streaming=True,
            openai_api_base=settings.COMPATIBLE_BASE_URL,
            openai_api_key=settings.COMPATIBLE_API_KEY,
        )

    elif model_name in FakeModelName:
        return FakeToolModel(responses=["This is a test response from the fake model."])

    else:
        raise ValueError(f"Unsupported model: {model_name}")
