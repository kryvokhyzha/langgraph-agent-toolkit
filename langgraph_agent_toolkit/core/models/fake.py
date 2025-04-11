from typing import Any, Callable, Dict, Sequence, Union, Optional, Literal

from langchain_community.chat_models import FakeListChatModel
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import BaseMessage


class FakeToolModel(FakeListChatModel):
    """A fake model that returns a fixed response for testing purposes."""

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
