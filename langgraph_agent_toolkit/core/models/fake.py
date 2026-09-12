from typing import Any, Callable, Dict, Literal, Optional, Sequence, Union

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool


class FakeToolModel(FakeListChatModel):
    """Fake model that returns a fixed test response."""

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
