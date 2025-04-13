from typing import Dict, Optional, Union

import openai
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI


class ChatOpenAIPatched(ChatOpenAI):
    def _create_chat_result(
        self,
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[Dict] = None,
    ) -> ChatResult:
        for idx in range(len(response.choices)):
            resp = response.choices[idx]
            # fix the role of the message
            if resp.message.role.startswith("assistant"):
                resp.message.role = "assistant"
            response.choices[idx] = resp

        return super()._create_chat_result(response, generation_info)
