import os

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models.base import init_chat_model
from pydantic import BaseModel, Field


load_dotenv(find_dotenv(".local.env"), override=True)


class ResponseSchema(BaseModel):
    response: str = Field(
        description="The response on user query.",
    )
    alternative_response: str = Field(
        description="The alternative response on user query.",
    )


if __name__ == "__main__":
    DEFAULT_CONFIG_PREFIX = "agent"
    DEFAULT_CONFIGURABLE_FIELDS = ("temperature", "max_tokens", "top_p", "streaming")
    DEFAULT_MODEL_PARAMETER_VALUES = dict(
        temperature=0.0,
        max_tokens=1024,
        top_p=0.7,
        streaming=False,
    )

    configurable_model = init_chat_model(
        model=os.getenv("OPENAI_MODEL_NAME"),
        model_provider="openai",
        config_prefix=DEFAULT_CONFIG_PREFIX,
        configurable_fields=DEFAULT_CONFIGURABLE_FIELDS,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE_URL"),
        **DEFAULT_MODEL_PARAMETER_VALUES,
    )

    print(f"Configurable model type: {type(configurable_model)}")
    print("=" * 25)

    model_so = configurable_model.with_structured_output(ResponseSchema)

    print(f"Model with SO type: {type(configurable_model)}")
    print("=" * 25)

    result = model_so.invoke("Who is the president of Brasil?", config={"temperature": 0.5})

    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    print("=" * 25)
