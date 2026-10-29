"""Request two form answers before each external write."""

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastmcp import Context, FastMCP
from mcp.types import ElicitRequest, ElicitRequestFormParams, InputRequiredResult
from pydantic import Field


events = Path(sys.argv[2])
server = FastMCP("MCP recovery journey")


@server.tool
async def save_answer(
    label: str,
    tenant: Annotated[str, Field(json_schema_extra={"x-mcp-header": "tenant"})],
    ctx: Context,
) -> Any:
    """Save the answer after two user input requests."""
    assert tenant == "test-client"
    with events.open("a") as file:
        file.write(json.dumps({"event": "call", "label": label}) + "\n")
    answers = ctx.input_responses
    prior = json.loads(ctx.request_state) if ctx.request_state else {}
    if answers is None:
        step = "answer"
        state = {"label": label}
    elif prior.get("answer") is None:
        step = "confirmation"
        state = {"label": label, "answer": answers["details"].content["answer"]}
    else:
        assert answers["details"].content["confirmation"] == "yes"
        result = {"label": label, "answer": prior["answer"]}
        with events.open("a") as file:
            file.write(json.dumps({"event": "write", **result}) + "\n")
        return result
    return InputRequiredResult(
        input_requests={
            "details": ElicitRequest(
                params=ElicitRequestFormParams(
                    message=f"Supply {step} for {label}.",
                    requested_schema={
                        "type": "object",
                        "properties": {step: {"type": "string"}},
                        "required": [step],
                    },
                )
            )
        },
        request_state=json.dumps(state),
    )


if __name__ == "__main__":
    uvicorn.run(
        server.http_app(stateless_http=True, json_response=True),
        host="127.0.0.1",
        port=int(sys.argv[1]),
        log_level="warning",
    )
