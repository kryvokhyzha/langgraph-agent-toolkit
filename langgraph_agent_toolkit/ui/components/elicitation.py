"""Render MCP questions without service-only dependencies."""

import hashlib
import json
import math
from typing import Any
from urllib.parse import urlsplit

import streamlit as st


def has_mcp_interrupts(pending: list[dict]) -> bool:
    """Check whether the conversation needs structured MCP answers."""
    return any(
        isinstance(item.get("value"), dict) and item["value"].get("type") == "mcp_elicitation" for item in pending
    )


def _simple_schema(schema: dict) -> bool:
    """Use widgets only for schemas that the form can represent."""
    if not isinstance(schema, dict) or schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        return False
    if set(schema) - {"type", "properties", "required", "title", "description", "additionalProperties", "$schema"}:
        return False
    if schema.get("additionalProperties", False) not in (True, False):
        return False
    required = schema.get("required", [])
    if not isinstance(required, list) or any(
        not isinstance(name, str) or name not in schema["properties"] for name in required
    ):
        return False
    for spec in schema["properties"].values():
        if not isinstance(spec, dict) or set(spec) - {"type", "title", "description", "default", "enum", "items"}:
            return False
        kind = spec.get("type")
        if not isinstance(kind, str) or kind not in {"string", "integer", "number", "boolean", "array"}:
            return False
        if "enum" in spec and (not isinstance(spec["enum"], list) or not spec["enum"]):
            return False
        if "enum" in spec:
            try:
                for value in spec["enum"]:
                    _wire_content({"value": value})
            except ValueError:
                return False
        if kind == "array":
            items = spec.get("items", {})
            if not isinstance(items, dict) or items.get("type") != "string" or set(items) - {"type", "enum"}:
                return False
            if "enum" in items and (
                not isinstance(items["enum"], list)
                or not items["enum"]
                or any(not isinstance(item, str) for item in items["enum"])
            ):
                return False
    return True


def _wire_content(content: Any) -> dict:
    """Reject values that the MCP form protocol cannot send."""
    if not isinstance(content, dict) or any(
        not isinstance(key, str)
        or not (
            value is None
            or isinstance(value, (str, bool, int))
            or isinstance(value, float)
            and math.isfinite(value)
            or isinstance(value, list)
            and all(isinstance(item, str) for item in value)
        )
        for key, value in content.items()
    ):
        raise ValueError("Use an object with strings, numbers, booleans, null, or lists of strings.")
    return content


def _form_content(schema: dict, key: str) -> tuple[dict, list[str]]:
    """Render fields and return their values and validation errors."""
    if not _simple_schema(schema):
        st.caption("Enter a JSON object that matches the requested schema.")
        with st.expander("Requested schema"):
            st.json(schema)
        raw = st.text_area("Answers as JSON", value="{}", key=f"{key}:json")
        try:
            return _wire_content(json.loads(raw)), []
        except (ValueError, TypeError):
            return {}, ["Enter a valid JSON object with supported MCP form values."]

    content = {}
    errors = []
    required = schema.get("required", [])
    for index, (name, spec) in enumerate(schema["properties"].items()):
        label = spec.get("title") or name
        field_key = f"{key}:field:{index}"
        include = name in required or st.checkbox(f"Include {label}", key=f"{field_key}:include")
        kwargs = {"key": field_key, "help": spec.get("description")}
        kind = spec["type"]
        default = spec.get("default")
        if "enum" in spec:
            options = spec["enum"]
            value = st.selectbox(label, options, index=options.index(default) if default in options else 0, **kwargs)
        elif kind == "boolean":
            value = st.checkbox(label, value=default if isinstance(default, bool) else False, **kwargs)
        elif kind in {"integer", "number"}:
            # Text preserves large integers and does not select an answer silently.
            raw = st.text_input(label, value=str(default) if default is not None else "", **kwargs)
            try:
                value = int(raw) if kind == "integer" else float(raw)
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError
            except ValueError:
                value = None
                if include:
                    errors.append(f"Enter a valid {kind} for {label}.")
        elif kind == "array":
            options = spec["items"].get("enum")
            if options is not None:
                selected = default if isinstance(default, list) and all(item in options for item in default) else []
                value = st.multiselect(label, options, default=selected, **kwargs)
            else:
                raw = st.text_area(label, value=json.dumps(default if isinstance(default, list) else []), **kwargs)
                try:
                    value = json.loads(raw)
                    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                        raise ValueError
                except ValueError:
                    value = []
                    if include:
                        errors.append(f"Enter a JSON list of strings for {label}.")
        else:
            value = st.text_input(label, value=default if isinstance(default, str) else "", **kwargs)
        if include:
            content[name] = value
    return content, errors


def render_elicitation(pending: list[dict], conversation: tuple, revision: int = 0) -> dict | None:
    """Collect each pending answer and return an explicit resume request."""
    # Later rounds can repeat both the ID and the question. Reset their widgets.
    scope = hashlib.sha256(
        json.dumps([conversation, pending, revision], sort_keys=True, default=str).encode()
    ).hexdigest()[:20]
    resume = {}
    errors = []
    st.info("The agent needs your answers before it can continue.")
    with st.form(f"mcp:{scope}"):
        for index, item in enumerate(pending):
            value = item.get("value")
            key = f"mcp:{scope}:{index}"
            if not isinstance(value, dict) or value.get("type") != "mcp_elicitation":
                st.write(value)
                raw = st.text_area("Other interrupt answer as JSON", value="{}", key=key)
                try:
                    resume[item["id"]] = json.loads(raw)
                except ValueError:
                    errors.append("Enter valid JSON for the other pending interrupt.")
                continue
            responses = {}
            for request_index, request in enumerate(value.get("requests", [])):
                request_key = f"{key}:{request_index}"
                st.write(request.get("message", "The tool needs more information."))
                mode = request.get("mode")
                if mode == "url":
                    url = request.get("url", "")
                    try:
                        parsed = urlsplit(url)
                        web_url = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
                    except ValueError:
                        web_url = False
                    if web_url:
                        st.link_button("Open requested page", url)
                    else:
                        st.code(url)
                    st.caption("Complete the requested page before you select accept.")
                action = st.selectbox("Action", ["accept", "decline", "cancel"], key=f"{request_key}:action")
                answer = {"action": action}
                if mode != "url":
                    content, field_errors = _form_content(request.get("requested_schema", {}), request_key)
                    if action == "accept":
                        answer["content"] = content
                        errors.extend(field_errors)
                responses[request["key"]] = answer
            resume[item["id"]] = {"responses": responses}
        submitted = st.form_submit_button("Send answers")
    if not submitted:
        return None
    if errors:
        for error in errors:
            st.error(error)
        return None
    return next(iter(resume.values())) if len(pending) == 1 else {"resume": resume}
