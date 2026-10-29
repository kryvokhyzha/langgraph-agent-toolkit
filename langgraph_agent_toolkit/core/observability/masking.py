"""Mask typed secrets before an observability SDK serializes data."""

from dataclasses import fields, is_dataclass
from typing import Any

from pydantic import BaseModel, SecretBytes, SecretStr
from pydantic.v1 import BaseModel as LegacyBaseModel
from pydantic.v1 import SecretBytes as LegacySecretBytes
from pydantic.v1 import SecretStr as LegacySecretStr


def mask_secret_values(*, data: Any) -> Any:
    """Copy nested data and replace typed secrets and circular references."""
    active: set[int] = set()

    def mask(value: Any) -> Any:
        if isinstance(value, (SecretStr, SecretBytes, LegacySecretStr, LegacySecretBytes)):
            return "<secret>"
        dataclass_value = is_dataclass(value) and not isinstance(value, type)
        if not isinstance(value, (BaseModel, LegacyBaseModel, dict, list, tuple)) and not dataclass_value:
            return value
        identity = id(value)
        if identity in active:
            return "<circular>"
        active.add(identity)
        try:
            if isinstance(value, BaseModel):
                return mask(value.model_dump())
            if isinstance(value, LegacyBaseModel):
                return mask(value.dict())
            if dataclass_value:
                return {field.name: mask(getattr(value, field.name)) for field in fields(value)}
            if isinstance(value, dict):
                return {mask(key): mask(item) for key, item in value.items()}
            return [mask(item) for item in value]
        finally:
            active.remove(identity)

    return mask(data)
