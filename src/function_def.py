"""Pydantic models for functions_definition.json."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, RootModel, field_validator


class ParamType(str, Enum):
    """Supported JSON-schema-like types for function parameters/returns."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ParameterSpec(BaseModel):
    """Schema for a single function parameter."""

    type: ParamType


class ReturnSpec(BaseModel):
    """Schema for a function's return value."""

    type: ParamType


class FunctionDefinition(BaseModel):
    """A single callable function as declared in functions_definition.json."""

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: ReturnSpec

    @field_validator("name")
    @classmethod
    def name_must_be_identifier(cls, v: str) -> str:
        """Ensure the function name is a valid, callable-looking identifier."""
        if not v.isidentifier():
            raise ValueError(f"Function name {v!r} is not a valid identifier")
        return v


class FunctionsDefinition(RootModel[list[FunctionDefinition]]):
    """The full functions_definition.json
    file: an array of FunctionDefinition."""

    root: list[FunctionDefinition]

    def get(self, name: str) -> FunctionDefinition | None:
        """Look up a function definition by name."""
        return next((fn for fn in self.root if fn.name == name), None)

    @property
    def names(self) -> list[str]:
        """All valid function names, for the name-selection grammar stage."""
        return [fn.name for fn in self.root]
