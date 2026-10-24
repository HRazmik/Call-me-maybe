"""Pydantic models for functions_definition.json."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, RootModel, field_validator


class ParamType(str, Enum):
    """Supported JSON-schema-like types for function parameters and returns.

    Attributes:
        STRING: A text value.
        NUMBER: A floating-point numeric value.
        INTEGER: A whole-number value.
        BOOLEAN: A true or false value.
        ARRAY: An ordered list of values.
        OBJECT: A mapping of string keys to values.
    """

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ParameterSpec(BaseModel):
    """Schema for a single function parameter.

    Attributes:
        type: The JSON-schema-like type the parameter must have.
    """

    type: ParamType


class ReturnSpec(BaseModel):
    """Schema for a function's return value.

    Attributes:
        type: The JSON-schema-like type the function returns.
    """

    type: ParamType


class FunctionDefinition(BaseModel):
    """A single callable function as declared in functions_definition.json.

    Attributes:
        name: The function's identifier. Must be a valid Python identifier.
        description: A human-readable explanation of what the function does.
        parameters: Mapping of parameter names to their specifications.
            Defaults to an empty dict for functions that take no arguments.
        returns: The specification of the function's return value.
    """

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: ReturnSpec

    @field_validator("name")
    @classmethod
    def name_must_be_identifier(cls, v: str) -> str:
        """Validate that the function name is a valid identifier.

        Args:
            v: The function name to validate.

        Returns:
            The unchanged function name if it is valid.

        Raises:
            ValueError: If the name is not a valid Python identifier.
        """
        if not v.isidentifier():
            raise ValueError(f"Function name {v!r} is not a valid identifier")
        return v


class FunctionsDefinition(RootModel[list[FunctionDefinition]]):
    """The full contents of a functions_definition.json file.

    The file is represented as a top-level JSON array of
    FunctionDefinition objects.

    Attributes:
        root: The list of function definitions.
    """

    root: list[FunctionDefinition]

    def get(self, name: str) -> FunctionDefinition | None:
        """Look up a function definition by name.

        Args:
            name: The exact name of the function to find.

        Returns:
            The matching FunctionDefinition, or None if no function
            with that name is defined.
        """
        return next((fn for fn in self.root if fn.name == name), None)

    @property
    def names(self) -> list[str]:
        """All function names, in declaration order.

        Returns:
            A list of every function name, used as the candidate set
            for the name-selection grammar stage.
        """
        return [fn.name for fn in self.root]
