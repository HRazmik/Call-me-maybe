from pydantic import BaseModel, Field, RootModel


class PromptInput(BaseModel):
    """A single entry from function_calling_tests.json."""
    prompt: str = Field(..., min_length=1)


class PromptInputs(RootModel[list[PromptInput]]):
    root: list[PromptInput]


class FunctionCallResult(BaseModel):
    """One entry of function_calling_results.json."""
    prompt: str
    name: str
    parameters: dict[str, object]
