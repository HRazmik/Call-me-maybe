"""Loading and validating input JSON files."""
import json
from pathlib import Path

from pydantic import ValidationError

from function_def import FunctionsDefinition
from io_models import PromptInputs


class InputFileError(Exception):
    """Raised when an input file is missing, malformed, or fails validation."""


def _load_json(path: str) -> object:
    file_path = Path(path)
    if not file_path.exists():
        raise InputFileError(f"File not found: {path}")
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputFileError(f"Invalid JSON in {path}: {exc}") from exc


def load_functions_definition(path: str) -> FunctionsDefinition:
    """Load and validate functions_definition.json."""
    data = _load_json(path)
    try:
        return FunctionsDefinition.model_validate(data)
    except ValidationError as exc:
        raise InputFileError(
            f"Invalid function definitions in {path}: {exc}") from exc


def load_prompts(path: str) -> PromptInputs:
    """Load and validate function_calling_tests.json."""
    data = _load_json(path)
    try:
        return PromptInputs.model_validate(data)
    except ValidationError as exc:
        raise InputFileError(f"Invalid prompts in {path}: {exc}") from exc
