"""Writing function_calling_results.json."""
import json
from pathlib import Path

from io_models import FunctionCallResult


def write_results(results: list[FunctionCallResult], path: str) -> None:
    """Write results to the output path, creating parent dirs if needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.model_dump() for r in results]
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
