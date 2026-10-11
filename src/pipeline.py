"""Orchestrates: prompts + function defs -> function call results,
using constrained generation for only the genuinely uncertain parts."""
import json

from llm_sdk import Small_LLM_Model

from .generator import (
    choose_function,
    build_name_pool,
    build_string_pool,
    encode_prompt,
    generate_value_for_type,
    inject_literal,
)
from .grammar import GrammarState
from .vocab import Vocab
from .function_def import FunctionsDefinition, ParamType
from .io_models import FunctionCallResult, PromptInputs


_TYPE_TO_GENERATOR_KIND = {
    ParamType.NUMBER: "number",
    ParamType.INTEGER: "number",
    ParamType.BOOLEAN: "boolean",
    ParamType.STRING: "string",
}


def _escape_for_json_string(text: str) -> str:
    """
    Escape a raw string so it's safe to splice into a JSON string literal.
    """
    return json.dumps(text)[1:-1]  # json.dumps gives '"..."'; strip the quotes


def _format_functions_for_context(functions: FunctionsDefinition) -> str:
    """Render the available functions as
    text the model can read before choosing."""
    lines = ["Available functions:"]
    for fn in functions.root:
        param_desc = ", ".join(
            f"{name}: {spe.type.value}" for name, spe in fn.parameters.items()
        )
        lines.append(f"- {fn.name}({param_desc}): {fn.description}")
    return "\n".join(lines)


def process_prompt(
    model: Small_LLM_Model,
    vocab: Vocab,
    functions: FunctionsDefinition,
    prompt_text: str,
) -> FunctionCallResult:
    """Run the full constrained pipeline for a single prompt."""
    # 1. Build the context the model sees: the natural-language prompt,
    #    followed by the start of the JSON object we're constructing.
    functions_context = _format_functions_for_context(functions)
    context_text = (
        "<|im_start|>system\n"
        "You convert a request into a single function call. "
        "Pick the best function "
        "and extract its arguments exactly as they appear in the request.\n"
        f"{functions_context}<|im_end|>\n"
        f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        "Rules: copy text values exactly as written in the request. "
        "Keep every value as "
        "short as possible. For pattern arguments, "
        "write one minimal pattern in standard "
        "syntax (for example \\s+ for whitespace or \\w+ for words). "
        "Always answer in English.\n"
    )
    input_ids = encode_prompt(model, context_text)
    grammar = GrammarState()

    # 2. Inject the fixed skeleton up through the prompt field - no model
    #    call needed, we already know this text.
    escaped_prompt = _escape_for_json_string(prompt_text)
    inject_literal(model, vocab, input_ids, grammar, '{"prompt":"')
    inject_literal(model, vocab, input_ids, grammar, escaped_prompt)
    inject_literal(model, vocab, input_ids, grammar, '","name":"')

    # 3. Let the model choose the function (scored, not heuristic).
    name_pool = build_name_pool(vocab, [fn.name for fn in functions.root])
    chosen_fn, name_text = choose_function(
        model, vocab, functions, input_ids, name_pool)
    grammar.commit_token(name_text)
    # 4. Inject the chosen name + move into parameters.
    # inject_literal(model, vocab, input_ids, grammar, chosen_fn.name)
    inject_literal(model, vocab, input_ids, grammar, '","parameters":{')

    # 5. For each declared parameter, inject its key, then constrained-
    #    generate its value based on declared type.
    parameters: dict[str, object] = {}
    param_items = list(chosen_fn.parameters.items())
    for i, (param_name, param_spec) in enumerate(param_items):
        escaped_key = _escape_for_json_string(param_name)
        inject_literal(model, vocab, input_ids, grammar, f'"{escaped_key}":')

        kind = _TYPE_TO_GENERATOR_KIND.get(param_spec.type)
        if kind is None:
            # array/object parameter values aren't supported by the current
            # type-restricted generator - flagged as a known gap rather than
            # silently producing wrong output.
            raise NotImplementedError(
                f"Parameter type {param_spec.type!r} not yet supported "
                f"for constrained value generation (param: {param_name!r})"
            )
        string_pool = build_string_pool(vocab, prompt_text)

        raw_value_text = generate_value_for_type(
            model, vocab, input_ids, kind, string_pool)
        print(raw_value_text)
        for ch in raw_value_text:
            grammar.advance(ch)
        parameters[param_name] = json.loads(raw_value_text)

        if i < len(param_items) - 1:
            inject_literal(model, vocab, input_ids, grammar, ",")

    # 6. Close parameters object and outer object.
    inject_literal(model, vocab, input_ids, grammar, "}}")

    return FunctionCallResult(
        prompt=prompt_text,
        name=chosen_fn.name,
        parameters=parameters,
    )


def run_pipeline(
    model: Small_LLM_Model,
    vocab: Vocab,
    functions: FunctionsDefinition,
    prompts: PromptInputs,
) -> list[FunctionCallResult]:
    """Process every prompt into a FunctionCallResult, skipping/reporting
    any that fail rather than crashing the whole run."""
    if not functions.root:
        raise ValueError("No functions available to call")

    results: list[FunctionCallResult] = []
    for prompt_input in prompts.root:
        try:
            result = process_prompt(
                model, vocab, functions, prompt_input.prompt)
            results.append(result)
        except json.JSONDecodeError as exc:
            print("Warning: failed to process prompt ",
                  f"{prompt_input.prompt!r}: {exc}")
        except Exception as exc:
            # noqa: BLE001 - replace with narrower types once known
            # graceful per-prompt failure, per the "never crash" requirement -
            # log and continue rather than aborting the whole batch
            print("Unexpected error: failed to process prompt",
                  f"{prompt_input.prompt!r}: {exc}")
    return results
