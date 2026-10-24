*This project has been created as part of the 42 curriculum by rovnania.*

# call me maybe

Introduction to function calling in LLMs, using constrained decoding on a small language model.

## Description

Large language models don't reliably produce structured output. Small ones are worse: asked for JSON, a 0.6B-parameter model may succeed only some of the time.

This project turns natural-language prompts into structured function calls. Given:

```
"What is the sum of 40 and 2?"
```

the program does **not** answer `42`. It outputs the function to call and its typed arguments:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

The model used is `Qwen/Qwen3-0.6B`, accessed through the provided `llm_sdk` wrapper. The program never relies on the model to produce valid JSON by itself. It intervenes at the logits level, token by token, so that invalid output cannot be generated.

## Instructions

### Requirements

- Python 3.10 or later
- [`uv`](https://docs.astral.sh/uv/)
- The provided `llm_sdk` package, placed next to `src/` and declared as a local dependency in `pyproject.toml`

### Installation

```bash
make install        # runs: uv sync
```

### Running

```bash
make run
```

or directly:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

All three arguments are optional. The defaults are the paths shown above.

### Makefile targets

| Target | Description |
|---|---|
| `install` | Install dependencies with `uv sync` |
| `run` | Run the program |
| `debug` | Run the program under `pdb` |
| `clean` | Remove `__pycache__`, `.mypy_cache`, `.pytest_cache` |
| `lint` | `flake8` and `mypy` with the required flags |
| `lint-strict` | `flake8` and `mypy --strict` |

### Input and output

**`functions_definition.json`**: a list of callable functions, each with `name`, `description`, `parameters` (name to type) and `returns`.

**`function_calling_tests.json`**: a list of `{"prompt": "..."}` objects.

**`function_calling_results.json`**: a list of `{"prompt", "name", "parameters"}` objects, one per successfully processed prompt, with no extra keys.

Missing files, invalid JSON and schema violations in the input files produce a clear error message and a non-zero exit code. They do not produce a traceback.

## Algorithm

### Constrained decoding in one paragraph

An LLM produces one logit per vocabulary token at every step. Normally the highest-scoring token is chosen. Here, before choosing, the program works out which tokens would keep the output valid, and chooses only among those. Since the model can only ever pick a legal token, the output is valid by construction.

### The generation pipeline

For each prompt, the output object is built in stages. Only the genuinely uncertain parts are decided by the model:

1. **Fixed skeleton.** The text `{"prompt":"...","name":"` is known in advance, so it is tokenized and appended directly. No model decision is needed.
2. **Function name.** Decoded under a constraint: at each step only tokens that keep the text a prefix of a real function name are allowed. The closing quote is allowed only once the text equals a complete name. The model's logits decide which function wins.
3. **Parameters.** For each parameter of the chosen function, the key is injected, then the value is decoded under a type-specific constraint (see below).
4. **Closing.** The closing braces are appended directly.

The model sees the list of available functions, with their descriptions and parameter types, in its context before it decides anything.

### The JSON grammar (`grammar.py`)

A character-level state machine with a stack tracks whether the text so far is a valid *prefix* of some JSON document. States include `EXPECT_VALUE`, `EXPECT_KEY`, `IN_VALUE_STRING`, `STRING_ESCAPE`, `IN_NUMBER_INT`, `IN_NUMBER_FRAC`, `LITERAL` and `AFTER_VALUE`. The stack records the current container (object or array) and whether it already has entries, which is how trailing commas are rejected.

A token is checked by running its rendered text through a *clone* of the state, one character at a time. If every character is accepted, the token is legal. The real state is only updated once a token has been chosen.

### Typed values

| Type | Constraint |
|---|---|
| `number` / `integer` | Only digit, `.` and `-` tokens while inside the number. A "stop" token (comma, brace, bracket) is accepted only once the number is complete. |
| `string` | Printable ASCII, plus any non-ASCII characters already present in the prompt. Control characters are never allowed unescaped. |
| `boolean` | Only `true` or `false`. |

When the model prefers to end a value, it does so by ranking a terminator token highest. That vote is honoured by appending the clean closing token (`"` for strings), so the model decides *when* a value ends, and the program decides *how* it is written. Every value also has a hard length cap: on the last allowed step the program forces a clean close, so output is always valid JSON even if the model rambles.

### Tokenizer details

The vocabulary uses byte-level BPE. Raw token strings contain placeholder characters (`Ġ` for a space, `Ċ` for a newline, and so on). The program inverts the full GPT-2 byte-to-unicode table to get the true text of each token, and precomputes it once at startup.

## Design decisions

- **Inject what is known, decode what is not.** Running the mask over the full vocabulary at every step of every character of the output was far too slow. Injecting the skeleton and keys, and pre-filtering candidate tokens by type and grammar state, makes the run time practical.
- **Pydantic everywhere.** All classes are Pydantic models, including the grammar state and the vocabulary tables. Hot paths (state cloning) use `model_construct` to skip revalidation.
- **Graceful failure.** A failure on one prompt is reported on stderr and the run continues with the remaining prompts.
- **No heuristics for function choice.** The function is selected from the model's own logits under a constraint. There is no keyword matching and no regex on the prompt.

## Performance and reliability

| Metric | Result |
|---|---|
| Valid JSON output | <!-- fill in: x / y prompts --> |
| Correct function selected | <!-- fill in: x / y on your own test set --> |
| Correct arguments | <!-- fill in --> |
| Total run time | <!-- fill in: seconds for N prompts --> |

Syntactic validity comes from the constrained decoding itself. Correctness of the *choices* depends on the model and is measured on a hand-labelled test set.

## Challenges and limitations

- **Speed.** The SDK exposes no key-value cache, so each generated token recomputes over the whole context. Narrowing the candidate token pool by grammar state was the main mitigation.
- **Multilingual drift.** The model sometimes drifted into other languages inside free-form strings. Restricting string content to ASCII, plus characters present in the prompt, fixed this.
- **Missing end-of-value signals.** Byte-pair tokens such as `"}` merge a quote with punctuation. Rejecting every token containing a quote also removed the model's preferred way to close a string. Treating any token that starts with a quote as a "close now" vote fixed this.
- **Regex arguments.** The schema only says `string`, so nothing tells the decoder that a value is a regular expression. Patterns can be syntactically invalid or poor. A grammar for regular expressions was not implemented.
- **Unsupported types.** Parameters of type `array` and `object` are not supported by the value generator and are reported as errors for that prompt.

## Testing

```bash
uv run pytest -v
```

The tests cover the JSON grammar (accepted and rejected sequences, empty containers, trailing commas, escapes, literals, state-clone isolation), input loading (missing file, malformed JSON, schema violations) and multi-character token acceptance.

## Project structure

<!-- adjust to match your actual tree -->

```
.
├── Makefile
├── pyproject.toml
├── uv.lock
├── llm_sdk/              # provided SDK
├── data/
│   ├── input/
│   └── output/
├── src/
│   ├── __main__.py       # CLI entry point
│   ├── pipeline.py       # per-prompt orchestration
│   ├── generator.py      # masked generation and value decoding
│   ├── grammar.py        # JSON prefix state machine
│   ├── vocab.py          # token tables and byte-level rendering
│   ├── models.py         # Pydantic models for input and output
│   ├── loaders.py / writers.py
│   └── exceptions.py
└── tests/
```

## Resources

- Qwen3 model card: https://huggingface.co/Qwen/Qwen3-0.6B
- JSON specification: https://www.json.org
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), for the byte-level BPE scheme
- Willard and Louf, *Efficient Guided Generation for Large Language Models*, for the idea of masking logits with a finite-state machine
- Pydantic documentation: https://docs.pydantic.dev
- `uv` documentation: https://docs.astral.sh/uv/

### Use of AI

<!-- Edit this so it matches what you actually did. -->

An AI assistant (Claude) was used during development for:

- explaining how constrained decoding and the tokenizer's byte-level encoding work
- discussing the state-machine design for the JSON grammar
- reviewing and debugging code (stack handling, token filtering, mypy and pydantic errors)
- drafting the structure of this README and the test cases

All generated code was read, run, tested and modified by me before being kept, and I can explain every part of it.

