"""CLI entrypoint for the function-calling tool."""
import argparse
import sys

from llm_sdk import Small_LLM_Model

from .loaders import load_functions_definition, load_prompts
from .writers import write_results
from .vocab import Vocab
from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="src")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json"
                        )
    parser.add_argument("--output",
                        default="data/output/function_calling_results.json"
                        )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        functions = load_functions_definition(args.functions_definition)
        prompts = load_prompts(args.input)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    model = Small_LLM_Model()
    vocab = Vocab(model.get_path_to_vocab_file())

    results = run_pipeline(model, vocab, functions, prompts)
    write_results(results, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
