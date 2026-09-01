"""Constrained generation, restricted to only the genuinely uncertain parts."""

from llm_sdk import Small_LLM_Model

from grammar import GrammarState
from vocab import Vocab
from function_def import FunctionsDefinition, FunctionDefinition


def encode_prompt(model: Small_LLM_Model, prompt: str) -> list[int]:
    return model.encode(prompt)[0].tolist()


_TERMINATOR_STRINGS = (",", "}", "]")


def _terminator_ids(vocab: Vocab) -> list[int]:
    """Token ids for the universal JSON terminators (comma, closing braces)."""
    return [tid for tid, s in vocab.id_to_rendered.items() if s in _TERMINATOR_STRINGS]


def masked_argmax(logits: list[float], allowed_ids: list[int]) -> int:
    if not allowed_ids:
        raise ValueError("No allowed tokens at this generation step — grammar bug")
    best_id = allowed_ids[0]
    best_score = logits[best_id]
    for token_id in allowed_ids[1:]:
        if logits[token_id] > best_score:
            best_score, best_id = logits[token_id], token_id
    return best_id


def allowed_token_ids(vocab: Vocab, grammar: GrammarState, candidate_ids: list[int]) -> list[int]:
    """Only check a pre-narrowed candidate pool, not the whole vocab."""
    return [tid for tid in candidate_ids if grammar.token_is_acceptable(vocab.rendered(tid))]


def inject_literal(
    model: Small_LLM_Model, vocab: Vocab, input_ids: list[int], grammar: GrammarState, text: str
) -> None:
    """Force known text into the sequence directly — no masking, no model call."""
    for tid in encode_fragment(model, text):
        input_ids.append(tid)
        grammar.commit_token(vocab.rendered(tid))


def encode_fragment(model: Small_LLM_Model, text: str) -> list[int]:
    """Tokenize a text fragment the same way the SDK tokenizes prompts."""
    return model.encode(text)[0].tolist()


def build_name_pool(vocab: Vocab, names: list[str]) -> list[int]:
    """Tokens whose text appears inside some function name (built once, reused)."""
    return [
        tid for tid, s in vocab.id_to_rendered.items()
        if s and any(s in n for n in names)
    ]


def choose_function(
    model: Small_LLM_Model,
    vocab: Vocab,
    functions: FunctionsDefinition,
    input_ids: list[int],
    name_pool: list[int],
) -> tuple[FunctionDefinition, str]:
    """Constrained decoding of the name field: only prefixes of real names allowed."""
    names = [fn.name for fn in functions.root]
    quote_ids = [tid for tid, s in vocab.id_to_rendered.items() if s == '"']
    generated = ""

    for _ in range(max(len(n) for n in names) + 1):
        allowed = [
            tid for tid in name_pool
            if any(n.startswith(generated + vocab.rendered(tid)) for n in names)
        ]
        if generated in names:
            allowed += quote_ids  # the model may now choose to close the string
        logits = model.get_logits_from_input_ids(input_ids)
        next_id = masked_argmax(logits, allowed)
        if next_id in quote_ids:
            break
        input_ids.append(next_id)
        generated += vocab.rendered(next_id)

    fn = next(f for f in functions.root if f.name == generated)
    return fn, generated


def _could_be_string_content(token_text: str) -> bool:
    """Conservative pre-filter for string-value candidate tokens.

    Excludes anything that can PROVABLY never be valid inside a JSON string
    (raw control chars), while correctly allowing escaped quotes (`\\"`)
    and a bare closing quote — rather than blanket-rejecting any token that
    merely contains a `"` character. The authoritative, escape-state-aware
    check still happens in grammar.token_is_acceptable(); this just narrows
    the pool cheaply before that slower per-char walk runs.
    """
    if not token_text:
        return False
    for i, ch in enumerate(token_text):
        if ch in ("\n", "\t", "\r", "\x00"):
            return False  # always invalid unescaped in a JSON string
        if ch == '"':
            preceded_by_backslash = i > 0 and token_text[i - 1] == "\\"
            is_last_char = i == len(token_text) - 1
            if not preceded_by_backslash and not is_last_char:
                return False  # a bare mid-token quote can't be valid string content
    return True


def _candidate_ids_for_state(vocab: Vocab, grammar: GrammarState, kind: str) -> list[int]:
    """Pick a cheap candidate pool based on BOTH the declared type AND the
    grammar's current position — not just the type alone."""
    state_name = grammar.top_state.name

    if kind == "string":
        if state_name == "EXPECT_VALUE":
            # a string value may ONLY begin with an opening quote - this is
            # what was missing, and is exactly why 'null' leaked in
            return [tid for tid, s in vocab.id_to_rendered.items() if s == '"']
        if state_name == "IN_VALUE_STRING":
            return [
                tid for tid, s in vocab.id_to_rendered.items()
                if _could_be_string_content(s)
            ]
        if state_name == "STRING_ESCAPE":
            return [
                tid for tid, s in vocab.id_to_rendered.items()
                if s in ('"', "\\", "/", "n", "t", "r", "b", "f", "u")
            ]
        return []

    if kind in ("number", "integer"):
        if state_name == "EXPECT_VALUE":
            return [
                tid for tid, s in vocab.id_to_rendered.items()
                if s and (s[0].isdigit() or s[0] == "-")
                and all(c in "0123456789.-" for c in s)
            ]
        if state_name in ("IN_NUMBER_INT", "IN_NUMBER_FRAC"):
            digit_ids = [
                tid for tid, s in vocab.id_to_rendered.items()
                if s and all(c in "0123456789." for c in s)
            ]
            return digit_ids + _terminator_ids(vocab)
        return []

    if kind == "boolean":
        if state_name == "EXPECT_VALUE":
            return [tid for tid, s in vocab.id_to_rendered.items() if s in ("true", "false")]
        if state_name == "LITERAL":
            return [tid for tid, s in vocab.id_to_rendered.items() if s.isalpha()]
        return []

    return []


def generate_value_for_type(
    model: Small_LLM_Model,
    vocab: Vocab,
    input_ids: list[int],
    value_type: str,
    max_new_tokens: int = 20,
) -> str:
    grammar = GrammarState()
    start_len = len(input_ids)
    max_new_tokens = 60 if value_type == "string" else 20

    for step in range(max_new_tokens):
        if grammar.top_state.name in ("AFTER_VALUE", "DONE"):
            break

        candidate_ids = _candidate_ids_for_state(vocab, grammar, value_type)
        if step == max_new_tokens - 1:
            if grammar.top_state.name == "IN_VALUE_STRING":
                candidate_ids = [tid for tid, s in vocab.id_to_rendered.items() if s == '"']
            elif grammar.top_state.name in ("IN_NUMBER_INT", "IN_NUMBER_FRAC"):
                candidate_ids = _terminator_ids(vocab)
        if not candidate_ids:
            break

        logits = model.get_logits_from_input_ids(input_ids)
        allowed = allowed_token_ids(vocab, grammar, candidate_ids)
        if not allowed:
            break

        next_id = masked_argmax(logits, allowed)
        input_ids.append(next_id)
        grammar.commit_token(vocab.rendered(next_id))

    generated = input_ids[start_len:]
    return model.decode(generated)
