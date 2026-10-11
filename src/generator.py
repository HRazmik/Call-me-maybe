"""Constrained generation, restricted to only the genuinely uncertain parts."""
from llm_sdk import Small_LLM_Model

from grammar import GrammarState
from vocab import Vocab
from function_def import FunctionsDefinition, FunctionDefinition


def encode_prompt(model: Small_LLM_Model, prompt: str) -> list[int]:
    return model.encode(prompt)[0].tolist()


def _terminator_ids(vocab: Vocab) -> list[int]:
    """Token ids for the universal JSON terminators (comma, closing braces)."""
    return [tid for tid, s in vocab.id_to_rendered.items() if s in (
        ",", "}", "]")]


def masked_argmax(logits: list[float], allowed_ids: list[int]) -> int:
    if not allowed_ids:
        raise ValueError("No allowed tokens at this"
                         "generation step — grammar bug")
    best_id = allowed_ids[0]
    best_score = logits[best_id]
    for token_id in allowed_ids[1:]:
        if logits[token_id] > best_score:
            best_score, best_id = logits[token_id], token_id
    return best_id


def allowed_token_ids(
        vocab: Vocab,
        grammar: GrammarState,
        candidate_ids: list[int]
        ) -> list[int]:
    """Only check a pre-narrowed candidate pool, not the whole vocab."""
    return [
        tid for tid in candidate_ids if grammar.token_is_acceptable(
            vocab.rendered(tid))]


def inject_literal(
    model: Small_LLM_Model,
    vocab: Vocab,
    input_ids: list[int],
    grammar: GrammarState,
    text: str
) -> None:
    """Force known text into the sequence directly — no masking,
    no model call."""
    for tid in encode_fragment(model, text):
        input_ids.append(tid)
        grammar.commit_token(vocab.rendered(tid))


def encode_fragment(model: Small_LLM_Model, text: str) -> list[int]:
    """Tokenize a text fragment the same way the SDK tokenizes prompts."""
    return model.encode(text)[0].tolist()


def build_name_pool(vocab: Vocab, names: list[str]) -> list[int]:
    """Tokens whose text appears inside some function name."""
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
    """Constrained decoding of the name field:
    only prefixes of real names allowed."""
    names = [fn.name for fn in functions.root]
    quote_ids = [tid for tid, s in vocab.id_to_rendered.items() if s == '"']
    generated = ""

    for _ in range(max(len(n) for n in names) + 1):
        allowed = [
            tid for tid in name_pool
            if any(
                n.startswith(generated + vocab.rendered(tid)) for n in names)
        ]
        if generated in names:
            allowed += quote_ids
            # the model may now choose to close the string
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
                return False
            # a bare mid-token quote can't be valid string content
    return True


def _candidate_ids_for_state(
        vocab: Vocab, grammar: GrammarState, kind: str) -> list[int]:
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
            return [tid for tid, s in vocab.id_to_rendered.items() if s in (
                    "true", "false")]
        if state_name == "LITERAL":
            return [tid for tid, s in vocab.id_to_rendered.items()
                    if s.isalpha()]
        return []

    return []


def build_string_pool(vocab: Vocab, prompt_text: str) -> list[int]:
    """
    Candidate tokens for string content:
    ASCII plus chars present in the prompt.
    """
    ok = ({chr(c) for c in range(32, 127)} |
          {c for c in prompt_text if ord(c) > 126})
    return [
        tid for tid, s in vocab.id_to_rendered.items()
        if _could_be_string_content(s) and all(c in ok for c in s)
    ]


def generate_value_for_type(
    model: Small_LLM_Model,
    vocab: Vocab,
    input_ids: list[int],
    value_type: str,
    string_pool: list[int],
) -> str:
    limit = 80 if value_type == "string" else 20
    grammar = GrammarState()
    start_len = len(input_ids)
    str_votes = set(vocab.string_close_votes)
    num_votes = set(vocab.number_close_votes)
    so_far = ""

    for step in range(limit):
        state = grammar.top_state.name
        if state in ("AFTER_VALUE", "DONE"):
            break
        logits = model.get_logits_from_input_ids(input_ids)

        if value_type == "string" and state == "IN_VALUE_STRING":
            content = allowed_token_ids(vocab, grammar, string_pool)
            winner = masked_argmax(logits, content + list(str_votes))
            # last step: force a clean close so the output is always valid
            next_id = vocab.quote_id if (
                winner in str_votes or step == limit - 1) else winner

        elif (value_type in ("number", "integer")
                and state in ("IN_NUMBER_INT", "IN_NUMBER_FRAC")):
            digits = allowed_token_ids(vocab, grammar, vocab.number_digit_ids)
            can_stop = bool(so_far) and not so_far.endswith((".", "-"))
            pool = digits + (list(num_votes) if can_stop else [])
            if not pool:
                break
            winner = masked_argmax(logits, pool)
            if winner in num_votes or (step == limit - 1 and can_stop):
                break  # value is complete; the pipeline injects the separator
            next_id = winner

        else:
            candidates = _candidate_ids_for_state(vocab, grammar, value_type)
            allowed = allowed_token_ids(vocab, grammar, candidates)
            if not allowed:
                break
            next_id = masked_argmax(logits, allowed)

        input_ids.append(next_id)
        text = vocab.rendered(next_id)
        grammar.commit_token(text)
        so_far += text

    return model.decode(input_ids[start_len:])
