"""Constrained generation, restricted to only the genuinely uncertain parts."""
from llm_sdk import Small_LLM_Model
from typing import Any
from .grammar import GrammarState
from .vocab import Vocab
from .function_def import FunctionsDefinition, FunctionDefinition


def encode_prompt(model: Small_LLM_Model, prompt: str) -> Any:
    """Tokenize a prompt string into a list of token ids.

    Args:
        model: The language model whose tokenizer encodes the prompt.
        prompt: The raw prompt text to encode.

    Returns:
        The token ids for the prompt as a flat list of ints.
    """
    return model.encode(prompt)[0].tolist()


def _terminator_ids(vocab: Vocab) -> list[int]:
    """Collect the token ids that end a JSON value.

    Args:
        vocab: The vocabulary mapping token ids to their rendered text.

    Returns:
        The ids of tokens whose rendered text is exactly ",", "}", or "]".
    """
    return [tid for tid, s in vocab.id_to_rendered.items() if s in (
        ",", "}", "]")]


def masked_argmax(logits: list[float], allowed_ids: list[int]) -> int:
    """Return the permitted token id with the highest logit.

    Args:
        logits: Raw model scores, indexed by token id.
        allowed_ids: Token ids that are permitted at this step.

    Returns:
        The id from allowed_ids whose logit is largest. On ties, the
        earliest id in allowed_ids wins.

    Raises:
        ValueError: If allowed_ids is empty, which indicates a grammar bug.
    """
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
    """Filter a candidate pool down to the tokens the grammar accepts.

    Passing a pre-narrowed pool avoids checking the full vocabulary at
    every generation step.

    Args:
        vocab: The vocabulary mapping token ids to their rendered text.
        grammar: The current grammar state, used to test each token.
        candidate_ids: Token ids to check.

    Returns:
        The subset of candidate_ids whose rendered text the grammar
        accepts from its current state.
    """
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
    """Append fixed text to the sequence without sampling.

    The text is tokenized, and each token is appended to input_ids and
    committed to the grammar. The model is not called and no masking
    is applied.

    Args:
        model: The language model whose tokenizer encodes the text.
        vocab: The vocabulary mapping token ids to their rendered text.
        input_ids: The running token sequence. It is extended in place.
        grammar: The grammar state to advance with each committed token.
        text: The literal text to force into the sequence.
    """
    for tid in encode_fragment(model, text):
        input_ids.append(tid)
        grammar.commit_token(vocab.rendered(tid))


def encode_fragment(model: Small_LLM_Model, text: str) -> Any:
    """Tokenize a text fragment the same way the SDK tokenizes prompts.

    Args:
        model: The language model whose tokenizer encodes the text.
        text: The text fragment to encode.

    Returns:
        The token ids for the fragment as a flat list of ints.
    """
    return model.encode(text)[0].tolist()


def build_name_pool(vocab: Vocab, names: list[str]) -> list[int]:
    """Collect the tokens that occur as substrings of any function name.

    Args:
        vocab: The vocabulary mapping token ids to their rendered text.
        names: All valid function names.

    Returns:
        The ids of non-empty tokens whose rendered text appears inside
        at least one of the names.
    """
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
    """Decode a function name one token at a time under a prefix constraint.

    At each step, only tokens that keep the generated text a prefix of
    some real function name are allowed. Once the generated text is a
    complete name, a closing quote is also allowed, which ends the field.

    Args:
        model: The language model used to score the next token.
        vocab: The vocabulary mapping token ids to their rendered text.
        functions: The parsed function definitions to choose from.
        input_ids: The running token sequence. It is extended in place
            with the generated name tokens.
        name_pool: Candidate token ids, typically from build_name_pool.

    Returns:
        A tuple of the matching FunctionDefinition and the generated name.

    Raises:
        StopIteration: If decoding ends without producing a defined name.
    """
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
        logits = model.get_logits_from_input_ids(input_ids)
        next_id = masked_argmax(logits, allowed)
        if next_id in quote_ids:
            break
        input_ids.append(next_id)
        generated += vocab.rendered(next_id)

    fn = next(f for f in functions.root if f.name == generated)
    return fn, generated


def _could_be_string_content(token_text: str) -> bool:
    """Apply a conservative pre-filter for string-value candidate tokens.

    The filter rejects tokens that can never be valid inside a JSON
    string, such as raw control characters. It allows escaped quotes
    and a bare closing quote, rather than rejecting every token that
    contains a quote character. The escape-aware check in
    grammar.token_is_acceptable() remains the authoritative test; this
    function only narrows the pool cheaply before that slower check runs.

    Args:
        token_text: The rendered text of a candidate token.

    Returns:
        True if the token might be valid string content, False if it
        can never be valid inside a JSON string.
    """
    if not token_text:
        return False
    for i, ch in enumerate(token_text):
        if ch in ("\n", "\t", "\r", "\x00"):
            return False
        if ch == '"':
            preceded_by_backslash = i > 0 and token_text[i - 1] == "\\"
            is_last_char = i == len(token_text) - 1
            if not preceded_by_backslash and not is_last_char:
                return False
    return True


def _candidate_ids_for_state(
        vocab: Vocab, grammar: GrammarState, kind: str) -> list[int]:
    """Pick a cheap candidate pool based on the value type and grammar state.

    Both the declared type and the grammar's current position are used,
    so the pool depends on where the parser is, not only on the type.

    Args:
        vocab: The vocabulary mapping token ids to their rendered text.
        grammar: The grammar state, whose top state determines the pool.
        kind: The declared value type, such as "string", "number",
            "integer", or "boolean".

    Returns:
        The candidate token ids for this type and state. Returns an
        empty list for unsupported type and state combinations.
    """
    state_name = grammar._top_state.name

    if kind == "string":
        if state_name == "EXPECT_VALUE":
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
    """Collect candidate tokens for string content.

    A token qualifies if it passes the string pre-filter and every
    character in it is either printable ASCII or appears in the prompt.

    Args:
        vocab: The vocabulary mapping token ids to their rendered text.
        prompt_text: The prompt, whose non-ASCII characters are allowed
            in string values.

    Returns:
        The ids of tokens eligible to appear in generated string content.
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
) -> Any:
    """Generate a single JSON value of the given type.

    Strings are capped at 80 tokens and other types at 20. Generation
    stops when the grammar reaches a terminal state, when a closing
    vote token wins, or when the step limit is reached. Generated
    tokens are appended to input_ids in place.

    Args:
        model: The language model used to score each step.
        vocab: The vocabulary mapping token ids to their rendered text.
        input_ids: The running token sequence. It is extended in place
            with the generated value tokens.
        value_type: The declared type, such as "string", "number",
            "integer", or "boolean".
        string_pool: Candidate token ids for string content, typically
            from build_string_pool.

    Returns:
        The decoded text of the tokens generated for this value.
    """
    limit = 80 if value_type == "string" else 20
    grammar = GrammarState()
    start_len = len(input_ids)
    str_votes = set(vocab.string_close_votes)
    num_votes = set(vocab.number_close_votes)
    so_far = ""

    for step in range(limit):
        state = grammar._top_state.name
        if state in ("AFTER_VALUE", "DONE"):
            break
        logits = model.get_logits_from_input_ids(input_ids)

        if value_type == "string" and state == "IN_VALUE_STRING":
            content = allowed_token_ids(vocab, grammar, string_pool)
            winner = masked_argmax(logits, content + list(str_votes))
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
                break
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
