"""Vocab loading: bidirectional mapping between token IDs and token strings."""
import json
from pathlib import Path


# src/llm/vocab.py

def _gpt2_bytes_to_unicode() -> dict[int, str]:
    """The standard GPT-2 byte<->printable-unicode mapping table."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_UNICODE_TO_BYTE: dict[str, int] = {v: k for k, v in _gpt2_bytes_to_unicode().items()}


def render_token(token_str: str) -> str:
    """Decode a raw vocab token into its true rendered text, inverting the
    full byte-level encoding (handles Ġ=space, Ċ=newline, etc.) — not just
    the single Ġ marker."""
    try:
        raw_bytes = bytes(_UNICODE_TO_BYTE[ch] for ch in token_str)
        return raw_bytes.decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        # a lone token can be an incomplete multi-byte UTF-8 sequence
        # (split across adjacent tokens) - fall back to raw text for now
        return token_str


class Vocab:
    """Bidirectional token_id <-> token_string
    mapping loaded from the SDK's vocab file."""

    def __init__(self, vocab_path: str) -> None:
        """Load the vocab file.

        Args:
            vocab_path: path returned by
            Small_LLM_Model.get_path_to_vocab_file().
        """
        raw: dict[str, int] = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self.str_to_id: dict[str, int] = raw
        self.id_to_str: dict[int, str] = {v: k for k, v in raw.items()}
        self.id_to_rendered: dict[int, str] = {
            token_id: render_token(token_str)
            for token_id, token_str in self.id_to_str.items()
            }

    def token_str(self, token_id: int) -> str:
        """Return the string for a given token id."""
        return self.id_to_str[token_id]

    def rendered(self, token_id: int) -> str:
        """The actual text this token contributes when appended."""
        return self.id_to_rendered[token_id]

    def token_id(self, token: str) -> int:
        """Return the id for a given token string."""
        return self.str_to_id[token]

    def __len__(self) -> int:
        return len(self.id_to_str)
