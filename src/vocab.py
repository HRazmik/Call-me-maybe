"""Vocab loading: bidirectional mapping between token IDs and token strings."""
import json
from pathlib import Path
from pydantic import BaseModel, PrivateAttr


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


_UNICODE_TO_BYTE: dict[str, int] = {
    v: k for k, v in _gpt2_bytes_to_unicode().items()
    }


def render_token(token_str: str) -> str:
    """Decode a raw vocab token into its true rendered text, inverting the
    full byte-level encoding (handles Ġ=space, Ċ=newline, etc.) — not just
    the single Ġ marker."""
    try:
        raw_bytes = bytes(_UNICODE_TO_BYTE[ch] for ch in token_str)
        return raw_bytes.decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return token_str


class Vocab(BaseModel):
    """Bidirectional token_id <-> token_string
    mapping loaded from the SDK's vocab file."""
    _str_to_id: dict[str, int] = PrivateAttr()
    _id_to_str: dict[int, str] = PrivateAttr()

    _id_to_rendered: dict[int, str] = PrivateAttr()
    _quote_id: int = PrivateAttr()
    _string_close_votes: list[int] = PrivateAttr()
    _number_close_votes: list[int] = PrivateAttr()
    _number_digit_ids: list[int] = PrivateAttr()

    def __init__(self, vocab_path: str) -> None:
        """Load the vocab file.

        Args:
            vocab_path: path returned by
            Small_LLM_Model.get_path_to_vocab_file().
        """
        raw: dict[str, int] = json.loads(
            Path(vocab_path).read_text(encoding="utf-8"))
        self._str_to_id: dict[str, int] = raw
        self._id_to_str: dict[int, str] = {v: k for k, v in raw.items()}
        self._id_to_rendered: dict[int, str] = {
            token_id: render_token(token_str)
            for token_id, token_str in self._id_to_str.items()
            }
        self._quote_id: int = next(
            t for t, s in self._id_to_rendered.items() if s == '"')
        self._string_close_votes: list[int] = [
            t for t, s in self._id_to_rendered.items() if s.startswith('"')
        ]
        self._number_close_votes: list[int] = [
            t for t, s in self._id_to_rendered.items() if s and s[0] in ",}]\n"
        ]
        self._number_digit_ids: list[int] = [
            t for t, s in self._id_to_rendered.items()
            if s and all(c in "0123456789." for c in s)
        ]

    @property
    def id_to_rendered(self) -> dict[int, str]:
        return self._id_to_rendered

    @property
    def quote_id(self) -> int:
        return self._quote_id

    @property
    def string_close_votes(self) -> list[int]:
        return self._string_close_votes

    @property
    def number_close_votes(self) -> list[int]:
        return self._number_close_votes

    @property
    def number_digit_ids(self) -> list[int]:
        return self._number_digit_ids

    def rendered(self, token_id: int) -> str:
        """The actual text this token contributes when appended."""
        return self._id_to_rendered[token_id]

    def token_str(self, token_id: int) -> str:
        return self._id_to_str[token_id]

    def token_id(self, token: str) -> int:
        """Return the id for a given token string."""
        return self._str_to_id[token]

    def __len__(self) -> int:
        return len(self._id_to_str)
