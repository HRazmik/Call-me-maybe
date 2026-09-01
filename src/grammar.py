"""Character-level JSON grammar as a stack-based state machine.

Tracks whether a partial string is a valid *prefix* of some eventual
valid JSON document, and which characters may legally come next.
"""
from __future__ import annotations

from enum import Enum, auto


class State(Enum):
    """A single grammar state within the current container frame."""

    EXPECT_VALUE = auto()       # about to write any JSON value
    EXPECT_KEY = auto()         # inside an object, about to write a key
    IN_KEY_STRING = auto()      # inside a key's quotes
    IN_VALUE_STRING = auto()    # inside a value's quotes
    STRING_ESCAPE = auto()      # just saw backslash inside a string
    EXPECT_COLON = auto()       # just closed a key string
    IN_NUMBER_INT = auto()      # integer part of a number
    IN_NUMBER_FRAC = auto()     # after the decimal point
    AFTER_VALUE = auto()        # just finished a value; expect ',' or closer
    LITERAL = auto()            # mid-way through true/false/null
    DONE = auto()                # whole document complete


class ContainerType(Enum):
    OBJECT = auto()
    ARRAY = auto()


class _Frame:
    """One level of nesting: what container we're in,
    and the state within it."""

    __slots__ = ("container", "state", "has_entries")

    def __init__(self, container: ContainerType, state: State) -> None:
        self.container = container
        self.state = state
        self.has_entries = False  # set True once the first ',' is consumed

    def copy(self) -> "_Frame":
        f = _Frame(self.container, self.state)
        f.has_entries = self.has_entries
        return f


class GrammarState:
    """Tracks legal-prefix status for a partial JSON string, char by char."""

    def __init__(self) -> None:
        self.top_state: State = State.EXPECT_VALUE
        self.stack: list[_Frame] = []
        self._literal_target: str = ""
        self._literal_progress: int = 0
        self._return_after_escape: State = State.IN_VALUE_STRING

    def copy(self) -> "GrammarState":
        clone = GrammarState()
        clone.top_state = self.top_state
        clone.stack = [f.copy() for f in self.stack]
        clone._literal_target = self._literal_target
        clone._literal_progress = self._literal_progress
        clone._return_after_escape = self._return_after_escape
        return clone
    # -- public API -------------------------------------------------

    def is_done(self) -> bool:
        return self.top_state is State.DONE

    def can_accept(self, ch: str) -> bool:
        """Would appending this single character
        keep the string a valid prefix?"""
        return self._transition(ch, dry_run=True)

    def advance(self, ch: str) -> None:
        """Commit: append this character, mutating state. Caller must have
        already confirmed can_accept(ch) is True."""
        ok = self._transition(ch, dry_run=False)
        if not ok:
            raise ValueError(f"advance() called with invalid char"
                             f"{ch!r} for state {self.top_state}")

    def token_is_acceptable(self, token_text: str) -> bool:
        """Trial-run a full token's rendered
        text against a COPY of this state."""
        trial = self.copy()
        for ch in token_text:
            if not trial.can_accept(ch):
                return False
            trial.advance(ch)
        return True

    def commit_token(self, token_text: str) -> None:
        """Apply an already-validated token's text to this (real) state."""
        for ch in token_text:
            self.advance(ch)

    # -- internal: the actual transition table -----------------------

    def _transition(self, ch: str, dry_run: bool) -> bool:
        state = self.top_state

        if state is State.EXPECT_VALUE:
            if ch == '"':
                new_state = State.IN_VALUE_STRING
            elif ch == "{":
                if not dry_run:
                    self.stack.append(_Frame(ContainerType.OBJECT,
                                             State.EXPECT_KEY))
                new_state = State.EXPECT_KEY if not dry_run else state
                # handled below
                if dry_run:
                    return True
                self.top_state = State.EXPECT_KEY
                return True
            elif ch == "[":
                if not dry_run:
                    self.stack.append(_Frame(ContainerType.ARRAY,
                                             State.EXPECT_VALUE))
                    self.top_state = State.EXPECT_VALUE
                return True
            elif (ch == "]" and self.stack
                  and self.stack[-1].container is ContainerType.ARRAY
                  and not self.stack[-1].has_entries):
                return self._close_container(not dry_run)
            elif ch.isdigit() or ch == "-":
                new_state = State.IN_NUMBER_INT
            elif ch in ("t", "f", "n"):
                if dry_run:
                    return True
                self._literal_target = {"t": "true",
                                        "f": "false",
                                        "n": "null"}[ch]
                self._literal_progress = 1
                self.top_state = State.LITERAL
                return True
            else:
                return False
            if not dry_run:
                self.top_state = new_state
            return True

        if state is State.EXPECT_KEY:
            if ch == '"':
                new_state = State.IN_KEY_STRING
            elif (ch == "}" and self.stack and not self.stack[-1].has_entries):
                return self._close_container(not dry_run)
            else:
                return False
            if not dry_run:
                self.top_state = new_state
            return True

        if state is State.IN_KEY_STRING:
            if ch == '"':
                if not dry_run:
                    self.top_state = State.EXPECT_COLON
                return True
            if ch == "\\":
                if not dry_run:
                    self.top_state = State.STRING_ESCAPE
                    self._return_after_escape = State.IN_KEY_STRING
                    # type: ignore[attr-defined]
                return True
            return ch != '"'  # any char except an unescaped quote

        if state is State.IN_VALUE_STRING:
            if ch == '"':
                if not dry_run:
                    self.top_state = State.AFTER_VALUE
                return True
            if ch == "\\":
                if not dry_run:
                    self.top_state = State.STRING_ESCAPE
                    self._return_after_escape = State.IN_VALUE_STRING
                    # type: ignore[attr-defined]
                return True
            return True  # any other char is fine inside a string

        if state is State.STRING_ESCAPE:
            if ch in ('"', "\\", "/", "n", "t", "r", "b", "f", "u"):
                if not dry_run:
                    self.top_state = self._return_after_escape
                    # type: ignore[attr-defined]
                return True
            return False

        if state is State.EXPECT_COLON:
            if ch == ":":
                if not dry_run:
                    self.top_state = State.EXPECT_VALUE
                return True
            return False

        if state is State.IN_NUMBER_INT:
            if ch.isdigit():
                return True
            if ch == ".":
                if not dry_run:
                    self.top_state = State.IN_NUMBER_FRAC
                return True
            if not dry_run:
                self.top_state = State.AFTER_VALUE
            return self._peek_after_value(ch, commit=not dry_run)

        if state is State.IN_NUMBER_FRAC:
            if ch.isdigit():
                return True
            if not dry_run:
                self.top_state = State.AFTER_VALUE
            return self._peek_after_value(ch, commit=not dry_run)

        if state is State.LITERAL:
            expected = self._literal_target[self._literal_progress]
            if ch != expected:
                return False
            if not dry_run:
                self._literal_progress += 1
                if self._literal_progress == len(self._literal_target):
                    self.top_state = State.AFTER_VALUE
            return True

        if state is State.AFTER_VALUE:
            return self._peek_after_value(ch, commit=not dry_run)
        return False  # DONE, or unknown state: nothing more is valid

    def _peek_after_value(self, ch: str, commit: bool = False) -> bool:
        if not self.stack:
            return False
        frame = self.stack[-1]
        if ch == ",":
            if commit:
                frame.has_entries = True
                self.top_state = (
                    State.EXPECT_KEY if frame.container is ContainerType.OBJECT
                    else State.EXPECT_VALUE
                )
            return True
        if ch == "}" and frame.container is ContainerType.OBJECT:
            return self._close_container(commit)
        if ch == "]" and frame.container is ContainerType.ARRAY:
            return self._close_container(commit)
        return False

    def _close_container(self, commit: bool) -> bool:
        if not commit:
            return True
        self.stack.pop()
        if not self.stack:
            self.top_state = State.DONE
        else:
            self.top_state = State.AFTER_VALUE
        return True
