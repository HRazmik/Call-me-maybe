"""Character-level JSON grammar as a stack-based state machine.

Tracks whether a partial string is a valid *prefix* of some eventual
valid JSON document, and which characters may legally come next.
"""
from __future__ import annotations

from enum import Enum, auto
from pydantic import BaseModel, PrivateAttr


class State(Enum):
    """A single grammar state within the current container frame.

    Each member marks a position in the JSON syntax, such as expecting a
    value, inside a string, or just after a completed value.
    """
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
    """The kind of JSON container a stack frame represents."""
    OBJECT = auto()
    ARRAY = auto()


class _Frame(BaseModel):
    """One level of nesting: what container we're in, and the state within it.

    Attributes:
        _container: Whether this frame is an object or an array.
        _state: The grammar state within this container.
        _has_entries: True once at least one entry has been completed,
            which determines whether a closing bracket is still legal
            without a preceding value.
    """
    _container: ContainerType = PrivateAttr()
    _state: State = PrivateAttr()
    _has_entries: bool = PrivateAttr()
    __slots__ = ("_container", "_state", "_has_entries")

    def __init__(self, container: ContainerType, state: State) -> None:
        """Create a frame for a container at the given state.

        Args:
            container: The kind of container this frame represents.
            state: The initial grammar state within the container.
        """
        self._container = container
        self._state = state
        self._has_entries = False

    def clone(self) -> "_Frame":
        """Create an independent clone of this frame.

        Returns:
            A new _Frame with the same container, state, and has_entries
            values as this one.
        """
        f = _Frame(self._container, self._state)
        f._has_entries = self._has_entries
        return f


class GrammarState(BaseModel):
    """Tracks legal-prefix status for a partial JSON string, char by char.

    Attributes:
        _top_state: The grammar state at the innermost position.
        _stack: The open containers, from outermost to innermost.
    """
    _top_state: State = PrivateAttr()
    _stack: list[_Frame] = PrivateAttr()
    _literal_target: str = PrivateAttr()
    _literal_progress: int = PrivateAttr()
    _return_after_escape: State = PrivateAttr()

    def __init__(self) -> None:
        """Create a grammar state that expects a top-level JSON value."""
        self._top_state: State = State.EXPECT_VALUE
        self._stack: list[_Frame] = []
        self._literal_target: str = ""
        self._literal_progress: int = 0
        self._return_after_escape: State = State.IN_VALUE_STRING

    def clone(self) -> "GrammarState":
        """Create an independent clone of this grammar state.

        Returns:
            A new GrammarState whose stack and literal tracking are
            copied, so changes to one do not affect the other.
        """
        clone = GrammarState()
        clone._top_state = self._top_state
        clone._stack = [f.clone() for f in self._stack]
        clone._literal_target = self._literal_target
        clone._literal_progress = self._literal_progress
        clone._return_after_escape = self._return_after_escape
        return clone

    def is_done(self) -> bool:
        """Report whether a complete JSON document has been read.

        Returns:
            True if the top-level value is closed, otherwise False.
        """
        return self._top_state is State.DONE

    def can_accept(self, ch: str) -> bool:
        """Check whether appending a character keeps the input a valid prefix.

        This does not modify the state.

        Args:
            ch: The single character to test.

        Returns:
            True if ch is a legal next character, otherwise False.
        """
        return self._transition(ch, dry_run=True)

    def advance(self, ch: str) -> None:
        """Append a character and update the state.

        The caller must confirm with can_accept() that ch is legal before
        calling this method.

        Args:
            ch: The single character to commit.

        Raises:
            ValueError: If ch is not a legal next character. This indicates
                that can_accept() was not checked first.
        """
        ok = self._transition(ch, dry_run=False)
        if not ok:
            raise ValueError(f"advance() called with invalid char"
                             f"{ch!r} for state {self._top_state}")

    def token_is_acceptable(self, token_text: str) -> bool:
        """Check whether a full token is legal from the current state.

        The check runs against a clone, so this state is not modified.

        Args:
            token_text: The rendered text of the token to test.

        Returns:
            True if every character of token_text can be accepted in
            sequence, otherwise False.
        """
        trial = self.clone()
        for ch in token_text:
            if not trial.can_accept(ch):
                return False
            trial.advance(ch)
        return True

    def commit_token(self, token_text: str) -> None:
        """Apply a validated token's text to this state.

        Args:
            token_text: The rendered text of a token that has already
                passed token_is_acceptable().
        """
        for ch in token_text:
            self.advance(ch)

    def _transition(self, ch: str, dry_run: bool) -> bool:
        """Compute the transition for one character from the current state.

        Args:
            ch: The character to process.
            dry_run: If True, check legality without modifying the state.
                If False, apply the transition.

        Returns:
            True if ch is legal from the current state, otherwise False.
        """
        state = self._top_state

        if state is State.EXPECT_VALUE:
            if ch == '"':
                newstate = State.IN_VALUE_STRING
            elif ch == "{":
                if not dry_run:
                    self._stack.append(_Frame(ContainerType.OBJECT,
                                       State.EXPECT_KEY))
                newstate = State.EXPECT_KEY if not dry_run else state
                if dry_run:
                    return True
                self._top_state = State.EXPECT_KEY
                return True
            elif ch == "[":
                if not dry_run:
                    self._stack.append(_Frame(ContainerType.ARRAY,
                                       State.EXPECT_VALUE))
                    self._top_state = State.EXPECT_VALUE
                return True
            elif (ch == "]" and self._stack
                  and self._stack[-1]._container is ContainerType.ARRAY
                  and not self._stack[-1]._has_entries):
                return self._closecontainer(not dry_run)
            elif ch.isdigit() or ch == "-":
                newstate = State.IN_NUMBER_INT
            elif ch in ("t", "f", "n"):
                if dry_run:
                    return True
                self._literal_target = {"t": "true",
                                        "f": "false",
                                        "n": "null"}[ch]
                self._literal_progress = 1
                self._top_state = State.LITERAL
                return True
            else:
                return False
            if not dry_run:
                self._top_state = newstate
            return True

        if state is State.EXPECT_KEY:
            if ch == '"':
                newstate = State.IN_KEY_STRING
            elif (ch == "}" and self._stack and
                  not self._stack[-1]._has_entries):
                return self._closecontainer(not dry_run)
            else:
                return False
            if not dry_run:
                self._top_state = newstate
            return True

        if state is State.IN_KEY_STRING:
            if ch == '"':
                if not dry_run:
                    self._top_state = State.EXPECT_COLON
                return True
            if ch == "\\":
                if not dry_run:
                    self._top_state = State.STRING_ESCAPE
                    self._return_after_escape = State.IN_KEY_STRING
                return True
            return ch != '"'
        if state is State.IN_VALUE_STRING:
            if ch == '"':
                if not dry_run:
                    self._top_state = State.AFTER_VALUE
                return True
            if ch == "\\":
                if not dry_run:
                    self._top_state = State.STRING_ESCAPE
                    self._return_after_escape = State.IN_VALUE_STRING
                return True
            return True

        if state is State.STRING_ESCAPE:
            if ch in ('"', "\\", "/", "n", "t", "r", "b", "f", "u"):
                if not dry_run:
                    self._top_state = self._return_after_escape
                return True
            return False

        if state is State.EXPECT_COLON:
            if ch == ":":
                if not dry_run:
                    self._top_state = State.EXPECT_VALUE
                return True
            return False

        if state is State.IN_NUMBER_INT:
            if ch.isdigit():
                return True
            if ch == ".":
                if not dry_run:
                    self._top_state = State.IN_NUMBER_FRAC
                return True
            if not dry_run:
                self._top_state = State.AFTER_VALUE
            return self._peek_after_value(ch, commit=not dry_run)

        if state is State.IN_NUMBER_FRAC:
            if ch.isdigit():
                return True
            if not dry_run:
                self._top_state = State.AFTER_VALUE
            return self._peek_after_value(ch, commit=not dry_run)

        if state is State.LITERAL:
            expected = self._literal_target[self._literal_progress]
            if ch != expected:
                return False
            if not dry_run:
                self._literal_progress += 1
                if self._literal_progress == len(self._literal_target):
                    self._top_state = State.AFTER_VALUE
            return True

        if state is State.AFTER_VALUE:
            return self._peek_after_value(ch, commit=not dry_run)
        return False

    def _peek_after_value(self, ch: str, commit: bool = False) -> bool:
        """Check or apply a separator or closer that may follow a value.

        A comma starts the next entry in the enclosing container. A closing
        brace or bracket must match the enclosing container type.

        Args:
            ch: The character that follows the value.
            commit: If True, apply the transition. If False, only check it.

        Returns:
            True if ch is a legal separator or closer in the current
            container, otherwise False. An empty stack always returns False.
        """
        if not self._stack:
            return False
        frame = self._stack[-1]
        if ch == ",":
            if commit:
                frame._has_entries = True
                self._top_state = (
                    State.EXPECT_KEY
                    if frame._container is ContainerType.OBJECT
                    else State.EXPECT_VALUE
                )
            return True
        if ch == "}" and frame._container is ContainerType.OBJECT:
            return self._closecontainer(commit)
        if ch == "]" and frame._container is ContainerType.ARRAY:
            return self._closecontainer(commit)
        return False

    def _closecontainer(self, commit: bool) -> bool:
        """Close the innermost container and update the parent state.

        Args:
            commit: If True, pop the stack and update the state. If False,
                only report that closing is legal.

        Returns:
            Always True. Closing is always legal once a caller has
            confirmed the closer matches the container type.
        """
        if not commit:
            return True
        self._stack.pop()
        if not self._stack:
            self._top_state = State.DONE
        else:
            self._top_state = State.AFTER_VALUE
        return True
