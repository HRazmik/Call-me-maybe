"""Custom exceptions for graceful error handling."""


class InputFileError(Exception):
    """Raised when an input file is missing, malformed, or fails validation."""
