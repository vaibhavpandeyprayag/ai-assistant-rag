"""Conservative text preprocessing applied before chunking.

The goal is deterministic normalization, not linguistic analysis:

- unify Windows/Mac line endings to ``\\n``
- convert tabs to spaces
- drop control/format characters (Unicode categories ``Cc``/``Cf``), keeping
  newlines
- collapse runs of spaces and runs of blank lines, preserving paragraph
  breaks (``\\n\\n``) so downstream recursive splitting can exploit them

The function is idempotent: cleaning already-clean text is a no-op.
"""

import re
import unicodedata

# Pre-render control characters to remove (everything in Cc/Cf except \n).
_STRIP_CHARS = {
    char
    for char in map(chr, range(0x110000))
    if unicodedata.category(char) in {"Cc", "Cf"} and char != "\n"
}
_STRIP_TABLE = {ord(char): None for char in _STRIP_CHARS}

_MULTIPLE_SPACES = re.compile(r" {2,}")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Return a normalized copy of ``text``; never returns leading/trailing whitespace."""
    # Unify line endings (Windows CRLF and legacy CR).
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # Tabs behave like single horizontal whitespace.
    normalized = normalized.replace("\t", " ")
    # Remove control/format characters such as NUL, BEL, BOM, zero-width marks.
    normalized = normalized.translate(_STRIP_TABLE)
    # Collapse horizontal and vertical whitespace runs.
    normalized = _MULTIPLE_SPACES.sub(" ", normalized)
    normalized = _EXCESS_BLANK_LINES.sub("\n\n", normalized)
    return normalized.strip()
