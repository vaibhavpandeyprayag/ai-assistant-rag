"""Unit tests for the conservative text cleaner."""

from app.ingestion.cleaning import clean_text


def test_unifies_line_endings() -> None:
    assert clean_text("a\r\nb\rc") == "a\nb\nc"


def test_tabs_become_single_spaces() -> None:
    assert clean_text("a\t\tb") == "a b"


def test_control_and_format_characters_removed_but_newlines_kept() -> None:
    dirty = "a\x00b\x07c\u200bd\ne"
    assert clean_text(dirty) == "abcd\ne"


def test_multiple_spaces_collapse_to_one() -> None:
    assert clean_text("too     many    spaces") == "too many spaces"


def test_blank_line_runs_collapse_but_paragraph_breaks_survive() -> None:
    cleaned = clean_text("para one\n\n\n\n\npara two")
    assert cleaned == "para one\n\npara two"


def test_strips_outer_whitespace() -> None:
    assert clean_text("  hello \n ") == "hello"


def test_cleaning_is_idempotent() -> None:
    dirty = "x\r\n\ty   z\u200b\n\n\n\nw  "
    once = clean_text(dirty)
    assert clean_text(once) == once


def test_empty_input_stays_empty() -> None:
    assert clean_text("") == ""
