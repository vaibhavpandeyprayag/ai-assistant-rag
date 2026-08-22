"""Loader tests for TXT and Markdown formats."""

from pathlib import Path

from app.ingestion.loaders import load_document


def test_loads_txt_file(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Hello world\nSecond line", encoding="utf-8")

    loaded = load_document(file_path)

    assert loaded.filename == "notes.txt"
    assert loaded.source == str(file_path)
    assert len(loaded.sections) == 1
    assert loaded.sections[0].page_number is None
    assert "Hello world" in loaded.sections[0].text
    assert "Second line" in loaded.full_text


def test_loads_markdown_file_as_plain_text(tmp_path: Path) -> None:
    """Markdown is extracted verbatim; markdown-specific parsing is out of scope."""
    file_path = tmp_path / "readme.md"
    file_path.write_text("# Title\n\nSome **markdown** body.", encoding="utf-8")

    loaded = load_document(file_path)

    assert "# Title" in loaded.full_text
    assert "Some **markdown** body." in loaded.full_text


def test_extension_case_is_ignored(tmp_path: Path) -> None:
    file_path = tmp_path / "uppercase.TXT"
    file_path.write_text("case insensitive", encoding="utf-8")

    loaded = load_document(file_path)

    assert "case insensitive" in loaded.full_text
