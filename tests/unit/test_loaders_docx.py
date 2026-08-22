"""Loader tests for the DOCX format."""

from pathlib import Path

from docx import Document as DocxDocumentBuilder

from app.ingestion.loaders import load_document


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    """Create a real DOCX fixture with one paragraph per entry."""
    document = DocxDocumentBuilder()
    for paragraph_text in paragraphs:
        document.add_paragraph(paragraph_text)
    document.save(str(path))


def test_loads_paragraph_text_without_page_numbers(tmp_path: Path) -> None:
    file_path = tmp_path / "letter.docx"
    _write_docx(file_path, ["First paragraph", "Second paragraph"])

    loaded = load_document(file_path)

    assert loaded.filename == "letter.docx"
    assert len(loaded.sections) == 1
    section = loaded.sections[0]
    assert section.page_number is None
    assert "First paragraph" in section.text
    assert "Second paragraph" in section.text


def test_empty_paragraph_only_document_still_has_structure(tmp_path: Path) -> None:
    """A DOCX with no text becomes an empty document at dispatch level."""
    file_path = tmp_path / "blank.docx"
    _write_docx(file_path, [])

    # The loader itself returns sections; emptiness is enforced by load_document.
    from app.errors import EmptyDocumentError

    try:
        load_document(file_path)
        raise AssertionError("EmptyDocumentError expected")
    except EmptyDocumentError:
        pass
