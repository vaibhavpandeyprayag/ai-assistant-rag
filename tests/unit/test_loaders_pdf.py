"""Loader tests for the PDF format."""

from pathlib import Path

from fpdf import FPDF

from app.ingestion.loaders import load_document


def _write_pdf(path: Path, page_texts: list[str]) -> None:
    """Create a real PDF fixture containing one text line per page."""
    pdf = FPDF()
    for page_text in page_texts:
        pdf.add_page()
        pdf.set_font("helvetica", size=12)
        pdf.cell(0, 10, page_text)
    pdf.output(str(path))


def test_loads_each_page_with_page_number(tmp_path: Path) -> None:
    file_path = tmp_path / "report.pdf"
    _write_pdf(file_path, ["Alpha page content", "Beta page content"])

    loaded = load_document(file_path)

    assert loaded.filename == "report.pdf"
    assert [section.page_number for section in loaded.sections] == [1, 2]
    assert "Alpha page content" in loaded.sections[0].text
    assert "Beta page content" in loaded.sections[1].text


def test_full_text_joins_all_pages(tmp_path: Path) -> None:
    file_path = tmp_path / "multi.pdf"
    _write_pdf(file_path, ["first", "second", "third"])

    loaded = load_document(file_path)

    assert len(loaded.sections) == 3
    for marker in ("first", "second", "third"):
        assert marker in loaded.full_text
