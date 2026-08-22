"""Error-handling tests for document loading."""

from pathlib import Path

import pytest

from app.errors import (
    DocumentParseError,
    EmptyDocumentError,
    UnsupportedFormatError,
)
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document


def test_unsupported_extension_raises_typed_error(tmp_path: Path) -> None:
    file_path = tmp_path / "image.rtf"
    file_path.write_text("{\\rtf1 fake}", encoding="utf-8")

    with pytest.raises(UnsupportedFormatError):
        load_document(file_path)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "missing.pdf")


def test_corrupt_pdf_raises_parse_error(tmp_path: Path) -> None:
    file_path = tmp_path / "broken.pdf"
    file_path.write_bytes(b"%PDF-1.4 this is not a real pdf body")

    with pytest.raises(DocumentParseError):
        load_document(file_path)


def test_corrupt_docx_raises_parse_error(tmp_path: Path) -> None:
    file_path = tmp_path / "broken.docx"
    file_path.write_bytes(b"definitely not a zip archive")

    with pytest.raises(DocumentParseError):
        load_document(file_path)


def test_non_utf8_text_file_raises_parse_error(tmp_path: Path) -> None:
    file_path = tmp_path / "binary.txt"
    file_path.write_bytes(b"\xff\xfe\xfa\x01")

    with pytest.raises(DocumentParseError):
        load_document(file_path)


@pytest.mark.parametrize("content", ["", "   \n\t  \n"])
def test_empty_or_whitespace_text_raises_empty_document_error(
    tmp_path: Path, content: str
) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text(content, encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        load_document(file_path)


def test_typed_errors_carry_http_metadata() -> None:
    error = UnsupportedFormatError("bad format")

    assert error.code == "unsupported_format"
    assert error.http_status == 415
    assert set(SUPPORTED_EXTENSIONS) == {".pdf", ".docx", ".txt", ".md"}
