"""Unit tests for the configurable chunker."""

import pytest

from app.ingestion.chunking import chunk_document, chunk_text
from app.ingestion.models import DocumentSection, LoadedDocument

DOCUMENT_ID = "doc123"


def _document(*sections: tuple[str, int | None]) -> LoadedDocument:
    return LoadedDocument(
        filename="file.txt",
        source="data/uploads/file.txt",
        sections=[DocumentSection(text=text, page_number=page) for text, page in sections],
    )


class TestChunkTextFixedStrategy:
    def test_windows_respect_size_and_overlap(self) -> None:
        # 50 characters; size 20 + overlap 5 => step 15 => windows at 0, 15, 30.
        text = "0123456789" * 5

        pieces = chunk_text(text, chunk_size=20, chunk_overlap=5, strategy="fixed")

        assert pieces[0] == text[0:20]
        assert pieces[1] == text[15:35]
        assert pieces[2] == text[30:50]
        # Consecutive windows share the overlap region.
        assert pieces[1][:5] == pieces[0][15:20]

    def test_short_text_returns_single_chunk(self) -> None:
        assert chunk_text("tiny", chunk_size=100, chunk_overlap=10, strategy="fixed") == ["tiny"]


class TestChunkTextRecursiveStrategy:
    def test_pieces_never_exceed_size_and_split_long_text(self) -> None:
        sentence = "The quick brown fox jumps over the lazy dog. "
        text = sentence * 20  # ~900 characters

        pieces = chunk_text(text, chunk_size=150, chunk_overlap=30)

        assert len(pieces) > 1
        assert all(len(piece) <= 150 for piece in pieces)
        assert all(piece.strip() for piece in pieces)


class TestChunkTextValidation:
    @pytest.mark.parametrize(
        ("size", "overlap"),
        [(0, 0), (-10, 0), (100, 100), (100, 150), (100, -1)],
    )
    def test_invalid_parameters_raise_value_error(self, size: int, overlap: int) -> None:
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size=size, chunk_overlap=overlap)


class TestChunkTextEdgeCases:
    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk_text("", chunk_size=100, chunk_overlap=10) == []
        assert chunk_text("   \n\t ", chunk_size=100, chunk_overlap=10) == []

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("abc", chunk_size=10, chunk_overlap=1, strategy="magic")


class TestChunkDocument:
    def test_chunks_carry_metadata_and_sequential_ids(self) -> None:
        long_text = "sentence here. " * 30  # forces several small chunks
        loaded = _document((long_text, 3), ("second part", None))

        chunks = chunk_document(
            loaded,
            DOCUMENT_ID,
            chunk_size=80,
            chunk_overlap=10,
            chunk_strategy="fixed",
        )

        assert len(chunks) > 2
        assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
        assert all(chunk.chunk_id == f"{DOCUMENT_ID}::{chunk.chunk_index}" for chunk in chunks)
        assert all(chunk.document_id == DOCUMENT_ID for chunk in chunks)
        assert all(chunk.filename == "file.txt" for chunk in chunks)
        assert all(chunk.source == "data/uploads/file.txt" for chunk in chunks)

    def test_page_number_inherited_from_own_section(self) -> None:
        long_text = "content " * 40
        loaded = _document((long_text, 7), ("tail", None))

        chunks = chunk_document(
            loaded,
            DOCUMENT_ID,
            chunk_size=60,
            chunk_overlap=10,
            chunk_strategy="fixed",
        )

        pages = {chunk.page_number for chunk in chunks}
        assert pages == {7, None}
        # Chunks of the paged section all carry that exact page number.
        paged = [chunk for chunk in chunks if chunk.page_number == 7]
        assert len(paged) > 1

    def test_whitespace_section_is_skipped(self) -> None:
        loaded = _document(("   \n ", 2), ("real content", 3))

        chunks = chunk_document(loaded, DOCUMENT_ID, chunk_size=100, chunk_overlap=10)

        assert [chunk.text for chunk in chunks] == ["real content"]
        assert chunks[0].page_number == 3

    def test_fully_empty_document_yields_no_chunks(self) -> None:
        loaded = _document(("", 1))

        assert (
            chunk_document(loaded, DOCUMENT_ID, chunk_size=100, chunk_overlap=10) == []
        )
