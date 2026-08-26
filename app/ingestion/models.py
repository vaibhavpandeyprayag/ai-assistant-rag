"""Internal document models shared by the ingestion pipeline."""

from dataclasses import dataclass


@dataclass(slots=True)
class DocumentSection:
    """A contiguous piece of extracted text, tied to a page when available."""

    text: str
    #: 1-based page number; ``None`` when the format has no page concept.
    page_number: int | None = None


@dataclass(slots=True)
class LoadedDocument:
    """Result of loading a source document: raw text sections plus metadata."""

    filename: str
    source: str
    sections: list[DocumentSection]

    @property
    def full_text(self) -> str:
        """All section text joined with blank lines between pages."""
        return "\n\n".join(section.text for section in self.sections)


@dataclass(slots=True)
class IngestionResult:
    """Outcome of a single file ingestion."""

    document_id: str
    filename: str
    n_chunks: int
    status: str
    elapsed_ms: float


@dataclass(slots=True)
class Chunk:
    """A retrieval-ready text unit with complete source metadata."""

    #: Stable identifier in the form ``"{document_id}::{chunk_index}"``.
    chunk_id: str
    document_id: str
    #: Zero-based position of the chunk within its document.
    chunk_index: int
    text: str
    #: Page the chunk starts on; ``None`` when the format has no pages.
    page_number: int | None
    filename: str
    source: str
