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
