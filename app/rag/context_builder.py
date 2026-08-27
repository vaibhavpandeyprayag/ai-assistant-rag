"""Assembly of retrieved chunks into a single grounded context block.

Chunks are formatted as numbered, attributed blocks::

    [1] (guide.txt, p.3)
    <chunk text>

Paging attribution uses the stored ``page_number`` when present; line-integer
page references are rendered as ``p.3`` and missing pages as ``p.?``.
"""

from dataclasses import dataclass

from app.vectordb.base import RetrievedChunk


@dataclass(slots=True)
class BuiltContext:
    """Result of assembling retrieved chunks into context."""

    text: str
    #: Indexes (into the input chunk list) of the chunks included in ``text``.
    included: list[int]


class ContextBuilder:
    """Build a bounded, attributed context block from retrieved chunks."""

    def __init__(self, *, max_chars: int = 6000) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self._max_chars = max_chars

    @staticmethod
    def _block(chunk: RetrievedChunk, index: int) -> str:
        metadata = chunk.metadata
        page = metadata.get("page_number")
        filename = metadata.get("filename", "unknown")
        page_label = f"p.{page}" if isinstance(page, int) else "p.?"
        return f"[{index + 1}] ({filename}, {page_label})\n{chunk.text}"

    def build(self, chunks: list[RetrievedChunk]) -> BuiltContext:
        """Assemble chunks until the character budget is exhausted.

        Chunks are consumed in the given (retrieval) order. A chunk whose text
        alone exceeds the remaining budget is truncated rather than dropped, so
        even an oversized match keeps contributing to the context.
        """
        if not chunks:
            return BuiltContext(text="", included=[])

        parts: list[str] = []
        used = 0
        included: list[int] = []

        for index, chunk in enumerate(chunks):
            block = self._block(chunk, index)
            if used + len(block) <= self._max_chars:
                parts.append(block)
                used += len(block)
            elif not parts and used == 0:
                # First block alone exceeds the budget: keep a truncated version.
                parts.append(block[: self._max_chars])
                used = len(parts[0])
            else:
                # Would exceed budget and we already have content: stop.
                break
            included.append(index)

        return BuiltContext(text="\n\n".join(parts), included=included)
