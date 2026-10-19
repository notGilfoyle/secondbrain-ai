"""Deterministic character-based text chunking."""


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into ordered character windows with a fixed overlap.

    Empty and whitespace-only input produces no chunks. Chunk boundaries are
    based on Python string indices, so Unicode code points are kept intact.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text or text.isspace():
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        # Since overlap < chunk_size, this step is always at least one.
        start = end - overlap
    return chunks
