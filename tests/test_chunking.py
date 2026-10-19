"""Unit tests for standalone text chunking."""

import pytest

from secondbrain.chunking import chunk_text


def test_empty_and_whitespace_input_produce_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text(" \t\n") == []


def test_short_text_produces_one_chunk() -> None:
    assert chunk_text("hello", chunk_size=10, overlap=2) == ["hello"]


def test_exact_size_text_produces_one_chunk() -> None:
    assert chunk_text("abcd", chunk_size=4, overlap=1) == ["abcd"]


def test_slightly_long_text_uses_configured_overlap() -> None:
    assert chunk_text("abcdef", chunk_size=4, overlap=2) == ["abcd", "cdef"]


def test_multiple_chunks_preserve_order_and_content() -> None:
    text = "abcdefghijkl"
    chunks = chunk_text(text, chunk_size=5, overlap=2)
    assert chunks == ["abcde", "defgh", "ghijk", "jkl"]
    assert chunks[0][-2:] == chunks[1][:2]
    assert chunks[1][-2:] == chunks[2][:2]
    assert chunks[2][-2:] == chunks[3][:2]
    assert (
        "".join(
            chunk if index == 0 else chunk[2:] for index, chunk in enumerate(chunks)
        )
        == text
    )


def test_unicode_text_is_preserved() -> None:
    text = "a😀é漢字b"
    chunks = chunk_text(text, chunk_size=3, overlap=1)
    assert chunks == ["a😀é", "é漢字", "字b"]


def test_chunk_size_one_with_zero_overlap_makes_progress() -> None:
    assert chunk_text("abc", chunk_size=1, overlap=0) == ["a", "b", "c"]


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_invalid_chunk_size_is_rejected(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("text", chunk_size=chunk_size)


def test_negative_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("text", chunk_size=4, overlap=-1)


@pytest.mark.parametrize("overlap", [4, 5])
def test_overlap_at_least_chunk_size_is_rejected(overlap: int) -> None:
    with pytest.raises(ValueError, match="smaller"):
        chunk_text("text", chunk_size=4, overlap=overlap)


def test_long_input_produces_many_chunks_and_preserves_reconstruction() -> None:
    text = "0123456789" * 1000
    chunks = chunk_text(text, chunk_size=17, overlap=3)
    assert len(chunks) > 100
    assert all(0 < len(chunk) <= 17 for chunk in chunks)
    assert "".join([chunks[0], *(chunk[3:] for chunk in chunks[1:])]) == text


def test_default_configuration() -> None:
    text = "x" * 2000
    chunks = chunk_text(text)
    assert [len(chunk) for chunk in chunks] == [1000, 1000, 300]
    assert chunks[0][-150:] == chunks[1][:150]
