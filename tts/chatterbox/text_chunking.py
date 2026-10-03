"""Lossless text chunking for engines with a short per-call limit."""

from __future__ import annotations

import re


MAX_CHUNK_CHARS = 300
MAX_TOTAL_TEXT_CHARS = 3000

_SENTENCE_BOUNDARY = re.compile(r"[.!?;:…。！？；：]+[\"'”’»）)\]}]*(?:\s+)")
_WORD_BOUNDARY = re.compile(r"\s+")


def split_text(
    text: str,
    *,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    max_total_chars: int = MAX_TOTAL_TEXT_CHARS,
) -> list[str]:
    """Split text at sentence/word boundaries, preserving every source character."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text cannot be empty")
    if max_chunk_chars < 1 or max_total_chars < 1:
        raise ValueError("text limits must be positive")
    if len(text) > max_total_chars:
        raise ValueError(
            f"text exceeds the {max_total_chars}-character generation limit; "
            "split the document into smaller passages"
        )

    chunks: list[str] = []
    offset = 0
    while offset < len(text):
        limit = min(offset + max_chunk_chars, len(text))
        if limit == len(text):
            chunks.append(text[offset:])
            break

        candidate = text[offset:limit]
        sentence_boundaries = list(_SENTENCE_BOUNDARY.finditer(candidate))
        word_boundaries = list(_WORD_BOUNDARY.finditer(candidate))
        if sentence_boundaries:
            end = offset + sentence_boundaries[-1].end()
        elif word_boundaries:
            end = offset + word_boundaries[-1].end()
        else:
            end = limit

        # Keep forward progress even for a long unbroken word or unusual Unicode.
        if end <= offset:
            end = limit
        chunks.append(text[offset:end])
        offset = end

    return chunks
