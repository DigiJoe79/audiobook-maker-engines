"""Regression coverage for Chatterbox cache and lossless chunk boundaries."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tts.chatterbox.conditioning_cache import (
    ConditionalsCache,
    prepare_or_reuse_conditionals,
)
from tts.chatterbox.text_chunking import split_text


class FakeModel:
    def __init__(self) -> None:
        self.conds = None
        self.preparations: list[tuple[str, float, object]] = []

    def prepare_conditionals(self, sample_path: str, *, exaggeration: float) -> None:
        conditionals = object()
        self.preparations.append((sample_path, exaggeration, conditionals))
        self.conds = conditionals


class ConditionalsCacheTests(unittest.TestCase):
    def _temporary_directory(self) -> tempfile.TemporaryDirectory[str]:
        return tempfile.TemporaryDirectory()

    def test_cache_is_bounded_lru_and_can_be_cleared(self) -> None:
        cache = ConditionalsCache(max_entries=2)
        cache.put(("sample-a", 0.5), "a")
        cache.put(("sample-b", 0.5), "b")
        self.assertEqual(cache.get(("sample-a", 0.5)), "a")
        cache.put(("sample-c", 0.5), "c")

        self.assertIsNone(cache.get(("sample-b", 0.5)))
        self.assertEqual(cache.get(("sample-a", 0.5)), "a")
        self.assertEqual(cache.get(("sample-c", 0.5)), "c")
        self.assertEqual(len(cache), 2)

        cache.clear()
        self.assertEqual(len(cache), 0)

    def test_same_sample_content_reuses_conditionals(self) -> None:
        with self._temporary_directory() as directory:
            first_sample = Path(directory) / "first.wav"
            same_sample = Path(directory) / "same-content.wav"
            first_sample.write_bytes(b"private sample bytes")
            same_sample.write_bytes(b"private sample bytes")
            model = FakeModel()
            cache = ConditionalsCache()

            first = prepare_or_reuse_conditionals(model, cache, first_sample, 0.8)
            first_conditionals = model.conds
            second = prepare_or_reuse_conditionals(model, cache, same_sample, 0.8)

            self.assertFalse(first[0])
            self.assertTrue(second[0])
            self.assertEqual(first[1], second[1])
            self.assertEqual(len(model.preparations), 1)
            self.assertIs(model.conds, first_conditionals)

    def test_sample_content_and_exaggeration_invalidate_the_key(self) -> None:
        with self._temporary_directory() as directory:
            sample = Path(directory) / "speaker.wav"
            sample.write_bytes(b"version one")
            model = FakeModel()
            cache = ConditionalsCache()

            self.assertFalse(prepare_or_reuse_conditionals(model, cache, sample, 0.5)[0])
            self.assertFalse(prepare_or_reuse_conditionals(model, cache, sample, 0.7)[0])
            sample.write_bytes(b"version two")
            self.assertFalse(prepare_or_reuse_conditionals(model, cache, sample, 0.5)[0])

            self.assertEqual(len(model.preparations), 3)


class TextChunkingTests(unittest.TestCase):
    def test_short_text_is_returned_unchanged(self) -> None:
        self.assertEqual(split_text("A short passage.  "), ["A short passage.  "])

    def test_prefers_sentence_boundaries_and_preserves_all_characters(self) -> None:
        first_sentence = "First sentence. "
        second_sentence = "Second sentence! "
        source = first_sentence * 12 + second_sentence * 12

        chunks = split_text(source, max_chunk_chars=64)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), source)
        self.assertTrue(all(len(chunk) <= 64 for chunk in chunks))
        self.assertTrue(all(chunk.endswith(" ") for chunk in chunks[:-1]))

    def test_falls_back_to_word_and_hard_boundaries_without_loss(self) -> None:
        word_text = "alpha beta gamma delta epsilon zeta"
        hard_text = "x" * 41

        word_chunks = split_text(word_text, max_chunk_chars=12)
        hard_chunks = split_text(hard_text, max_chunk_chars=12)

        self.assertEqual("".join(word_chunks), word_text)
        self.assertTrue(all(len(chunk) <= 12 for chunk in word_chunks))
        self.assertEqual("".join(hard_chunks), hard_text)
        self.assertTrue(all(len(chunk) <= 12 for chunk in hard_chunks))

    def test_default_limit_splits_301_characters_without_truncation(self) -> None:
        source = "x" * 301

        chunks = split_text(source)

        self.assertEqual([len(chunk) for chunk in chunks], [300, 1])
        self.assertEqual("".join(chunks), source)

    def test_rejects_empty_and_oversized_text_with_recovery_hint(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            split_text("  \n")
        with self.assertRaisesRegex(ValueError, "split the document"):
            split_text("x" * 11, max_total_chars=10)


if __name__ == "__main__":
    unittest.main()
