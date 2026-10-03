"""Bounded in-memory cache for Chatterbox speaker conditionals."""

from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from struct import pack, unpack
from time import perf_counter
from typing import Any


ConditioningKey = tuple[str, float]


class ConditionalsCache:
    """Keep the most recently used speaker conditionals without disk persistence."""

    def __init__(self, max_entries: int = 2) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self._entries: OrderedDict[ConditioningKey, Any] = OrderedDict()

    def get(self, key: ConditioningKey) -> Any | None:
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(self, key: ConditioningKey, value: Any) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def _hash_sample(sample_path: Path) -> tuple[str, float]:
    started_at = perf_counter()
    digest = sha256()
    with sample_path.open("rb") as sample_file:
        for block in iter(lambda: sample_file.read(128 * 1024), b""):
            digest.update(block)
    elapsed_ms = (perf_counter() - started_at) * 1000
    return digest.hexdigest(), elapsed_ms


def _canonical_float32(value: float) -> float:
    """Match the precision Chatterbox stores in its emotion conditioning tensor."""

    return unpack("!f", pack("!f", float(value)))[0]


def prepare_or_reuse_conditionals(
    model: Any,
    cache: ConditionalsCache,
    sample_path: Path,
    exaggeration: float,
) -> tuple[bool, float, float, float]:
    """Restore cached conditionals or prepare them once for this sample and setting.

    Returns ``(cache_hit, effective_exaggeration, sample_hash_ms, conditioning_ms)``.
    The caller must serialize this operation with model inference because Chatterbox
    stores active conditionals on the mutable model instance.
    """

    sample_hash, sample_hash_ms = _hash_sample(sample_path)
    effective_exaggeration = _canonical_float32(exaggeration)
    key = (sample_hash, effective_exaggeration)

    conditioning_started_at = perf_counter()
    conditionals = cache.get(key)
    cache_hit = conditionals is not None
    if cache_hit:
        model.conds = conditionals
    else:
        model.prepare_conditionals(
            str(sample_path),
            exaggeration=effective_exaggeration,
        )
        conditionals = getattr(model, "conds", None)
        if conditionals is None:
            raise RuntimeError("Chatterbox did not create speaker conditionals")
        cache.put(key, conditionals)

    conditioning_ms = (perf_counter() - conditioning_started_at) * 1000
    return cache_hit, effective_exaggeration, sample_hash_ms, conditioning_ms
