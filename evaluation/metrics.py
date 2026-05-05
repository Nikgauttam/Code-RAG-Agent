"""Retrieval evaluation metrics.

A retrieved item is "relevant" if its file path matches one of the
ground-truth expected files for the question.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def recall_at_k(retrieved_files: Sequence[str], expected_files: Iterable[str], k: int) -> float:
    """Fraction of expected files that appear in the top-k retrieved files."""
    expected = set(expected_files)
    if not expected:
        return 0.0
    top_k = list(dict.fromkeys(retrieved_files))[:k]  # de-dup, preserve order
    hits = sum(1 for f in top_k if f in expected)
    return hits / len(expected)


def reciprocal_rank(retrieved_files: Sequence[str], expected_files: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant file (0 if none)."""
    expected = set(expected_files)
    seen = []
    for f in retrieved_files:
        if f in seen:
            continue
        seen.append(f)
        if f in expected:
            return 1.0 / len(seen)
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
