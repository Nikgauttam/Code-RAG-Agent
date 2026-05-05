"""Tests for the Python-managed ID vector store."""

from __future__ import annotations

import numpy as np
import pytest

from core.retrieval.vector_store import VectorStore


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def test_add_and_search_returns_nearest_first() -> None:
    rng = np.random.default_rng(0)
    vecs = _normalize(rng.standard_normal((10, 8)).astype(np.float32))
    ids = np.arange(100, 110, dtype=np.int64)

    store = VectorStore(dimension=8, metric="ip")
    store.add(vecs, ids)
    assert store.size == 10

    scores, returned_ids = store.search(vecs[3], top_k=3)
    # The exact vector must always be the top hit (cosine == 1.0).
    assert returned_ids[0][0] == 103
    assert pytest.approx(scores[0][0], rel=1e-4) == 1.0


def test_remove_drops_vectors_by_id() -> None:
    rng = np.random.default_rng(1)
    vecs = _normalize(rng.standard_normal((5, 4)).astype(np.float32))
    ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)

    store = VectorStore(dimension=4, metric="ip")
    store.add(vecs, ids)

    removed = store.remove(np.array([20, 40], dtype=np.int64))
    assert removed == 2
    assert store.size == 3

    _, returned_ids = store.search(vecs[1], top_k=5)
    assert 20 not in returned_ids[0].tolist()
    assert 40 not in returned_ids[0].tolist()


def test_save_and_load_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(2)
    vecs = _normalize(rng.standard_normal((6, 4)).astype(np.float32))
    ids = np.arange(0, 6, dtype=np.int64)

    store = VectorStore(dimension=4)
    store.add(vecs, ids)
    path = str(tmp_path / "test.index")
    store.save(path)

    fresh = VectorStore(dimension=4)
    fresh.load(path)
    assert fresh.size == 6

    _, ids_back = fresh.search(vecs[2], top_k=1)
    assert ids_back[0][0] == 2


def test_backwards_compat_add_embeddings() -> None:
    rng = np.random.default_rng(3)
    vecs = _normalize(rng.standard_normal((4, 4)).astype(np.float32))

    store = VectorStore(dimension=4)
    store.add_embeddings(vecs)
    assert store.size == 4
    _, ids_back = store.search(vecs[0], top_k=1)
    assert ids_back[0][0] == 0
