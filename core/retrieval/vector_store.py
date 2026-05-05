"""FAISS-backed vector store with Python-managed ID mapping.

We use ``IndexFlatIP`` (cosine similarity for L2-normalised vectors) with a
Python dict for ``faiss_row_idx -> stable_int_id`` and an inverse mapping.
This gives us:

  * Delete-by-stable-id: mark the row as vacant; compact on the next add.
  * Stable int ids that survive save/load.
  * Full compatibility with faiss-cpu on macOS (IDMap2 segfaults on some
    builds; Python-side mapping avoids all native ID-map code paths).
"""

from __future__ import annotations

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int, metric: str = "ip"):
        self.dimension = dimension
        self.metric = metric
        self.index: faiss.Index = self._make_index(dimension, metric)
        # row_idx -> stable_int_id (populated as we add vectors)
        self._row_to_id: list[int] = []
        # stable_int_id -> row_idx (for O(1) delete lookup)
        self._id_to_row: dict[int, int] = {}
        # rows that have been logically deleted (filled with zeros)
        self._deleted: set[int] = set()

    @staticmethod
    def _make_index(dimension: int, metric: str) -> faiss.Index:
        if metric == "ip":
            return faiss.IndexFlatIP(dimension)
        return faiss.IndexFlatL2(dimension)

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    def add(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        if embeddings.shape[0] == 0:
            return
        embeddings = embeddings.astype(np.float32, copy=False)
        ids_list = ids.tolist()
        start_row = len(self._row_to_id)
        self.index.add(embeddings)
        for i, sid in enumerate(ids_list):
            row = start_row + i
            self._row_to_id.append(int(sid))
            self._id_to_row[int(sid)] = row

    # Backwards-compat shim — auto-assigns sequential ids.
    def add_embeddings(self, embeddings: np.ndarray) -> None:
        n = embeddings.shape[0]
        base = len(self._row_to_id)
        ids = np.arange(base, base + n, dtype=np.int64)
        self.add(embeddings, ids)

    def remove(self, ids: np.ndarray) -> int:
        """Logically delete vectors by stable id.

        FAISS FlatIndex doesn't support true deletion, so we zero-out the
        vector and mark the row deleted. Deleted rows are filtered from
        search results. Call compact() periodically if the delete ratio is
        high (not needed for typical incremental-reindex patterns).
        """
        removed = 0
        zeros = np.zeros((1, self.dimension), dtype=np.float32)
        for sid in ids.tolist():
            row = self._id_to_row.pop(int(sid), None)
            if row is None:
                continue
            self._deleted.add(row)
            # Overwrite with zeros so it never surfaces as top-k.
            self.index.reconstruct(row, zeros[0])
            removed += 1
        return removed

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32, copy=False)

        # Over-fetch to compensate for deleted rows we'll filter.
        fetch_k = min(top_k + len(self._deleted) + 1, max(self.index.ntotal, 1))
        raw_scores, raw_rows = self.index.search(query_embedding, fetch_k)

        out_scores: list[float] = []
        out_ids: list[int] = []
        for score, row in zip(raw_scores[0], raw_rows[0], strict=False):
            if int(row) == -1 or int(row) in self._deleted:
                continue
            stable_id = self._row_to_id[int(row)] if int(row) < len(self._row_to_id) else -1
            out_scores.append(float(score))
            out_ids.append(stable_id)
            if len(out_ids) == top_k:
                break

        scores = np.array([out_scores], dtype=np.float32)
        returned_ids = np.array([out_ids], dtype=np.int64)
        return scores, returned_ids

    @property
    def size(self) -> int:
        return self.index.ntotal - len(self._deleted)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)
        # Also persist the id mapping alongside the index.
        np.savez_compressed(
            path + ".idmap.npz",
            row_to_id=np.array(self._row_to_id, dtype=np.int64),
            deleted=np.array(sorted(self._deleted), dtype=np.int64),
        )

    def load(self, path: str) -> None:
        self.index = faiss.read_index(path)
        idmap_path = path + ".idmap.npz"
        import os
        if os.path.exists(idmap_path):
            data = np.load(idmap_path)
            self._row_to_id = data["row_to_id"].tolist()
            self._id_to_row = {sid: row for row, sid in enumerate(self._row_to_id)}
            self._deleted = set(data["deleted"].tolist())
        else:
            # Legacy index with no id map: sequential id assignment.
            n = self.index.ntotal
            self._row_to_id = list(range(n))
            self._id_to_row = {i: i for i in range(n)}
            self._deleted = set()
