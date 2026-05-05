"""Sentence-Transformer based embedder with optional content-hash cache.

Cosine-similarity quality improves substantially when vectors are L2-
normalized at encode time, so ``normalize`` defaults to True. The
companion FAISS store can then use either ``IndexFlatIP`` (cosine) or
the original ``IndexFlatL2`` interchangeably; with normalized vectors,
the two rankings are equivalent.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import DEFAULT_CONFIG, EmbeddingConfig
from core.retrieval.chunker import CodeChunk

logger = logging.getLogger(__name__)


class CodeEmbedder:
    def __init__(
        self,
        model_name: str | None = None,
        config: EmbeddingConfig | None = None,
    ):
        self.config = config or DEFAULT_CONFIG.embedding
        if model_name:
            self.config = EmbeddingConfig(
                model_name=model_name,
                batch_size=self.config.batch_size,
                normalize=self.config.normalize,
            )
        logger.info("loading embedding model %s", self.config.model_name)
        self.model = SentenceTransformer(self.config.model_name)

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension() or 0)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    def embed_chunks(
        self,
        chunks: Sequence[CodeChunk],
        cache: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Embed chunks, optionally reusing a ``content_hash -> vector`` cache.

        Cached vectors are returned as-is; only new chunks are sent through
        the model. Order of the returned matrix matches ``chunks``.
        """
        if not chunks:
            return np.zeros((0, self.dimension), dtype="float32")

        if cache is None:
            return self.embed_texts([c.content for c in chunks])

        to_compute_idx: list[int] = []
        to_compute_text: list[str] = []
        for i, c in enumerate(chunks):
            if c.content_hash not in cache:
                to_compute_idx.append(i)
                to_compute_text.append(c.content)

        if to_compute_text:
            new_vecs = self.embed_texts(to_compute_text)
            for idx, vec in zip(to_compute_idx, new_vecs, strict=True):
                cache[chunks[idx].content_hash] = vec

        out = np.stack([cache[c.content_hash] for c in chunks])
        return out.astype("float32", copy=False)

    @staticmethod
    def cache_from_chunks(
        chunks: Iterable[CodeChunk], embeddings: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Build a hash->vector cache from chunks aligned with an embedding matrix."""
        chunk_list = list(chunks)
        if len(chunk_list) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunk_list)}) and embeddings ({embeddings.shape[0]}) misaligned"
            )
        return {c.content_hash: embeddings[i] for i, c in enumerate(chunk_list)}
