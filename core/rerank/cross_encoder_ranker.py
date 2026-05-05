"""Cross-encoder reranker.

The bi-encoder retrieval stage is fast but coarse: it scores query and
chunk independently. A cross-encoder reads (query, chunk) jointly and is
much better at fine-grained ranking, at the cost of one model forward
per candidate. The standard trick is to over-fetch with the bi-encoder
(~20 candidates) and let the cross-encoder pick the final top-k.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from core.config import DEFAULT_CONFIG, RerankerConfig
from core.retrieval.chunker import CodeChunk

logger = logging.getLogger(__name__)


@dataclass
class RerankedChunk:
    chunk: CodeChunk
    score: float


class CrossEncoderRanker:
    def __init__(
        self,
        model_name: str | None = None,
        config: RerankerConfig | None = None,
    ):
        self.config = config or DEFAULT_CONFIG.reranker
        if model_name:
            self.config = RerankerConfig(
                model_name=model_name, batch_size=self.config.batch_size
            )
        logger.info("loading reranker %s", self.config.model_name)
        self.model = CrossEncoder(self.config.model_name)

    def rerank(
        self, query: str, chunks: Sequence[CodeChunk]
    ) -> list[CodeChunk]:
        scored = self.rerank_with_scores(query, chunks)
        return [r.chunk for r in scored]

    def rerank_with_scores(
        self, query: str, chunks: Sequence[CodeChunk]
    ) -> list[RerankedChunk]:
        if not chunks:
            return []
        pairs = [(query, c.content) for c in chunks]
        scores = self.model.predict(
            pairs, batch_size=self.config.batch_size, show_progress_bar=False
        )
        ranked = [RerankedChunk(c, float(s)) for c, s in zip(chunks, scores, strict=True)]
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked
