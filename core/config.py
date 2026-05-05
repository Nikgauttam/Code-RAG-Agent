"""Central configuration for the code agent.

All tunable knobs live here so they can be overridden via environment
variables or programmatically without hunting through the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None and raw.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None and raw.strip() else default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw is not None and raw.strip() else default


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    rerank_pool: int = 20
    final_k: int = 8
    semantic_weight: float = 0.7
    graph_weight: float = 0.3
    graph_max_depth: int = 2


@dataclass(frozen=True)
class ChunkConfig:
    max_lines: int = 200
    include_class_context: bool = True


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize: bool = True


@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32


@dataclass(frozen=True)
class LLMConfig:
    model: str = "llama3"
    base_url: str = "http://localhost:11434"
    timeout_s: float = 60.0
    max_retries: int = 2
    temperature: float = 0.2


@dataclass(frozen=True)
class StorageConfig:
    directory: str = "storage"
    metadata_filename: str = "metadata.json"
    embeddings_filename: str = "embeddings.npz"
    index_filename: str = "faiss.index"
    schema_version: int = 2


@dataclass(frozen=True)
class Config:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            retrieval=RetrievalConfig(
                top_k=_env_int("CODE_AGENT_TOP_K", 5),
                rerank_pool=_env_int("CODE_AGENT_RERANK_POOL", 20),
                final_k=_env_int("CODE_AGENT_FINAL_K", 8),
                semantic_weight=_env_float("CODE_AGENT_SEMANTIC_WEIGHT", 0.7),
                graph_weight=_env_float("CODE_AGENT_GRAPH_WEIGHT", 0.3),
                graph_max_depth=_env_int("CODE_AGENT_GRAPH_MAX_DEPTH", 2),
            ),
            chunk=ChunkConfig(
                max_lines=_env_int("CODE_AGENT_CHUNK_MAX_LINES", 200),
            ),
            embedding=EmbeddingConfig(
                model_name=_env_str("CODE_AGENT_EMBED_MODEL", "all-MiniLM-L6-v2"),
                batch_size=_env_int("CODE_AGENT_EMBED_BATCH", 32),
            ),
            reranker=RerankerConfig(
                model_name=_env_str(
                    "CODE_AGENT_RERANK_MODEL",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                ),
            ),
            llm=LLMConfig(
                model=_env_str("CODE_AGENT_MODEL", "llama3"),
                base_url=_env_str("CODE_AGENT_LLM_URL", "http://localhost:11434"),
                timeout_s=_env_float("CODE_AGENT_LLM_TIMEOUT", 60.0),
                max_retries=_env_int("CODE_AGENT_LLM_RETRIES", 2),
                temperature=_env_float("CODE_AGENT_LLM_TEMPERATURE", 0.2),
            ),
            storage=StorageConfig(
                directory=_env_str("CODE_AGENT_STORAGE", "storage"),
            ),
        )


DEFAULT_CONFIG = Config()
