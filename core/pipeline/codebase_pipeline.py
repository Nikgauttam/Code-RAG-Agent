"""End-to-end indexing + retrieval pipeline.

Cold path:
  1. Walk the repo, AST-parse every .py file.
  2. Build the file dependency graph (resolved imports) and the symbol
     call graph (caller -> callees).
  3. Chunk per function/class/method; hash each chunk's content.
  4. Embed chunks (batched, L2-normalized).
  5. Store vectors in FAISS (IDMap2 over IndexFlatIP); persist a JSON
     manifest + .npz embeddings + the FAISS index.

Warm path: load the manifest. Hash every file on disk; for any file
whose hash changed, drop its chunks from FAISS, re-chunk + re-embed,
and add the new chunks back. Untouched files are reused as-is.

The retrieve / explain / trace / find_usages methods sit on top.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from core.config import DEFAULT_CONFIG, Config
from core.graph.dependency_graph import DependencyGraph, FileGraph, SymbolGraph
from core.llm.ollama_client import LLMError, OllamaClient
from core.parser.ast_parser import CodeParser, FileSymbols
from core.rerank.cross_encoder_ranker import CrossEncoderRanker, RerankedChunk
from core.retrieval.chunker import CodeChunk, CodeChunker
from core.retrieval.embedder import CodeEmbedder
from core.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fp:
            for buf in iter(lambda: fp.read(65536), b""):
                h.update(buf)
    except OSError:
        return ""
    return h.hexdigest()


def _stable_int_id(stable_id: str) -> int:
    """Map a chunk stable_id string to a 64-bit signed int for FAISS."""
    digest = hashlib.blake2b(stable_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


@dataclass
class RetrievedChunk:
    chunk: CodeChunk
    score: float


@dataclass
class IndexState:
    """In-memory representation of everything persisted to disk."""

    schema_version: int = 0
    repo_path: str = ""
    embed_model: str = ""
    dimension: int = 0
    chunks: list[CodeChunk] = field(default_factory=list)
    file_hashes: dict[str, str] = field(default_factory=dict)  # rel_path -> sha256
    embeddings: np.ndarray | None = None
    file_graph_edges: dict[str, list[str]] = field(default_factory=dict)
    symbol_graph_edges: dict[str, list[str]] = field(default_factory=dict)


class CodebasePipeline:
    def __init__(
        self,
        repo_path: str,
        model: str | None = None,
        config: Config | None = None,
        *,
        embedder: CodeEmbedder | None = None,
        cross_ranker: CrossEncoderRanker | None = None,
    ):
        self.repo_path = os.path.abspath(repo_path)
        self.config = config or DEFAULT_CONFIG
        if model:
            from core.config import LLMConfig

            self.config = Config(
                retrieval=self.config.retrieval,
                chunk=self.config.chunk,
                embedding=self.config.embedding,
                reranker=self.config.reranker,
                llm=LLMConfig(
                    model=model,
                    base_url=self.config.llm.base_url,
                    timeout_s=self.config.llm.timeout_s,
                    max_retries=self.config.llm.max_retries,
                    temperature=self.config.llm.temperature,
                ),
                storage=self.config.storage,
            )

        self.parser = CodeParser(self.repo_path)
        self.embedder = embedder or CodeEmbedder(config=self.config.embedding)
        self.cross_ranker = cross_ranker or CrossEncoderRanker(config=self.config.reranker)
        self.llm = OllamaClient(config=self.config.llm)

        self.symbol_table: dict[str, FileSymbols] = {}
        self.file_graph: FileGraph = FileGraph()
        self.symbol_graph: SymbolGraph = SymbolGraph()
        self.chunks: list[CodeChunk] = []
        self.vector_store: VectorStore | None = None
        self._chunk_by_id: dict[int, CodeChunk] = {}

    # ------------------------------------------------------------------ #
    # Convenience views
    # ------------------------------------------------------------------ #

    @property
    def graph(self) -> dict[str, list[str]]:
        """File-level adjacency dict (backwards-compat alias)."""
        return self.file_graph.edges

    # ------------------------------------------------------------------ #
    # Storage paths
    # ------------------------------------------------------------------ #

    def _storage_paths(self) -> tuple[str, str, str]:
        s = self.config.storage
        os.makedirs(s.directory, exist_ok=True)
        return (
            os.path.join(s.directory, s.metadata_filename),
            os.path.join(s.directory, s.embeddings_filename),
            os.path.join(s.directory, s.index_filename),
        )

    # ------------------------------------------------------------------ #
    # Build / load
    # ------------------------------------------------------------------ #

    def index_codebase(self) -> None:
        meta_path, emb_path, idx_path = self._storage_paths()

        cached = self._load_state(meta_path, emb_path, idx_path)
        if cached is not None:
            self._apply_state(cached)
            self._reindex_changed_files(meta_path, emb_path, idx_path)
            return

        logger.info("indexing %s from scratch", self.repo_path)
        print("Indexing codebase...")
        self.symbol_table = self.parser.parse_repository()

        dg = DependencyGraph(self.symbol_table, repo_root=self.repo_path).build()
        self.file_graph = dg.file_graph
        self.symbol_graph = dg.symbol_graph

        chunker = CodeChunker(self.symbol_table, repo_root=self.repo_path, config=self.config.chunk)
        self.chunks = chunker.chunk_repository()

        embeddings = self.embedder.embed_chunks(self.chunks)
        self._build_vector_store(self.chunks, embeddings)

        file_hashes = {self._rel(p): _file_hash(p) for p in self.symbol_table}

        state = IndexState(
            schema_version=self.config.storage.schema_version,
            repo_path=self.repo_path,
            embed_model=self.config.embedding.model_name,
            dimension=embeddings.shape[1] if embeddings.size else self.embedder.dimension,
            chunks=self.chunks,
            file_hashes=file_hashes,
            embeddings=embeddings,
            file_graph_edges=self.file_graph.edges,
            symbol_graph_edges=self.symbol_graph.edges,
        )
        self._save_state(state, meta_path, emb_path, idx_path)
        print(f"Index built: {len(self.chunks)} chunks across {len(self.symbol_table)} files.")

    def _apply_state(self, state: IndexState) -> None:
        # Re-derive symbol table from disk (cheap; chunk content is the only
        # thing we *had* to persist; AST re-parse on warm boot is sub-second
        # for typical repos and gives us a fresh, correct symbol view).
        self.symbol_table = self.parser.parse_repository()
        dg = DependencyGraph(self.symbol_table, repo_root=self.repo_path).build()
        self.file_graph = dg.file_graph
        self.symbol_graph = dg.symbol_graph
        self.chunks = state.chunks
        self.vector_store = VectorStore(state.dimension)
        meta_path, emb_path, idx_path = self._storage_paths()
        self.vector_store.load(idx_path)
        self._chunk_by_id = {_stable_int_id(c.stable_id): c for c in state.chunks}

    def _reindex_changed_files(self, meta_path: str, emb_path: str, idx_path: str) -> None:
        # Detect which files changed since the last index by comparing hashes.
        prev_state = self._load_state(meta_path, emb_path, idx_path)
        if prev_state is None:
            return

        current_files = {self._rel(p): p for p in self.symbol_table}
        prev_hashes = prev_state.file_hashes

        added: list[str] = []
        modified: list[str] = []
        for rel, abs_path in current_files.items():
            new_hash = _file_hash(abs_path)
            old_hash = prev_hashes.get(rel)
            if old_hash is None:
                added.append(abs_path)
            elif old_hash != new_hash:
                modified.append(abs_path)

        removed_rels = [rel for rel in prev_hashes if rel not in current_files]

        if not (added or modified or removed_rels):
            print("Index loaded from disk (no changes).")
            return

        print(f"Incremental reindex: +{len(added)} ~{len(modified)} -{len(removed_rels)} files.")

        # Drop chunks from removed + modified files.
        drop_files = set(modified) | {os.path.join(self.repo_path, rel) for rel in removed_rels}
        keep_chunks = [c for c in self.chunks if c.file not in drop_files]
        drop_ids = np.array(
            [_stable_int_id(c.stable_id) for c in self.chunks if c.file in drop_files],
            dtype=np.int64,
        )
        if self.vector_store is not None and drop_ids.size:
            self.vector_store.remove(drop_ids)

        # Build new chunks from added + modified files.
        sub_table = {p: syms for p, syms in self.symbol_table.items() if p in (added + modified)}
        chunker = CodeChunker(sub_table, repo_root=self.repo_path, config=self.config.chunk)
        new_chunks = chunker.chunk_repository()

        if new_chunks and self.vector_store is not None:
            new_embeddings = self.embedder.embed_chunks(new_chunks)
            new_ids = np.array([_stable_int_id(c.stable_id) for c in new_chunks], dtype=np.int64)
            self.vector_store.add(new_embeddings, new_ids)

        self.chunks = keep_chunks + new_chunks
        self._chunk_by_id = {_stable_int_id(c.stable_id): c for c in self.chunks}

        # Re-persist.
        new_state = IndexState(
            schema_version=self.config.storage.schema_version,
            repo_path=self.repo_path,
            embed_model=self.config.embedding.model_name,
            dimension=prev_state.dimension,
            chunks=self.chunks,
            file_hashes={self._rel(p): _file_hash(p) for p in self.symbol_table},
            embeddings=None,  # not strictly needed once the index is on disk
            file_graph_edges=self.file_graph.edges,
            symbol_graph_edges=self.symbol_graph.edges,
        )
        self._save_state(new_state, meta_path, emb_path, idx_path, write_embeddings=False)

    def _build_vector_store(self, chunks: list[CodeChunk], embeddings: np.ndarray) -> None:
        dim = embeddings.shape[1] if embeddings.size else self.embedder.dimension
        self.vector_store = VectorStore(dim)
        if not chunks:
            self._chunk_by_id = {}
            return
        ids = np.array([_stable_int_id(c.stable_id) for c in chunks], dtype=np.int64)
        self.vector_store.add(embeddings, ids)
        self._chunk_by_id = {int(i): c for i, c in zip(ids.tolist(), chunks, strict=True)}

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _save_state(
        self,
        state: IndexState,
        meta_path: str,
        emb_path: str,
        idx_path: str,
        *,
        write_embeddings: bool = True,
    ) -> None:
        manifest = {
            "schema_version": state.schema_version,
            "repo_path": state.repo_path,
            "embed_model": state.embed_model,
            "dimension": state.dimension,
            "file_hashes": state.file_hashes,
            "file_graph": state.file_graph_edges,
            "symbol_graph": state.symbol_graph_edges,
            "chunks": [c.to_dict() for c in state.chunks],
        }
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2)
        if write_embeddings and state.embeddings is not None:
            np.savez_compressed(emb_path, embeddings=state.embeddings)
        if self.vector_store is not None:
            self.vector_store.save(idx_path)

    def _load_state(self, meta_path: str, emb_path: str, idx_path: str) -> IndexState | None:
        if not (os.path.exists(meta_path) and os.path.exists(idx_path)):
            return None
        try:
            with open(meta_path, encoding="utf-8") as fp:
                manifest = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not read manifest: %s", exc)
            return None

        if manifest.get("schema_version") != self.config.storage.schema_version:
            logger.info("schema version changed; rebuilding from scratch")
            return None
        if manifest.get("embed_model") != self.config.embedding.model_name:
            logger.info("embed model changed; rebuilding from scratch")
            return None
        if manifest.get("repo_path") != self.repo_path:
            logger.info("repo path changed; rebuilding from scratch")
            return None

        chunks = [CodeChunk(**c) for c in manifest["chunks"]]
        embeddings: np.ndarray | None = None
        if os.path.exists(emb_path):
            try:
                embeddings = np.load(emb_path)["embeddings"]
            except (OSError, KeyError):
                embeddings = None

        return IndexState(
            schema_version=manifest["schema_version"],
            repo_path=manifest["repo_path"],
            embed_model=manifest["embed_model"],
            dimension=int(manifest["dimension"]),
            chunks=chunks,
            file_hashes=dict(manifest.get("file_hashes", {})),
            embeddings=embeddings,
            file_graph_edges=dict(manifest.get("file_graph", {})),
            symbol_graph_edges=dict(manifest.get("symbol_graph", {})),
        )

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[CodeChunk, float]]:
        if self.vector_store is None or not self.chunks:
            return []
        k = top_k or self.config.retrieval.top_k
        q_emb = self.embedder.embed_query(query)
        scores, ids = self.vector_store.search(q_emb, top_k=k)
        out: list[tuple[CodeChunk, float]] = []
        for raw_id, score in zip(ids[0], scores[0], strict=True):
            int_id = int(raw_id)
            if int_id == -1:
                continue
            chunk = self._chunk_by_id.get(int_id)
            if chunk is None:
                continue
            # IndexFlatIP returns inner product; for normalized vectors that's
            # cosine similarity in [-1, 1]; rescale to [0, 1] for hybrid math.
            sim = (float(score) + 1.0) / 2.0
            out.append((chunk, sim))
        return out

    def _hybrid_scored(self, query: str, top_k: int) -> list[tuple[CodeChunk, float]]:
        cfg = self.config.retrieval
        retrieved = self.retrieve(query, top_k=top_k)
        if not retrieved:
            return []

        seed_files = {c.file for c, _ in retrieved}

        # Bounded DFS over file graph from each seed.
        file_depth: dict[str, int] = {}
        stack: list[tuple[str, int]] = [(f, 0) for f in seed_files]
        while stack:
            current, depth = stack.pop()
            if current in file_depth and file_depth[current] <= depth:
                continue
            if depth > cfg.graph_max_depth:
                continue
            file_depth[current] = depth
            for nb in self.file_graph.neighbors(current):
                stack.append((nb, depth + 1))

        scored: dict[str, dict[str, object]] = {}
        for chunk, sem in retrieved:
            scored[chunk.stable_id] = {
                "chunk": chunk,
                "score": cfg.semantic_weight * sem,
            }
        for chunk in self.chunks:
            depth = file_depth.get(chunk.file)
            if depth is None:
                continue
            graph_score = cfg.graph_weight * (1.0 / (1 + depth))
            entry = scored.get(chunk.stable_id)
            if entry is None:
                scored[chunk.stable_id] = {"chunk": chunk, "score": graph_score}
            else:
                entry["score"] = float(entry["score"]) + graph_score

        ordered = sorted(scored.values(), key=lambda x: float(x["score"]), reverse=True)
        return [(e["chunk"], float(e["score"])) for e in ordered]  # type: ignore[misc]

    # ------------------------------------------------------------------ #
    # Public answer modes
    # ------------------------------------------------------------------ #

    def generate_answer(self, query: str) -> dict[str, object]:
        cfg = self.config.retrieval
        hybrid = self._hybrid_scored(query, top_k=cfg.top_k)
        if not hybrid:
            return {
                "answer": "No relevant code found in the indexed repository.",
                "sources": [],
            }

        candidates = [c for c, _ in hybrid[: cfg.rerank_pool]]
        reranked = self.cross_ranker.rerank(query, candidates)[: cfg.final_k]

        prompt = self._build_qa_prompt(query, reranked)
        try:
            answer = self.llm.generate(prompt)
        except LLMError as exc:
            answer = f"[LLM unavailable: {exc}]"

        return {
            "answer": answer,
            "sources": [self._rel(c.file) for c in reranked],
        }

    def stream_answer(self, query: str):
        cfg = self.config.retrieval
        hybrid = self._hybrid_scored(query, top_k=cfg.top_k)
        if not hybrid:
            yield "No relevant code found in the indexed repository."
            return
        candidates = [c for c, _ in hybrid[: cfg.rerank_pool]]
        reranked = self.cross_ranker.rerank(query, candidates)[: cfg.final_k]
        prompt = self._build_qa_prompt(query, reranked)
        yield from self.llm.stream(prompt)

    def _build_qa_prompt(self, query: str, chunks: Iterable[CodeChunk]) -> str:
        parts: list[str] = []
        for c in chunks:
            header = (
                f"# FILE: {self._rel(c.file)}  ({c.type} {c.qualname})  L{c.lineno}-{c.end_lineno}"
            )
            parts.append(f"{header}\n{c.content}")
        context = "\n\n".join(parts)
        return f"""You are a senior software engineer answering a question about a codebase.

You MUST ground your answer ONLY in the code snippets below.
If the snippets do not contain enough information, say so explicitly.
Do NOT invent files, classes, or functions.

Question:
{query}

Code snippets:
{context}

Answer in clear, technical prose. Cite file paths and line ranges when relevant.
"""

    def explain_symbol(self, symbol: str) -> str:
        context = self._symbol_context(symbol)
        if not context:
            return f"Symbol '{symbol}' not found in the indexed codebase."
        prompt = f"""You are a senior software engineer analyzing a codebase.

Explain the symbol below using ONLY the provided code context.

Symbol: {symbol}

Code Context:
{context}

Respond STRICTLY in this format:

SUMMARY:
(1-2 sentence overview)

PURPOSE:
(Why this exists)

INPUTS:
(List parameters and their meaning)

OUTPUTS:
(What it returns or produces)

INTERNAL LOGIC:
(How it works step-by-step)

DEPENDENCIES:
(Other classes/functions it relies on)
"""
        try:
            return self.llm.generate(prompt)
        except LLMError as exc:
            return f"[LLM unavailable: {exc}]"

    def trace_symbol(self, symbol: str) -> str:
        # Walk the *symbol* graph (caller -> callees), bounded to avoid runaway.
        visited: list[str] = []
        seen: set[str] = set()
        stack: list[str] = [symbol]
        while stack and len(visited) < 50:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            visited.append(cur)
            for nb in self.symbol_graph.neighbors(cur):
                stack.append(nb)
        # Find any matching symbol qualnames whose short name == symbol, too.
        for qn in self.symbol_graph.edges:
            if qn.split(".")[-1] == symbol and qn not in seen:
                stack.append(qn)
                seen.add(qn)
                visited.append(qn)

        trace_context = "\n".join(visited[:20]) or symbol

        prompt = f"""You are a static code analysis engine.

You MUST ONLY use the provided trace.
Do NOT invent files, classes, or architecture.

Symbol being traced:
{symbol}

Trace (caller first, then callees in DFS order):
{trace_context}

Respond in this format:

ENTRY POINT:
(symbol or file where this trace begins)

CALLS:
(functions/classes it calls)

CALLED BY:
(callers of this symbol — say "unknown" if not in the trace)

FLOW SUMMARY:
(step-by-step explanation grounded in the trace)

ARCHITECTURAL ROLE:
(role inferred only from the trace)
"""
        try:
            return self.llm.generate(prompt)
        except LLMError as exc:
            return f"[LLM unavailable: {exc}]"

    def find_usages(self, symbol: str) -> dict[str, object]:
        """Find usages of a symbol via the symbol table (not substring match).

        Definition: any FileSymbols that has a definition with this exact
        short name. Usages: any CallSite whose callee's last segment matches.
        """
        definitions: list[dict[str, object]] = []
        usages: list[dict[str, object]] = []

        for file_path, syms in self.symbol_table.items():
            for d in syms.all_definitions:
                if d.name == symbol:
                    definitions.append(
                        {
                            "file": self._rel(file_path),
                            "qualname": d.qualname,
                            "kind": d.kind,
                            "line": d.lineno,
                        }
                    )
            for call in syms.calls:
                if not call.callee:
                    continue
                last = call.callee.split(".")[-1]
                if last == symbol:
                    usages.append(
                        {
                            "file": self._rel(file_path),
                            "caller": call.caller_qualname or "<module>",
                            "callee": call.callee,
                            "line": call.lineno,
                        }
                    )

        return {
            "symbol": symbol,
            "definitions": definitions,
            "usages": usages,
            "definition_count": len(definitions),
            "usage_count": len(usages),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _symbol_context(self, symbol: str) -> str:
        relevant = [c.content for c in self.chunks if c.name == symbol or c.qualname == symbol]
        if not relevant:
            # Fallback: substring, but bounded — keeps explain-by-keyword usable.
            relevant = [c.content for c in self.chunks if symbol in c.content][:5]
        return "\n\n".join(relevant[:10])

    def _rel(self, file_path: str) -> str:
        try:
            return os.path.relpath(file_path, self.repo_path)
        except ValueError:
            return file_path

    def _hybrid_with_rerank_scores(self, query: str) -> list[RerankedChunk]:
        cfg = self.config.retrieval
        hybrid = self._hybrid_scored(query, top_k=cfg.top_k)
        if not hybrid:
            return []
        candidates = [c for c, _ in hybrid[: cfg.rerank_pool]]
        return self.cross_ranker.rerank_with_scores(query, candidates)[: cfg.final_k]


# Mapping[str, Mapping[str, object]] alias kept for callers expecting the old shape
LegacySymbolTable = Mapping[str, Mapping[str, object]]
