"""Run retrieval evaluation against the eval set.

Compares four retrieval strategies:
  1. BM25 over chunk content (lexical baseline)
  2. Semantic only          (FAISS top-k)
  3. Hybrid                 (semantic + graph DFS expansion, weighted)
  4. Hybrid + cross-encoder rerank (the production path)

Reports Recall@5, Recall@10, MRR, and p50/p95 latency for each.

Usage:
    python -m evaluation.run_eval [--repo PATH] [--top-k 10]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.pipeline.codebase_pipeline import CodebasePipeline
from core.retrieval.chunker import CodeChunk
from evaluation.eval_set import EVAL_SET
from evaluation.flask_eval_set import FLASK_EVAL_SET
from evaluation.metrics import mean, percentile, recall_at_k, reciprocal_rank


def _to_relative(file_path: str, repo_path: str) -> str:
    repo_path = os.path.abspath(repo_path)
    file_path = os.path.abspath(file_path)
    if file_path.startswith(repo_path + os.sep):
        return file_path[len(repo_path) + 1 :]
    return file_path


def _hybrid_files(pipeline: CodebasePipeline, query: str, top_k: int) -> list[str]:
    scored = pipeline._hybrid_scored(query, top_k=top_k)
    return [c.file for c, _ in scored[:top_k]]


def _hybrid_reranked_files(pipeline: CodebasePipeline, query: str, top_k: int) -> list[str]:
    scored = pipeline._hybrid_scored(query, top_k=top_k)
    candidates = [c for c, _ in scored[: pipeline.config.retrieval.rerank_pool]]
    if not candidates:
        return []
    reranked = pipeline.cross_ranker.rerank(query, candidates)[:top_k]
    return [c.file for c in reranked]


def _semantic_files(pipeline: CodebasePipeline, query: str, top_k: int) -> list[str]:
    retrieved = pipeline.retrieve(query, top_k=top_k)
    return [c.file for c, _ in retrieved]


_BM25_CACHE: dict[int, object] = {}


def _bm25_files(pipeline: CodebasePipeline, query: str, top_k: int) -> list[str]:
    """Lexical BM25 baseline. Falls back to semantic if rank_bm25 missing."""
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except ImportError:
        return _semantic_files(pipeline, query, top_k)

    key = id(pipeline)
    bm25: BM25Okapi
    chunks: list[CodeChunk]
    if key not in _BM25_CACHE:
        chunks = pipeline.chunks
        tokenized = [c.content.split() for c in chunks]
        bm25 = BM25Okapi(tokenized)
        _BM25_CACHE[key] = (bm25, chunks)  # type: ignore[assignment]
    bm25, chunks = _BM25_CACHE[key]  # type: ignore[assignment]
    scores = bm25.get_scores(query.split())
    order = scores.argsort()[::-1][:top_k]
    return [chunks[i].file for i in order]


def _evaluate_strategy(
    name: str,
    fn: Callable[[CodebasePipeline, str, int], list[str]],
    pipeline: CodebasePipeline,
    eval_set: Sequence[tuple[str, list[str]]],
    top_k: int,
) -> dict[str, object]:
    recall5: list[float] = []
    recall10: list[float] = []
    rr: list[float] = []
    latencies_ms: list[float] = []

    for question, expected in eval_set:
        t0 = time.perf_counter()
        retrieved_abs = fn(pipeline, question, top_k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        retrieved = [_to_relative(p, pipeline.repo_path) for p in retrieved_abs]

        recall5.append(recall_at_k(retrieved, expected, 5))
        recall10.append(recall_at_k(retrieved, expected, 10))
        rr.append(reciprocal_rank(retrieved, expected))

    return {
        "strategy": name,
        "recall@5": mean(recall5),
        "recall@10": mean(recall10),
        "mrr": mean(rr),
        "p50_ms": percentile(latencies_ms, 0.5),
        "p95_ms": percentile(latencies_ms, 0.95),
        "n": len(eval_set),
    }


def _print_table(rows: list[dict[str, object]]) -> None:
    headers = ["strategy", "recall@5", "recall@10", "mrr", "p50_ms", "p95_ms"]
    widths = {h: max(len(h), max(len(_fmt(r[h])) for r in rows)) for h in headers}
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for r in rows:
        print(" | ".join(_fmt(r[h]).ljust(widths[h]) for h in headers))


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if v < 100 else f"{v:.1f}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--repo", default=REPO_ROOT, help="Repo to index and query.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval.")
    parser.add_argument(
        "--eval",
        choices=["self", "flask"],
        default="self",
        help="Eval set to use: 'self' (this repo) or 'flask' (pallets/flask).",
    )
    args = parser.parse_args()

    eval_set = FLASK_EVAL_SET if args.eval == "flask" else EVAL_SET

    print(f"Indexing {args.repo} ...")
    pipeline = CodebasePipeline(args.repo)
    pipeline.index_codebase()
    print(f"Indexed {len(pipeline.chunks)} chunks across {len(pipeline.symbol_table)} files.\n")

    strategies: list[tuple[str, Callable[[CodebasePipeline, str, int], list[str]]]] = [
        ("bm25", _bm25_files),
        ("semantic", _semantic_files),
        ("hybrid", _hybrid_files),
        ("hybrid+rerank", _hybrid_reranked_files),
    ]

    rows = [_evaluate_strategy(name, fn, pipeline, eval_set, args.top_k) for name, fn in strategies]
    _print_table(rows)
    print(f"\nN = {rows[0]['n']} questions  |  eval={args.eval}")


if __name__ == "__main__":
    main()
