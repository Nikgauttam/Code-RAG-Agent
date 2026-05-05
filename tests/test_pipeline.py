"""Integration tests for the codebase pipeline.

ML models are loaded ONCE per session (via shared_embedder / shared_ranker
in conftest.py) and injected into every pipeline instance. This avoids the
double-load segfault on macOS/Apple Silicon.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("faiss")


def _make_repo(base: Path) -> Path:
    (base / "a.py").write_text(
        textwrap.dedent("""
            def add(x, y):
                return x + y

            class Calculator:
                def multiply(self, x, y):
                    return x * y

                def square(self, x):
                    return self.multiply(x, x)
        """).lstrip()
    )
    (base / "b.py").write_text(
        textwrap.dedent("""
            from a import add, Calculator

            def compute(values):
                total = 0
                for v in values:
                    total = add(total, v)
                return total

            def double(x):
                return Calculator().multiply(x, 2)
        """).lstrip()
    )
    return base


@pytest.fixture(scope="module")
def indexed_pipeline(tmp_path_factory, shared_embedder, shared_ranker):  # type: ignore[no-untyped-def]
    """One indexed pipeline, shared across all tests in this module."""
    repo = _make_repo(tmp_path_factory.mktemp("repo"))
    storage = tmp_path_factory.mktemp("storage")

    from core.config import Config, StorageConfig
    from core.pipeline.codebase_pipeline import CodebasePipeline

    cfg = Config(storage=StorageConfig(directory=str(storage)))
    pipeline = CodebasePipeline(
        str(repo),
        config=cfg,
        embedder=shared_embedder,
        cross_ranker=shared_ranker,
    )
    pipeline.llm.generate = lambda prompt: "STUB"  # type: ignore[method-assign]
    pipeline.index_codebase()
    return pipeline


def test_index_codebase_builds_vector_store(indexed_pipeline) -> None:  # type: ignore[no-untyped-def]
    assert indexed_pipeline.vector_store is not None
    assert indexed_pipeline.vector_store.size > 0
    assert len(indexed_pipeline.chunks) == 6  # add, Calculator, multiply, square, compute, double


def test_retrieve_returns_relevant_chunk(indexed_pipeline) -> None:  # type: ignore[no-untyped-def]
    results = indexed_pipeline.retrieve("how is multiplication done?", top_k=3)
    assert results
    names = [c.name for c, _ in results]
    assert "multiply" in names or "Calculator" in names


def test_find_usages_uses_symbol_table_not_substring(indexed_pipeline) -> None:  # type: ignore[no-untyped-def]
    result = indexed_pipeline.find_usages("add")
    assert result["symbol"] == "add"

    defs = [d["file"] for d in result["definitions"]]  # type: ignore[union-attr]
    assert any(str(p).endswith("a.py") for p in defs)

    usages = result["usages"]
    assert isinstance(usages, list)
    assert any(u["caller"] == "compute" for u in usages)  # type: ignore[index]


def test_generate_answer_returns_answer_and_sources(indexed_pipeline) -> None:  # type: ignore[no-untyped-def]
    out = indexed_pipeline.generate_answer("how does compute aggregate values?")
    assert "answer" in out
    assert isinstance(out["sources"], list)
    assert out["sources"]


def test_incremental_reindex_picks_up_new_function(
    tmp_path_factory: pytest.TempPathFactory,
    shared_embedder,  # type: ignore[no-untyped-def]
    shared_ranker,
) -> None:
    repo = tmp_path_factory.mktemp("repo_incr")
    storage = tmp_path_factory.mktemp("storage_incr")
    (repo / "a.py").write_text("def add(x, y):\n    return x + y\n")

    from core.config import Config, StorageConfig
    from core.pipeline.codebase_pipeline import CodebasePipeline

    cfg = Config(storage=StorageConfig(directory=str(storage)))

    p1 = CodebasePipeline(
        str(repo), config=cfg, embedder=shared_embedder, cross_ranker=shared_ranker
    )
    p1.llm.generate = lambda prompt: "STUB"  # type: ignore[method-assign]
    p1.index_codebase()
    ids_before = {c.stable_id for c in p1.chunks}

    with open(repo / "a.py", "a") as fp:
        fp.write("\ndef brand_new():\n    return 99\n")

    p2 = CodebasePipeline(
        str(repo), config=cfg, embedder=shared_embedder, cross_ranker=shared_ranker
    )
    p2.llm.generate = lambda prompt: "STUB"  # type: ignore[method-assign]
    p2.index_codebase()

    new_ids = {c.stable_id for c in p2.chunks} - ids_before
    assert any("brand_new" in sid for sid in new_ids)
