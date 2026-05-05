"""Tests for the code chunker."""

from __future__ import annotations

from pathlib import Path

from core.parser.ast_parser import CodeParser
from core.retrieval.chunker import CodeChunker


def test_chunk_per_function_class_and_method(tiny_repo: Path) -> None:
    table = CodeParser(str(tiny_repo)).parse_repository()
    chunks = CodeChunker(table, repo_root=str(tiny_repo)).chunk_repository()

    names = sorted(c.name for c in chunks)
    # Top-level fns + class + each method individually.
    assert names == ["Calculator", "add", "compute", "double", "multiply", "square"]


def test_chunk_respects_end_lineno(tiny_repo: Path) -> None:
    table = CodeParser(str(tiny_repo)).parse_repository()
    chunks = CodeChunker(table, repo_root=str(tiny_repo)).chunk_repository()

    add_chunk = next(c for c in chunks if c.name == "add")
    assert "def add" in add_chunk.content
    assert "return x + y" in add_chunk.content
    assert "class Calculator" not in add_chunk.content


def test_chunk_has_stable_id_and_content_hash(tiny_repo: Path) -> None:
    table = CodeParser(str(tiny_repo)).parse_repository()
    chunks = CodeChunker(table, repo_root=str(tiny_repo)).chunk_repository()

    ids = [c.stable_id for c in chunks]
    assert len(ids) == len(set(ids)), "stable ids must be unique"

    add_chunk = next(c for c in chunks if c.name == "add")
    assert add_chunk.content_hash and len(add_chunk.content_hash) == 64
    assert "::add@" in add_chunk.stable_id


def test_method_chunk_includes_class_context_header(tiny_repo: Path) -> None:
    table = CodeParser(str(tiny_repo)).parse_repository()
    chunks = CodeChunker(table, repo_root=str(tiny_repo)).chunk_repository()
    sq = next(c for c in chunks if c.name == "square")
    assert "Calculator" in sq.content


def test_dict_style_access_for_back_compat(tiny_repo: Path) -> None:
    table = CodeParser(str(tiny_repo)).parse_repository()
    chunks = CodeChunker(table, repo_root=str(tiny_repo)).chunk_repository()
    add_chunk = next(c for c in chunks if c.name == "add")
    # Old code expected dict-style access; CodeChunk supports it.
    assert add_chunk["name"] == "add"
    assert add_chunk["file"].endswith("a.py")
