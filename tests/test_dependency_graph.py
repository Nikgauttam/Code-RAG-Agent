"""Tests for the dependency graph (file + symbol)."""

from __future__ import annotations

from pathlib import Path

from core.graph.dependency_graph import DependencyGraph
from core.parser.ast_parser import CodeParser


def test_file_graph_has_b_to_a_import_edge(tiny_repo: Path) -> None:
    table = CodeParser(str(tiny_repo)).parse_repository()
    dg = DependencyGraph(table, repo_root=str(tiny_repo)).build()

    a_path = str(tiny_repo / "a.py")
    b_path = str(tiny_repo / "b.py")

    assert a_path in dg.file_graph[b_path]
    assert dg.file_graph[a_path] == []


def test_relative_imports_resolve_within_package(package_repo: Path) -> None:
    table = CodeParser(str(package_repo)).parse_repository()
    dg = DependencyGraph(table, repo_root=str(package_repo)).build()

    mod_path = str(package_repo / "pkg" / "sub" / "mod.py")
    utils_path = str(package_repo / "pkg" / "utils.py")
    other_utils = str(package_repo / "other" / "utils.py")

    edges = dg.file_graph[mod_path]
    # `from ..utils import helper` must point to pkg/utils.py specifically.
    assert utils_path in edges
    # And it must NOT cross-wire into the unrelated other/utils.py.
    assert other_utils not in edges


def test_symbol_graph_has_caller_to_callee_edges(tiny_repo: Path) -> None:
    table = CodeParser(str(tiny_repo)).parse_repository()
    dg = DependencyGraph(table, repo_root=str(tiny_repo)).build()

    # compute -> add
    assert "add" in dg.symbol_graph.neighbors("compute")
    # double -> Calculator (and `multiply` may also appear)
    double_callees = dg.symbol_graph.neighbors("double")
    assert "Calculator" in double_callees


def test_file_graph_dfs_bounded(tiny_repo: Path) -> None:
    table = CodeParser(str(tiny_repo)).parse_repository()
    dg = DependencyGraph(table, repo_root=str(tiny_repo)).build()

    b_path = str(tiny_repo / "b.py")
    reachable = dg.file_graph.dfs(b_path, max_depth=1)
    assert b_path in reachable
    assert str(tiny_repo / "a.py") in reachable


def test_legacy_graph_alias_still_works(tiny_repo: Path) -> None:
    table = CodeParser(str(tiny_repo)).parse_repository()
    dg = DependencyGraph(table, repo_root=str(tiny_repo))
    edges = dg.build_graph()  # legacy single-call API
    assert isinstance(edges, dict)
    assert str(tiny_repo / "b.py") in edges
