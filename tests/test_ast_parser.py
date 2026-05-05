"""Tests for the AST parser (visitor-based, FQN-aware)."""

from __future__ import annotations

from pathlib import Path

from core.parser.ast_parser import CodeParser


def test_parses_functions_classes_methods_and_imports(tiny_repo: Path) -> None:
    parser = CodeParser(str(tiny_repo))
    table = parser.parse_repository()

    assert len(table) == 2

    a = table[str(tiny_repo / "a.py")]
    b = table[str(tiny_repo / "b.py")]

    assert {f.name for f in a.functions} == {"add"}
    assert {c.name for c in a.classes} == {"Calculator"}
    # multiply + square live as methods now, not as top-level functions.
    assert {m.name for m in a.methods} == {"multiply", "square"}

    # Methods carry their parent class qualname.
    sq = next(m for m in a.methods if m.name == "square")
    assert sq.qualname == "Calculator.square"
    assert sq.parent == "Calculator"

    # b.py: top-level functions only, no classes.
    assert {f.name for f in b.functions} == {"compute", "double"}
    assert b.classes == []

    # ImportFrom now produces structured ImportEntry objects.
    import_modules = [(i.module, i.name) for i in b.imports]
    assert ("a", "add") in import_modules
    assert ("a", "Calculator") in import_modules


def test_records_end_lineno_for_chunking(tiny_repo: Path) -> None:
    table = CodeParser(str(tiny_repo)).parse_repository()
    a = table[str(tiny_repo / "a.py")]
    add_fn = next(f for f in a.functions if f.name == "add")
    assert add_fn.end_lineno >= add_fn.lineno


def test_call_sites_track_enclosing_scope(tiny_repo: Path) -> None:
    table = CodeParser(str(tiny_repo)).parse_repository()
    b = table[str(tiny_repo / "b.py")]

    # `compute` calls `add`; the call must be tagged with caller_qualname='compute'.
    add_calls = [c for c in b.calls if c.callee == "add"]
    assert add_calls, "expected at least one call to add()"
    assert any(c.caller_qualname == "compute" for c in add_calls)

    # `double` calls Calculator() then .multiply(); both should be present and
    # tagged with caller='double'.
    double_calls = [c for c in b.calls if c.caller_qualname == "double"]
    callees = {c.callee for c in double_calls}
    assert "Calculator" in callees
    assert "multiply" in callees or any(
        c.callee and c.callee.endswith("multiply") for c in double_calls
    )


def test_method_call_inside_class_tracked_with_method_qualname(tiny_repo: Path) -> None:
    table = CodeParser(str(tiny_repo)).parse_repository()
    a = table[str(tiny_repo / "a.py")]

    # `Calculator.square` calls `self.multiply(...)` — caller should be
    # `Calculator.square`, callee should mention `multiply`.
    square_calls = [c for c in a.calls if c.caller_qualname == "Calculator.square"]
    assert square_calls
    assert any(c.callee and c.callee.endswith("multiply") for c in square_calls)


def test_handles_syntax_errors_gracefully(tmp_path: Path) -> None:
    bad = tmp_path / "broken.py"
    bad.write_text("def foo(:\n    pass\n")
    table = CodeParser(str(tmp_path)).parse_repository()
    assert table[str(bad)].parse_error is not None
    assert table[str(bad)].functions == []


def test_skips_non_python_and_dunder_dirs(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("def x():\n    pass\n")
    (tmp_path / "x.txt").write_text("hello")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "x.py").write_text("def cached():\n    pass\n")

    table = CodeParser(str(tmp_path)).parse_repository()
    paths = set(table)
    assert str(tmp_path / "x.py") in paths
    assert not any("__pycache__" in p for p in paths)
    assert not any(p.endswith(".txt") for p in paths)
