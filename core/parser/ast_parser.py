"""AST-based parser for a Python repository.

Produces a typed symbol table per file, including:
  - imports (raw module strings, plus parsed metadata for resolution)
  - top-level classes and functions
  - methods nested inside classes
  - call sites tagged with their *enclosing* function/class scope
    (so the dependency graph can build true caller -> callee edges,
    not just file -> all_calls_in_file)

The parser uses a NodeVisitor subclass and explicit scope tracking
(rather than ast.walk) so we don't lose the relationship between a
call expression and the function that contains it.
"""

from __future__ import annotations

import ast
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SKIP_DIRS: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "build",
        "dist",
        ".tox",
        ".eggs",
    }
)


@dataclass(frozen=True)
class ImportEntry:
    """A parsed import statement.

    `module`     — dotted module path (e.g. ``"foo.bar"``); empty string for
                    bare ``from . import x`` style relative imports.
    `name`       — the imported symbol (``None`` for ``import foo``).
    `alias`      — local binding name (defaults to ``name`` or last segment
                    of ``module``).
    `level`      — number of leading dots for relative imports (0 for absolute).
    """

    module: str
    name: str | None
    alias: str
    level: int = 0


@dataclass
class SymbolDef:
    """A class or function definition."""

    name: str
    qualname: str
    kind: str  # "function" | "method" | "class" | "async_function"
    lineno: int
    end_lineno: int
    parent: str | None = None  # qualname of the enclosing class, if any
    decorators: list[str] = field(default_factory=list)


@dataclass
class CallSite:
    """A call expression, tagged with the symbol that contains it.

    ``callee`` is the textual call target as it appears in source — either
    a bare name (``foo``), a dotted attribute chain (``self.bar.baz``), or
    ``None`` if the call target is too dynamic to capture statically (e.g.
    ``returned_factory()()``).
    """

    caller_qualname: str  # "" for module-level calls
    callee: str | None
    lineno: int


@dataclass
class FileSymbols:
    """Parsed contents of a single source file."""

    file_path: str
    classes: list[SymbolDef] = field(default_factory=list)
    functions: list[SymbolDef] = field(default_factory=list)
    methods: list[SymbolDef] = field(default_factory=list)
    imports: list[ImportEntry] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)
    parse_error: str | None = None

    @property
    def all_definitions(self) -> list[SymbolDef]:
        return [*self.classes, *self.functions, *self.methods]

    @property
    def symbol_names(self) -> list[str]:
        return [d.name for d in self.all_definitions]


class _SymbolVisitor(ast.NodeVisitor):
    """Walks an AST while maintaining an explicit scope stack.

    The scope stack is the key: every CallSite emitted carries the
    qualname of the function/method that lexically contains it. That's
    what makes the downstream graph a real call graph instead of a bag.
    """

    def __init__(self) -> None:
        self.classes: list[SymbolDef] = []
        self.functions: list[SymbolDef] = []
        self.methods: list[SymbolDef] = []
        self.imports: list[ImportEntry] = []
        self.calls: list[CallSite] = []

        # (qualname, kind) frames; kind in {"class","function","method"}
        self._scope: list[tuple[str, str]] = []

    @property
    def _current_qualname(self) -> str:
        return self._scope[-1][0] if self._scope else ""

    @property
    def _enclosing_function_qualname(self) -> str:
        """Nearest enclosing function/method qualname, skipping classes."""
        for qn, kind in reversed(self._scope):
            if kind in ("function", "method"):
                return qn
        return ""

    def _qualify(self, name: str) -> str:
        return f"{self._current_qualname}.{name}" if self._current_qualname else name

    @staticmethod
    def _decorator_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call):
            return _SymbolVisitor._decorator_name(node.func)
        return ast.unparse(node) if hasattr(ast, "unparse") else "<decorator>"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = self._qualify(node.name)
        parent = self._current_qualname or None
        self.classes.append(
            SymbolDef(
                name=node.name,
                qualname=qualname,
                kind="class",
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", node.lineno) or node.lineno,
                parent=parent,
                decorators=[self._decorator_name(d) for d in node.decorator_list],
            )
        )
        self._scope.append((qualname, "class"))
        try:
            self.generic_visit(node)
        finally:
            self._scope.pop()

    def _visit_function_like(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, is_async: bool
    ) -> None:
        qualname = self._qualify(node.name)
        in_class = bool(self._scope and self._scope[-1][1] == "class")
        kind = "method" if in_class else ("async_function" if is_async else "function")
        scope_kind = "method" if in_class else "function"
        bucket = self.methods if in_class else self.functions
        bucket.append(
            SymbolDef(
                name=node.name,
                qualname=qualname,
                kind=kind,
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", node.lineno) or node.lineno,
                parent=self._current_qualname or None,
                decorators=[self._decorator_name(d) for d in node.decorator_list],
            )
        )
        self._scope.append((qualname, scope_kind))
        try:
            self.generic_visit(node)
        finally:
            self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_like(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_like(node, is_async=True)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportEntry(
                    module=alias.name,
                    name=None,
                    alias=alias.asname or alias.name.split(".")[0],
                    level=0,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        level = node.level or 0
        for alias in node.names:
            self.imports.append(
                ImportEntry(
                    module=module,
                    name=alias.name,
                    alias=alias.asname or alias.name,
                    level=level,
                )
            )

    @staticmethod
    def _callee_text(func: ast.expr) -> str | None:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts: list[str] = [func.attr]
            cur: ast.expr = func.value
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                return ".".join(reversed(parts))
            return func.attr  # receiver too dynamic; keep just the method name
        return None

    def visit_Call(self, node: ast.Call) -> None:
        callee = self._callee_text(node.func)
        self.calls.append(
            CallSite(
                caller_qualname=self._enclosing_function_qualname,
                callee=callee,
                lineno=node.lineno,
            )
        )
        self.generic_visit(node)


class CodeParser:
    """Parses every ``.py`` file in a repo into a typed symbol table."""

    def __init__(self, root_dir: str, skip_dirs: Iterable[str] = SKIP_DIRS):
        self.root_dir = os.path.abspath(root_dir)
        self.skip_dirs = frozenset(skip_dirs)
        self.symbol_table: dict[str, FileSymbols] = {}

    def parse_repository(self) -> dict[str, FileSymbols]:
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in self.skip_dirs]
            for filename in files:
                if filename.endswith(".py"):
                    file_path = os.path.join(root, filename)
                    self.symbol_table[file_path] = self.parse_file(file_path)
        return self.symbol_table

    def parse_file(self, file_path: str) -> FileSymbols:
        try:
            with open(file_path, encoding="utf-8") as fp:
                source = fp.read()
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("could not read %s: %s", file_path, exc)
            return FileSymbols(file_path=file_path, parse_error=str(exc))

        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError as exc:
            logger.warning("syntax error in %s: %s", file_path, exc)
            return FileSymbols(file_path=file_path, parse_error=f"SyntaxError: {exc}")

        visitor = _SymbolVisitor()
        visitor.visit(tree)

        return FileSymbols(
            file_path=file_path,
            classes=visitor.classes,
            functions=visitor.functions,
            methods=visitor.methods,
            imports=visitor.imports,
            calls=visitor.calls,
        )

    def to_dict(self) -> dict[str, dict[str, object]]:
        """Legacy-style flattened view for code that still wants plain dicts."""
        out: dict[str, dict[str, object]] = {}
        for path, syms in self.symbol_table.items():
            out[path] = {
                "classes": [c.__dict__ for c in syms.classes],
                "functions": [f.__dict__ for f in syms.functions],
                "methods": [m.__dict__ for m in syms.methods],
                "imports": [i.__dict__ for i in syms.imports],
                "calls": [c.__dict__ for c in syms.calls],
                "symbols": syms.symbol_names,
                "parse_error": syms.parse_error,
            }
        return out
