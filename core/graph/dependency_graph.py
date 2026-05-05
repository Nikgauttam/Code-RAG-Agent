"""Dependency graphs derived from a parsed symbol table.

Two distinct graphs are built and exposed separately so file-path nodes
and symbol-name nodes never live in the same dict (which the previous
implementation did, causing subtle correctness bugs):

  * ``FileGraph``    : ``file_path -> [file_path, ...]`` based on resolved
                       imports (absolute and relative).
  * ``SymbolGraph``  : ``caller_qualname -> [callee_name, ...]`` built from
                       call sites tagged with their enclosing scope.

A combined ``DependencyGraph`` facade exposes both and a backwards-
compatible ``.graph`` mapping (file -> [files]) for callers that only
care about file-level relations.

Import resolution rules:
  * ``from foo.bar import baz`` resolves to the file that *implements*
    ``foo.bar`` (i.e. ``<root>/foo/bar.py`` or ``<root>/foo/bar/__init__.py``).
    If ``baz`` itself is a submodule, both files are linked.
  * Relative imports (``from .foo import x``) are resolved against the
    importing file's package, walking up ``level`` directories.
  * Module names are matched against fully-qualified module paths
    (``a.b.c``), not bare basenames — so two ``utils.py`` files in
    different packages no longer cross-wire.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field

from core.parser.ast_parser import FileSymbols, ImportEntry

logger = logging.getLogger(__name__)


def _module_path_for_file(root: str, file_path: str) -> str:
    """Convert /root/a/b/c.py -> 'a.b.c'  and  /root/a/b/__init__.py -> 'a.b'."""
    rel = os.path.relpath(file_path, root)
    rel_no_ext, _ = os.path.splitext(rel)
    parts = rel_no_ext.split(os.sep)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(p for p in parts if p)


@dataclass
class FileGraph:
    """Adjacency list of file -> imported files."""

    edges: dict[str, list[str]] = field(default_factory=dict)

    def neighbors(self, node: str) -> list[str]:
        return self.edges.get(node, [])

    def __getitem__(self, key: str) -> list[str]:
        return self.edges.get(key, [])

    def __contains__(self, key: str) -> bool:
        return key in self.edges

    def __iter__(self) -> Iterator[str]:
        return iter(self.edges)

    def get(self, key: str, default: list[str] | None = None) -> list[str]:
        return self.edges.get(key, default if default is not None else [])

    def items(self) -> Iterable[tuple[str, list[str]]]:
        return self.edges.items()

    def dfs(self, start: str, max_depth: int | None = None) -> list[str]:
        """Iterative DFS reachable set; bounded by ``max_depth`` if given."""
        visited: set[str] = set()
        order: list[str] = []
        stack: list[tuple[str, int]] = [(start, 0)]
        while stack:
            node, depth = stack.pop()
            if node in visited:
                continue
            if max_depth is not None and depth > max_depth:
                continue
            visited.add(node)
            order.append(node)
            for nb in self.edges.get(node, []):
                stack.append((nb, depth + 1))
        return order


@dataclass
class SymbolGraph:
    """Adjacency list of caller qualname -> [callee names]."""

    edges: dict[str, list[str]] = field(default_factory=dict)

    def neighbors(self, qualname: str) -> list[str]:
        return self.edges.get(qualname, [])

    def __getitem__(self, key: str) -> list[str]:
        return self.edges.get(key, [])

    def __contains__(self, key: str) -> bool:
        return key in self.edges

    def items(self) -> Iterable[tuple[str, list[str]]]:
        return self.edges.items()


class DependencyGraph:
    """Builds and exposes file- and symbol-level dependency graphs."""

    def __init__(
        self,
        symbol_table: Mapping[str, FileSymbols],
        repo_root: str | None = None,
    ):
        if not symbol_table:
            self.repo_root = os.path.abspath(repo_root) if repo_root else ""
        else:
            inferred = repo_root or _common_root(symbol_table.keys())
            self.repo_root = os.path.abspath(inferred)

        self.symbol_table = symbol_table
        self.file_graph = FileGraph()
        self.symbol_graph = SymbolGraph()
        self._module_to_file: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #

    def build(self) -> DependencyGraph:
        self._index_modules()
        self._build_file_graph()
        self._build_symbol_graph()
        return self

    # Backwards-compatible single-call API.
    def build_graph(self) -> dict[str, list[str]]:
        self.build()
        return self.file_graph.edges

    # ------------------------------------------------------------------ #
    # File graph
    # ------------------------------------------------------------------ #

    def _index_modules(self) -> None:
        for file_path in self.symbol_table:
            module = _module_path_for_file(self.repo_root, file_path)
            if module:
                self._module_to_file[module] = file_path

    def _resolve_module(self, module: str) -> str | None:
        if not module:
            return None
        if module in self._module_to_file:
            return self._module_to_file[module]
        # Walk up: `from a.b.c import x` may map to `a.b` (a package whose
        # __init__ exposes the symbol), even if `a.b.c` isn't its own file.
        parts = module.split(".")
        for i in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in self._module_to_file:
                return self._module_to_file[candidate]
        return None

    def _resolve_relative(self, importer: str, imp: ImportEntry) -> str | None:
        """Resolve `from .x import y` against the importer's package."""
        if imp.level <= 0:
            return None
        importer_module = _module_path_for_file(self.repo_root, importer)
        importer_parts = importer_module.split(".") if importer_module else []
        # `level=1` means current package: drop the trailing module name.
        # `level=2` means parent package: drop one more.
        drop = imp.level
        if drop > len(importer_parts):
            return None
        base_parts = importer_parts[: len(importer_parts) - drop]
        target_parts = base_parts + ([imp.module] if imp.module else [])
        target_module = ".".join(p for p in target_parts if p)
        if not target_module:
            return None
        resolved = self._resolve_module(target_module)
        if resolved:
            return resolved
        if imp.name:
            return self._resolve_module(f"{target_module}.{imp.name}")
        return None

    def _build_file_graph(self) -> None:
        for file_path, syms in self.symbol_table.items():
            self.file_graph.edges.setdefault(file_path, [])
            seen: set[str] = set()
            for imp in syms.imports:
                target = (
                    self._resolve_relative(file_path, imp)
                    if imp.level > 0
                    else self._resolve_absolute(imp)
                )
                if not target or target == file_path or target in seen:
                    continue
                seen.add(target)
                self.file_graph.edges[file_path].append(target)

    def _resolve_absolute(self, imp: ImportEntry) -> str | None:
        # Try the most specific form first.
        if imp.module and imp.name:
            specific = self._resolve_module(f"{imp.module}.{imp.name}")
            if specific:
                return specific
        if imp.module:
            return self._resolve_module(imp.module)
        if imp.name:
            return self._resolve_module(imp.name)
        return None

    # ------------------------------------------------------------------ #
    # Symbol (call) graph
    # ------------------------------------------------------------------ #

    def _build_symbol_graph(self) -> None:
        for syms in self.symbol_table.values():
            for call in syms.calls:
                if not call.callee or not call.caller_qualname:
                    continue
                self.symbol_graph.edges.setdefault(call.caller_qualname, []).append(
                    call.callee
                )

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    @property
    def graph(self) -> dict[str, list[str]]:
        """Backwards-compat alias for the file graph adjacency dict."""
        return self.file_graph.edges

    def dfs_related_files(self, start_file: str, max_depth: int | None = None) -> list[str]:
        return self.file_graph.dfs(start_file, max_depth=max_depth)


def _common_root(paths: Iterable[str]) -> str:
    paths = [os.path.abspath(p) for p in paths]
    if not paths:
        return ""
    return os.path.commonpath(paths) if len(paths) > 1 else os.path.dirname(paths[0])
