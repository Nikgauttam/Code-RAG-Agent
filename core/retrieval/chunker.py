"""Code chunker.

Strategy:
  * One chunk per top-level function and class.
  * One chunk per method (so a 1000-line class doesn't dilute into a
    single embedding).
  * Each chunk carries a *stable* id (``stable_id``) of the form
    ``rel/path::QualName@startline`` that survives serialization, so it
    can be used as the FAISS vector id and for incremental delta updates.
  * Each chunk carries a SHA-256 ``content_hash`` so the indexer can do
    fast diff-based reindexing (skip chunks whose content hasn't changed).
  * Methods optionally include a one-line ``class context`` header
    (``class Foo:``) to give the embedder enough scope to disambiguate.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from core.config import DEFAULT_CONFIG, ChunkConfig
from core.parser.ast_parser import FileSymbols, SymbolDef

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    file: str           # absolute path
    rel_path: str       # repo-relative path
    type: str           # "function" | "method" | "class"
    name: str           # short name
    qualname: str       # fully-qualified name
    lineno: int         # 1-based start line
    end_lineno: int     # 1-based end line (inclusive in source)
    content: str        # source text of the chunk
    content_hash: str   # sha256 of content
    stable_id: str      # rel_path::qualname@lineno
    parent: str | None = None  # parent class qualname for methods
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    # Allow code that still treats chunks like dicts to keep working.
    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def get(self, key: str, default: object = None) -> object:
        return getattr(self, key, default)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class CodeChunker:
    def __init__(
        self,
        symbol_table: Mapping[str, FileSymbols],
        repo_root: str | None = None,
        config: ChunkConfig | None = None,
    ):
        self.symbol_table = symbol_table
        self.config = config or DEFAULT_CONFIG.chunk
        if repo_root:
            self.repo_root = os.path.abspath(repo_root)
        elif symbol_table:
            self.repo_root = os.path.commonpath([os.path.abspath(p) for p in symbol_table])
        else:
            self.repo_root = ""

    def chunk_repository(self) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        for file_path, syms in self.symbol_table.items():
            chunks.extend(self._chunks_for_file(file_path, syms))
        return chunks

    def _chunks_for_file(self, file_path: str, syms: FileSymbols) -> list[CodeChunk]:
        try:
            with open(file_path, encoding="utf-8") as fp:
                lines = fp.readlines()
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("could not read %s for chunking: %s", file_path, exc)
            return []

        rel = self._relpath(file_path)
        out: list[CodeChunk] = []

        for fn in syms.functions:
            out.append(self._make_chunk(file_path, rel, fn, lines))

        for cls in syms.classes:
            out.append(self._make_chunk(file_path, rel, cls, lines))

        for method in syms.methods:
            out.append(self._make_chunk(file_path, rel, method, lines))

        return out

    def _make_chunk(
        self,
        file_path: str,
        rel: str,
        sym: SymbolDef,
        lines: list[str],
    ) -> CodeChunk:
        body = self._extract_block(lines, sym.lineno, sym.end_lineno)
        if (
            self.config.include_class_context
            and sym.kind == "method"
            and sym.parent
        ):
            body = f"# (in class {sym.parent})\n{body}"

        return CodeChunk(
            file=file_path,
            rel_path=rel,
            type=sym.kind if sym.kind != "async_function" else "function",
            name=sym.name,
            qualname=sym.qualname,
            lineno=sym.lineno,
            end_lineno=sym.end_lineno,
            content=body,
            content_hash=_hash(body),
            stable_id=f"{rel}::{sym.qualname}@{sym.lineno}",
            parent=sym.parent,
        )

    def _extract_block(
        self,
        lines: list[str],
        start_line: int,
        end_line: int | None,
    ) -> str:
        start = max(start_line - 1, 0)
        end = min(start + 40, len(lines)) if end_line is None else min(end_line, len(lines))
        end = min(end, start + self.config.max_lines)
        return "".join(lines[start:end])

    def _relpath(self, file_path: str) -> str:
        if not self.repo_root:
            return file_path
        try:
            return os.path.relpath(file_path, self.repo_root)
        except ValueError:
            return file_path
