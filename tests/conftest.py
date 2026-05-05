"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    """Two-file fixture repo with import edge b.py → a.py."""
    (tmp_path / "a.py").write_text(
        textwrap.dedent(
            '''
            def add(x, y):
                """Return the sum of two numbers."""
                return x + y


            class Calculator:
                def multiply(self, x, y):
                    return x * y

                def square(self, x):
                    return self.multiply(x, x)
            '''
        ).lstrip()
    )
    (tmp_path / "b.py").write_text(
        textwrap.dedent(
            """
            from a import add, Calculator


            def compute(values):
                total = 0
                for v in values:
                    total = add(total, v)
                return total


            def double(x):
                return Calculator().multiply(x, 2)
            """
        ).lstrip()
    )
    return tmp_path


@pytest.fixture
def package_repo(tmp_path: Path) -> Path:
    """Repo with a package + relative imports + a same-name shadow file."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "utils.py").write_text("def helper():\n    return 1\n")
    (tmp_path / "pkg" / "sub").mkdir()
    (tmp_path / "pkg" / "sub" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "sub" / "mod.py").write_text(
        "from ..utils import helper\n\ndef use():\n    return helper()\n"
    )
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "utils.py").write_text("def unrelated():\n    return 0\n")
    return tmp_path


# ------------------------------------------------------------------ #
# Session-scoped ML models — load ONCE for the entire test session.
# Heavy sentence-transformer and cross-encoder models segfault on
# macOS/Apple Silicon when instantiated more than once in the same
# process, so we share a single instance across all integration tests.
# ------------------------------------------------------------------ #


@pytest.fixture(scope="session")
def shared_embedder():  # type: ignore[no-untyped-def]
    """One CodeEmbedder instance shared across the whole test session."""
    try:
        from core.retrieval.embedder import CodeEmbedder

        return CodeEmbedder()
    except Exception:
        return None


@pytest.fixture(scope="session")
def shared_ranker():  # type: ignore[no-untyped-def]
    """One CrossEncoderRanker instance shared across the whole test session."""
    try:
        from core.rerank.cross_encoder_ranker import CrossEncoderRanker

        return CrossEncoderRanker()
    except Exception:
        return None
