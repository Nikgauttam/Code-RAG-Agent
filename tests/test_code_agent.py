"""Tests for the CodeAgent command registry + ask() routing."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("faiss")


@pytest.fixture(scope="module")
def agent_fixture(tmp_path_factory, shared_embedder, shared_ranker):  # type: ignore[no-untyped-def]
    """One agent, set up once, shared across all tests in this module."""
    repo = tmp_path_factory.mktemp("agent_repo")
    storage = tmp_path_factory.mktemp("agent_storage")

    (repo / "a.py").write_text(
        textwrap.dedent("""
            def add(x, y):
                return x + y

            class Calculator:
                def multiply(self, x, y):
                    return x * y
        """).lstrip()
    )
    (repo / "b.py").write_text(
        textwrap.dedent("""
            from a import add, Calculator

            def compute(values):
                total = 0
                for v in values:
                    total = add(total, v)
                return total
        """).lstrip()
    )

    from agent.code_agent import CodeAgent
    from core.config import Config, StorageConfig

    cfg = Config(storage=StorageConfig(directory=str(storage)))
    ag = CodeAgent(
        str(repo),
        config=cfg,
        embedder=shared_embedder,
        cross_ranker=shared_ranker,
    )
    ag.pipeline.llm.generate = lambda p: "STUB_LLM"  # type: ignore[method-assign]
    ag.setup()
    return ag


def test_help_text_lists_all_commands() -> None:
    from agent.code_agent import COMMANDS, CodeAgent

    text = CodeAgent.help_text()
    for cmd in COMMANDS:
        assert cmd.prefix.strip() in text


def test_ask_routes_to_explain_handler(agent_fixture) -> None:  # type: ignore[no-untyped-def]
    out = agent_fixture.ask("explain add")
    assert isinstance(out, str)
    assert "STUB_LLM" in out


def test_ask_routes_to_usages_handler(agent_fixture) -> None:  # type: ignore[no-untyped-def]
    out = agent_fixture.ask("find usages of add")
    assert isinstance(out, dict)
    assert out["symbol"] == "add"


def test_ask_before_setup_raises(tmp_path: Path) -> None:
    from agent.code_agent import AgentNotReady, CodeAgent
    from core.config import Config, StorageConfig

    cfg = Config(storage=StorageConfig(directory=str(tmp_path)))
    ag = CodeAgent(str(tmp_path), config=cfg)
    with pytest.raises(AgentNotReady):
        ag.ask("anything")


def test_ask_falls_through_to_generate_for_freeform(agent_fixture) -> None:  # type: ignore[no-untyped-def]
    out = agent_fixture.ask("how does compute work?")
    assert isinstance(out, dict)
    assert "answer" in out
