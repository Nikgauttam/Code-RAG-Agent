"""Tests for the central Config dataclass."""

from __future__ import annotations

from core.config import DEFAULT_CONFIG, Config


def test_default_config_has_expected_values() -> None:
    cfg = DEFAULT_CONFIG
    assert cfg.retrieval.top_k == 5
    assert 0.0 <= cfg.retrieval.semantic_weight <= 1.0
    assert 0.0 <= cfg.retrieval.graph_weight <= 1.0
    assert cfg.embedding.normalize is True
    assert cfg.storage.schema_version >= 1


def test_from_env_overrides(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CODE_AGENT_TOP_K", "11")
    monkeypatch.setenv("CODE_AGENT_GRAPH_MAX_DEPTH", "4")
    monkeypatch.setenv("CODE_AGENT_MODEL", "mistral")
    monkeypatch.setenv("CODE_AGENT_SEMANTIC_WEIGHT", "0.5")

    cfg = Config.from_env()
    assert cfg.retrieval.top_k == 11
    assert cfg.retrieval.graph_max_depth == 4
    assert cfg.llm.model == "mistral"
    assert cfg.retrieval.semantic_weight == 0.5


def test_from_env_falls_back_to_defaults_for_missing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("CODE_AGENT_TOP_K", raising=False)
    cfg = Config.from_env()
    assert cfg.retrieval.top_k == 5
