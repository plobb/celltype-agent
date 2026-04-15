"""Tests for AnnotationResult.to_methods() and to_narrative()."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from celltype_agent import __version__
from celltype_agent.models import AnnotationResult, CellTypeAnnotation


def _make_result(**kwargs) -> AnnotationResult:
    annotations = [
        CellTypeAnnotation(
            cluster_id="0",
            predicted_type="CD4+ T cell",
            confidence=0.92,
            markers_used=["CD3D", "CD3E", "CD4", "IL7R"],
            reasoning="CD3D/E are pan-T markers; CD4 and IL7R confirm helper T identity.",
            database_support="PanglaoDB + CellMarker confirmed",
            database_markers_matched=4,
            database_markers_total=4,
        ),
        CellTypeAnnotation(
            cluster_id="1",
            predicted_type="B cell",
            confidence=0.88,
            markers_used=["MS4A1", "CD79A", "CD79B"],
            reasoning="MS4A1 (CD20) and CD79A/B are canonical B cell markers.",
            database_support="Both databases confirmed B cell",
            database_markers_matched=3,
            database_markers_total=3,
        ),
        CellTypeAnnotation(
            cluster_id="2",
            predicted_type="Unknown",
            confidence=0.35,
            markers_used=["GENE1", "GENE2"],
            reasoning="Unclear marker pattern.",
        ),
    ]
    defaults = dict(n_markers=10, model_used="claude-opus-4-6")
    defaults.update(kwargs)
    return AnnotationResult(
        annotations=annotations,
        species="human",
        tissue="PBMC",
        n_clusters=3,
        **defaults,
    )


# ---------------------------------------------------------------------------
# to_methods
# ---------------------------------------------------------------------------


def test_to_methods_returns_string():
    result = _make_result()
    methods = result.to_methods()
    assert isinstance(methods, str)
    assert len(methods) > 50


def test_to_methods_contains_version():
    result = _make_result()
    assert __version__ in result.to_methods()


def test_to_methods_contains_model():
    result = _make_result()
    assert "claude-opus-4-6" in result.to_methods()


def test_to_methods_contains_n_markers():
    result = _make_result()
    assert "10" in result.to_methods()


def test_to_methods_mentions_databases():
    result = _make_result()
    methods = result.to_methods()
    assert "PanglaoDB" in methods
    assert "CellMarker" in methods


def test_to_methods_mentions_wilcoxon():
    result = _make_result()
    assert "Wilcoxon" in result.to_methods()


def test_to_methods_mentions_strategy():
    result = _make_result()
    methods = result.to_methods()
    assert "hypothesise" in methods or "hypothesize" in methods or "hypothesise-then-verify" in methods


def test_to_methods_uses_custom_n_markers():
    result = _make_result(n_markers=15)
    assert "15" in result.to_methods()


def test_to_methods_uses_custom_model():
    result = _make_result(model_used="claude-sonnet-4-6")
    assert "claude-sonnet-4-6" in result.to_methods()


# ---------------------------------------------------------------------------
# to_narrative (mocked)
# ---------------------------------------------------------------------------


def _mock_anthropic_response(text: str):
    """Build a minimal mock that mimics anthropic.Anthropic().messages.create()."""
    content_block = MagicMock()
    content_block.text = text

    response = MagicMock()
    response.content = [content_block]

    client = MagicMock()
    client.messages.create.return_value = response
    return client


def test_to_narrative_returns_string():
    result = _make_result()
    fake_client = _mock_anthropic_response("The dataset contained T cells and B cells.")

    with patch("anthropic.Anthropic", return_value=fake_client):
        narrative = result.to_narrative(api_key="fake-key")

    assert isinstance(narrative, str)
    assert len(narrative) > 0


def test_to_narrative_strips_whitespace():
    result = _make_result()
    fake_client = _mock_anthropic_response("  Some narrative.  ")

    with patch("anthropic.Anthropic", return_value=fake_client):
        narrative = result.to_narrative(api_key="fake-key")

    assert narrative == "Some narrative."


def test_to_narrative_passes_api_key():
    result = _make_result()
    fake_client = _mock_anthropic_response("narrative")

    with patch("anthropic.Anthropic", return_value=fake_client) as mock_cls:
        result.to_narrative(api_key="my-test-key")

    mock_cls.assert_called_once_with(api_key="my-test-key")


def test_to_narrative_uses_env_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-123")
    result = _make_result()
    fake_client = _mock_anthropic_response("narrative")

    with patch("anthropic.Anthropic", return_value=fake_client) as mock_cls:
        result.to_narrative()

    mock_cls.assert_called_once_with(api_key="env-key-123")


def test_to_narrative_live_skipped_without_key(monkeypatch):
    """Skip the live API test when no real key is present."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("No ANTHROPIC_API_KEY set — skipping live narrative test")
