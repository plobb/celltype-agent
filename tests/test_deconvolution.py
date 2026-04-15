"""Tests for LDA deconvolution utilities."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="scikit-learn required for deconvolution tests")

import anndata as ad
import pandas as pd
import scipy.sparse

from celltype_agent.deconvolution import format_topics_for_prompt, run_lda


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_structured_adata(
    n_spots: int = 60,
    n_genes: int = 120,
    n_programs: int = 5,
    random_state: int = 42,
) -> ad.AnnData:
    """Create a synthetic AnnData with *n_programs* separable gene programs.

    Each program activates a disjoint block of genes, so LDA should be able to
    recover approximately *n_programs* coherent topics.
    """
    rng = np.random.default_rng(random_state)
    genes_per_prog = n_genes // n_programs

    # Program × gene weight matrix (block-diagonal structure)
    program_weights = np.zeros((n_programs, n_genes))
    for i in range(n_programs):
        start = i * genes_per_prog
        end = start + genes_per_prog
        program_weights[i, start:end] = rng.exponential(1.0, size=end - start)
    # Normalise to probability distributions
    program_weights = program_weights / program_weights.sum(axis=1, keepdims=True)

    # Spot × program mixture (Dirichlet, concentrated on one program per spot)
    alpha = np.ones(n_programs) * 0.1
    spot_mixtures = rng.dirichlet(alpha, size=n_spots)  # (n_spots, n_programs)

    # Generate count matrix via multinomial sampling
    total_counts = rng.integers(200, 600, size=n_spots)
    X = np.zeros((n_spots, n_genes), dtype=np.float32)
    for i in range(n_spots):
        probs = spot_mixtures[i] @ program_weights
        probs = np.clip(probs, 0, None)
        probs /= probs.sum()
        X[i] = rng.multinomial(total_counts[i], probs).astype(np.float32)

    obs = pd.DataFrame(index=[f"spot_{i}" for i in range(n_spots)])
    var = pd.DataFrame(index=[f"GENE{i:03d}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# run_lda — basic functionality
# ---------------------------------------------------------------------------


def test_run_lda_fixed_k():
    adata = _make_structured_adata()
    result = run_lda(adata, k=5, random_state=0)
    assert result["n_topics"] == 5
    assert result["best_k"] == 5
    assert len(result["topic_genes"]) == 5
    assert result["coherence_scores"] == {}


def test_run_lda_returns_required_keys():
    adata = _make_structured_adata()
    result = run_lda(adata, k=4, random_state=0)
    for key in ("n_topics", "topic_genes", "spot_topic_proportions", "coherence_scores", "best_k"):
        assert key in result, f"Missing key: {key}"


def test_run_lda_spot_proportions_shape():
    adata = _make_structured_adata(n_spots=60, n_genes=120)
    result = run_lda(adata, k=4, random_state=0)
    proportions = result["spot_topic_proportions"]
    assert proportions.shape == (60, 4)


def test_run_lda_spot_proportions_sum_to_one():
    adata = _make_structured_adata()
    result = run_lda(adata, k=5, random_state=0)
    row_sums = result["spot_topic_proportions"].sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_run_lda_topic_genes_structure():
    adata = _make_structured_adata()
    result = run_lda(adata, k=5, random_state=0)
    topic_genes = result["topic_genes"]
    # Each topic should map to a list of (gene_name, weight) tuples
    for t_id, genes in topic_genes.items():
        assert isinstance(t_id, int)
        assert len(genes) > 0
        for gene, weight in genes:
            assert isinstance(gene, str)
            assert 0.0 <= weight <= 1.0


def test_run_lda_gene_names_are_from_adata():
    adata = _make_structured_adata(n_genes=80)
    result = run_lda(adata, k=4, random_state=0)
    all_var_names = set(adata.var_names)
    for genes in result["topic_genes"].values():
        for gene, _ in genes:
            assert gene in all_var_names


def test_run_lda_sparse_input():
    adata = _make_structured_adata()
    adata.X = scipy.sparse.csr_matrix(adata.X)
    result = run_lda(adata, k=4, random_state=0)
    assert result["n_topics"] == 4


def test_run_lda_non_negative_counts():
    """LDA should handle slightly negative values (e.g. from normalisation) gracefully."""
    adata = _make_structured_adata()
    adata.X = adata.X - 0.01  # introduce tiny negatives
    result = run_lda(adata, k=4, random_state=0)
    assert result["n_topics"] == 4


# ---------------------------------------------------------------------------
# run_lda — auto-K selection
# ---------------------------------------------------------------------------


def test_auto_k_returns_in_range():
    """Auto-selected K should lie within [min_k, max_k]."""
    adata = _make_structured_adata(n_spots=80, n_genes=120, n_programs=5)
    result = run_lda(adata, k=None, min_k=3, max_k=8, random_state=0)
    assert 3 <= result["best_k"] <= 8


def test_auto_k_coherence_scores_populated():
    adata = _make_structured_adata(n_spots=80, n_genes=120, n_programs=4)
    result = run_lda(adata, k=None, min_k=3, max_k=6, random_state=0)
    scores = result["coherence_scores"]
    assert len(scores) > 0
    for k_val, score in scores.items():
        assert isinstance(k_val, int)
        assert score > 0  # perplexity is always positive


def test_auto_k_best_k_in_scores():
    adata = _make_structured_adata(n_spots=80, n_genes=120, n_programs=4)
    result = run_lda(adata, k=None, min_k=3, max_k=6, random_state=0)
    assert result["best_k"] in result["coherence_scores"]


def test_auto_k_small_dataset():
    """Auto-K should not crash on a very small dataset."""
    adata = _make_structured_adata(n_spots=15, n_genes=40, n_programs=3)
    result = run_lda(adata, k=None, min_k=2, max_k=5, random_state=0)
    assert result["n_topics"] >= 2


# ---------------------------------------------------------------------------
# format_topics_for_prompt
# ---------------------------------------------------------------------------


def test_format_topics_for_prompt_basic():
    topic_genes: dict = {
        0: [("CD3D", 0.1), ("CD3E", 0.08), ("CD4", 0.07)],
        1: [("MS4A1", 0.12), ("CD79A", 0.09)],
    }
    text = format_topics_for_prompt(topic_genes, n_genes=3)
    assert "CD3D" in text
    assert "MS4A1" in text
    assert "0" in text
    assert "1" in text


def test_format_topics_for_prompt_respects_n_genes():
    # Each topic has 5 genes, ask for only 2
    topic_genes = {
        0: [(f"GENE{i}", 0.1 - i * 0.01) for i in range(5)],
    }
    text = format_topics_for_prompt(topic_genes, n_genes=2)
    # Only first 2 genes should appear
    assert "GENE0" in text
    assert "GENE1" in text
    assert "GENE2" not in text


def test_format_topics_for_prompt_sorted():
    topic_genes = {2: [("A", 0.1)], 0: [("B", 0.1)], 1: [("C", 0.1)]}
    text = format_topics_for_prompt(topic_genes)
    pos_0 = text.index("  0 |")
    pos_1 = text.index("  1 |")
    pos_2 = text.index("  2 |")
    assert pos_0 < pos_1 < pos_2


def test_format_topics_for_prompt_returns_string():
    topic_genes = {0: [("GAPDH", 0.05)]}
    result = format_topics_for_prompt(topic_genes)
    assert isinstance(result, str)
    assert len(result) > 0
