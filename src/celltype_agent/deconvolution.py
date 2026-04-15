"""LDA-based topic deconvolution for Visium spatial transcriptomics."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_N_TOP_GENES_PER_TOPIC = 15  # genes stored per topic
_N_PROMPT_GENES = 10  # genes shown in the prompt by default


def run_lda(
    adata,
    k: Optional[int] = None,
    max_k: int = 20,
    min_k: int = 3,
    n_top_genes: int = 2000,
    random_state: int = 42,
) -> dict:
    """Run LDA deconvolution on a spatial (or single-cell) AnnData.

    Parameters
    ----------
    adata:
        AnnData with expression data in ``X``. Raw counts preferred; normalised
        values are rounded to non-negative integers before fitting.
    k:
        Fixed number of topics. When ``None`` the best K is auto-selected
        from ``[min_k, max_k]`` using held-out perplexity.
    max_k:
        Upper bound for the auto-K search.
    min_k:
        Lower bound for the auto-K search.
    n_top_genes:
        Number of highly-variable genes to retain before fitting.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:

    * ``n_topics`` — int
    * ``topic_genes`` — ``{topic_id: [(gene, weight), ...]}`` (top 15 genes)
    * ``spot_topic_proportions`` — numpy array ``(n_spots, n_topics)``
    * ``coherence_scores`` — ``{k: perplexity}`` from auto-K search (empty if k fixed)
    * ``best_k`` — int
    """
    try:
        from sklearn.decomposition import LatentDirichletAllocation
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for LDA deconvolution. "
            "Install it with: pip install celltype-agent[spatial]"
        ) from exc

    import scipy.sparse

    # ------------------------------------------------------------------
    # 1. Subset to highly-variable genes
    # ------------------------------------------------------------------
    if "highly_variable" not in adata.var.columns:
        try:
            import scanpy as sc  # noqa: PLC0415

            # "seurat_v3" expects raw counts and is more numerically stable than "seurat"
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
            log.debug("Computed HVGs; selected %d genes.", adata.var["highly_variable"].sum())
        except ImportError:
            log.warning(
                "scanpy not available; using all %d genes for LDA.", adata.n_vars
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "HVG computation failed (%s); using all %d genes.", exc, adata.n_vars
            )

    if "highly_variable" in adata.var.columns:
        mask = adata.var["highly_variable"].values
        X_raw = adata.X[:, mask]
        gene_names: list[str] = adata.var_names[mask].tolist()
    else:
        X_raw = adata.X
        gene_names = adata.var_names.tolist()

    # Convert to dense non-negative integers
    if scipy.sparse.issparse(X_raw):
        X_raw = X_raw.toarray()
    X: np.ndarray = np.maximum(np.round(np.asarray(X_raw, dtype=np.float64)), 0).astype(
        np.int32
    )

    n_spots, n_genes = X.shape
    log.info("Running LDA on %d spots × %d genes.", n_spots, n_genes)

    # ------------------------------------------------------------------
    # 2. Auto-select K (if not fixed)
    # ------------------------------------------------------------------
    effective_max_k = min(max_k, n_spots - 1, n_genes)
    effective_min_k = max(min_k, 2)
    k_range = range(effective_min_k, effective_max_k + 1)

    if k is not None:
        best_k: int = k
        coherence_scores: dict[int, float] = {}
    else:
        log.info("Auto-selecting K from range [%d, %d]…", effective_min_k, effective_max_k)
        best_k, coherence_scores = _auto_select_k(X, k_range=k_range, random_state=random_state)
        log.info("Selected K=%d (perplexity=%.1f).", best_k, coherence_scores.get(best_k, 0.0))

    # ------------------------------------------------------------------
    # 3. Fit final model with best K
    # ------------------------------------------------------------------
    lda = LatentDirichletAllocation(
        n_components=best_k,
        random_state=random_state,
        max_iter=100,
        learning_method="online",
    )
    spot_topic_proportions: np.ndarray = lda.fit_transform(X)  # (n_spots, best_k)

    # ------------------------------------------------------------------
    # 4. Extract top genes per topic
    # ------------------------------------------------------------------
    topic_genes: dict[int, list[tuple[str, float]]] = {}
    for t_idx in range(best_k):
        weights = lda.components_[t_idx]
        weights = weights / weights.sum()  # normalise to proportions
        top_idx = np.argsort(weights)[::-1][:_N_TOP_GENES_PER_TOPIC]
        topic_genes[t_idx] = [(gene_names[i], float(weights[i])) for i in top_idx]

    return {
        "n_topics": best_k,
        "topic_genes": topic_genes,
        "spot_topic_proportions": spot_topic_proportions,
        "coherence_scores": coherence_scores,
        "best_k": best_k,
    }


def _auto_select_k(
    X: np.ndarray,
    k_range: range,
    random_state: int,
) -> tuple[int, dict[int, float]]:
    """Pick K by minimising held-out perplexity.

    Returns
    -------
    (best_k, {k: perplexity})
    """
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.model_selection import train_test_split

    n_samples = X.shape[0]

    if n_samples < 10:
        # Too few samples to split — return the smallest valid K
        return k_range.start, {k_range.start: float("inf")}

    test_frac = max(0.1, min(0.2, 10.0 / n_samples))
    X_train, X_test = train_test_split(X, test_size=test_frac, random_state=random_state)

    perplexities: dict[int, float] = {}
    for k in k_range:
        if k >= X_train.shape[0]:
            break
        lda = LatentDirichletAllocation(
            n_components=k,
            random_state=random_state,
            max_iter=50,
            learning_method="online",
        )
        lda.fit(X_train)
        try:
            perp = float(lda.perplexity(X_test))
        except Exception:
            continue
        perplexities[k] = perp
        log.debug("K=%d: perplexity=%.1f", k, perp)

    if not perplexities:
        return k_range.start, {}

    best_k = min(perplexities, key=perplexities.__getitem__)
    return best_k, perplexities


def format_topics_for_prompt(
    topic_genes: dict[int, list[tuple[str, float]]],
    n_genes: int = _N_PROMPT_GENES,
) -> str:
    """Format LDA topics as a plain-text table suitable for Claude's prompt.

    Parameters
    ----------
    topic_genes:
        Mapping of topic_id → list of (gene, weight) tuples as returned by
        :func:`run_lda`.
    n_genes:
        Number of top genes to include per topic.

    Returns
    -------
    Multi-line string with one row per topic.
    """
    lines = ["Topic | Top genes (by weight)"]
    lines.append("------+-" + "-" * 60)
    for topic_id in sorted(topic_genes.keys()):
        genes = topic_genes[topic_id][:n_genes]
        gene_str = ", ".join(g for g, _ in genes)
        lines.append(f"  {topic_id:3d} | {gene_str}")
    return "\n".join(lines)
