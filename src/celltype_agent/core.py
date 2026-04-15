"""High-level annotate() entry point."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from .agent import annotate_clusters
from .markers import extract_markers
from .models import AnnotationResult, CellTypeAnnotation

log = logging.getLogger(__name__)


def annotate(
    adata,
    *,
    species: str = "human",
    tissue: Optional[str] = None,
    cluster_key: str = "leiden",
    n_markers: int = 10,
    method: str = "wilcoxon",
    api_key: Optional[str] = None,
    model: str = "claude-opus-4-6",
    add_to_obs: bool = True,
    obs_key: str = "cell_type",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> AnnotationResult:
    """Annotate cell clusters in *adata* using Claude.

    Parameters
    ----------
    adata:
        AnnData object.  Must have cluster labels in ``adata.obs[cluster_key]``.
    species:
        ``'human'`` or ``'mouse'``.
    tissue:
        Optional tissue name for context-aware annotation (e.g. ``'PBMC'``,
        ``'lung'``, ``'bone marrow'``).
    cluster_key:
        Column in ``adata.obs`` with cluster labels.
    n_markers:
        Number of top markers to extract per cluster.
    method:
        DE method for ``sc.tl.rank_genes_groups`` if not pre-computed.
    api_key:
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    model:
        Claude model identifier.
    add_to_obs:
        If ``True`` (default), write predicted cell types back into
        ``adata.obs[obs_key]``.
    obs_key:
        Column name to write cell type labels when *add_to_obs* is ``True``.

    Returns
    -------
    :class:`~celltype_agent.models.AnnotationResult` containing one
    :class:`~celltype_agent.models.CellTypeAnnotation` per cluster.

    Examples
    --------
    >>> import scanpy as sc
    >>> from celltype_agent import annotate
    >>> adata = sc.datasets.pbmc3k_processed()
    >>> result = annotate(adata, species="human", tissue="PBMC")
    >>> print(result.to_labels())
    """
    def _progress(msg: str) -> None:
        log.info(msg)
        if progress_callback is not None:
            progress_callback(msg)

    species = species.lower().strip()
    if species not in {"human", "mouse"}:
        raise ValueError(f"species must be 'human' or 'mouse', got '{species}'")

    # --- 1. Extract markers ---------------------------------------------------
    _progress(f"Extracting top {n_markers} markers per cluster (key='{cluster_key}') …")
    markers = extract_markers(
        adata,
        cluster_key=cluster_key,
        n_markers=n_markers,
        method=method,
    )
    _progress(f"Found {len(markers)} clusters. Sending to {model} for annotation …")

    # --- 2. Call the agent ---------------------------------------------------
    annotations: list[CellTypeAnnotation] = annotate_clusters(
        markers,
        species=species,
        tissue=tissue,
        api_key=api_key,
        model=model,
    )
    _progress(f"Annotation complete — {len(annotations)} cluster(s) annotated.")

    result = AnnotationResult(
        annotations=annotations,
        species=species,
        tissue=tissue,
        n_clusters=len(markers),
        model_used=model,
    )

    # --- 3. Optionally write back to adata -----------------------------------
    if add_to_obs:
        label_map = result.to_labels()
        adata.obs[obs_key] = adata.obs[cluster_key].astype(str).map(label_map)
        log.info("Wrote cell type labels to adata.obs['%s'].", obs_key)

    return result
