"""Extract top marker genes per cluster from an AnnData object."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def extract_markers(
    adata,  # anndata.AnnData — avoid hard import so the module loads without scanpy
    cluster_key: str = "leiden",
    n_markers: int = 10,
    method: str = "wilcoxon",
    groupby: Optional[str] = None,
) -> dict[str, list[str]]:
    """Return top *n_markers* genes per cluster.

    If ``rank_genes_groups`` has already been computed (and matches the
    requested *cluster_key*) the cached results are used directly.  Otherwise
    ``scanpy.tl.rank_genes_groups`` is run in-place.

    Parameters
    ----------
    adata:
        AnnData object with cells × genes.
    cluster_key:
        Column in ``adata.obs`` that contains cluster labels.
    n_markers:
        Number of top markers to return per cluster.
    method:
        Differential expression method passed to ``sc.tl.rank_genes_groups``
        when a fresh run is needed.
    groupby:
        If supplied, overrides *cluster_key* for the DE grouping key.

    Returns
    -------
    dict mapping cluster label → list of top marker gene names.
    """
    groupby = groupby or cluster_key

    if groupby not in adata.obs.columns:
        raise ValueError(
            f"Column '{groupby}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    # Check whether cached results are reusable
    rgg = adata.uns.get("rank_genes_groups", {})
    cached_groupby = rgg.get("params", {}).get("groupby")

    if cached_groupby == groupby:
        log.info("Using cached rank_genes_groups (groupby='%s')", groupby)
    else:
        log.info(
            "Running sc.tl.rank_genes_groups (groupby='%s', method='%s') …",
            groupby,
            method,
        )
        try:
            import scanpy as sc  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "scanpy is required to compute marker genes. "
                "Install it with: pip install scanpy"
            ) from exc

        sc.tl.rank_genes_groups(adata, groupby=groupby, method=method)

    return _parse_rank_genes_groups(adata, n_markers=n_markers)


def _parse_rank_genes_groups(adata, n_markers: int) -> dict[str, list[str]]:
    """Parse ``adata.uns['rank_genes_groups']`` into a tidy dict."""
    rgg = adata.uns["rank_genes_groups"]
    groups: list[str] = list(rgg["names"].dtype.names)

    markers: dict[str, list[str]] = {}
    for group in groups:
        names = rgg["names"][group]
        scores = rgg["scores"][group] if "scores" in rgg else np.zeros(len(names))

        # names / scores may be numpy recarrays; coerce to plain lists
        gene_list = [str(g) for g in names[:n_markers]]
        markers[str(group)] = gene_list

    return markers


def format_markers_for_prompt(
    markers: dict[str, list[str]],
    species: str,
    tissue: Optional[str] = None,
) -> str:
    """Render a markdown-style marker table suitable for the Claude prompt."""
    lines: list[str] = []

    if tissue:
        lines.append(f"**Tissue:** {tissue}  |  **Species:** {species}\n")
    else:
        lines.append(f"**Species:** {species}\n")

    lines.append("| Cluster | Top marker genes |")
    lines.append("|---------|-----------------|")

    for cluster_id, genes in sorted(markers.items(), key=lambda x: _sort_key(x[0])):
        gene_str = ", ".join(genes)
        lines.append(f"| {cluster_id} | {gene_str} |")

    return "\n".join(lines)


def _sort_key(label: str):
    """Sort cluster labels numerically where possible, else lexicographically."""
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)
