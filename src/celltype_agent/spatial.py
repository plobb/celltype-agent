"""Spatial transcriptomics platform detection and clustering utilities."""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


def detect_spatial(adata) -> Optional[str]:
    """Infer the spatial platform from AnnData metadata.

    Heuristics
    ----------
    * **Visium** — ``adata.uns["spatial"]`` is present and contains a dict with a
      ``"scalefactors"`` sub-key for at least one library.  When ``uns["spatial"]``
      is present but has no scalefactors, Visium is still assumed (the key is
      Visium-specific).
    * **Xenium** — ``adata.obsm["spatial"]`` present **without** ``uns["spatial"]``,
      OR Xenium-specific obs columns (``cell_id``, ``transcript_counts``, etc.) are
      present.
    * **None** — no spatial evidence found.

    Parameters
    ----------
    adata:
        AnnData object to inspect.

    Returns
    -------
    ``"visium"``, ``"xenium"``, or ``None``.
    """
    # ------------------------------------------------------------------ Visium
    if "spatial" in adata.uns:
        uns_spatial = adata.uns["spatial"]
        if isinstance(uns_spatial, dict) and uns_spatial:
            for lib_data in uns_spatial.values():
                if isinstance(lib_data, dict) and "scalefactors" in lib_data:
                    log.debug("Detected Visium via uns['spatial'][...]['scalefactors']")
                    return "visium"
        # uns["spatial"] exists even without scalefactors → Visium
        log.debug("Detected Visium via uns['spatial']")
        return "visium"

    # ------------------------------------------------------------------ Xenium
    _xenium_obs_cols = {"cell_id", "transcript_counts", "control_probe_counts", "nucleus_area"}
    if _xenium_obs_cols.intersection(set(adata.obs.columns)):
        log.debug("Detected Xenium via obs columns")
        return "xenium"

    if "spatial" in adata.obsm:
        # Spatial coords present but no uns["spatial"]:
        # large datasets → Xenium; small → treat as Visium-like
        if adata.n_obs > 10_000:
            log.debug("Detected Xenium via obsm['spatial'] and large cell count")
            return "xenium"
        log.debug("Detected Visium-like via obsm['spatial']")
        return "visium"

    return None


def run_spatial_clustering(adata, resolution: float = 1.0) -> str:
    """Compute spatial neighbours (squidpy) then run Leiden clustering.

    Requires ``squidpy`` and ``leidenalg``.  Install with:
    ``pip install celltype-agent[spatial]``.

    Parameters
    ----------
    adata:
        AnnData with spatial coordinates in ``obsm["spatial"]``.
    resolution:
        Leiden resolution parameter.

    Returns
    -------
    str
        Key in ``adata.obs`` where cluster labels were written
        (``"spatial_leiden"``).
    """
    try:
        import squidpy as sq  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "squidpy is required for spatial clustering. "
            "Install it with: pip install celltype-agent[spatial]"
        ) from exc

    import scanpy as sc  # noqa: PLC0415

    obs_key = "spatial_leiden"

    log.info("Computing spatial neighbours…")
    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)

    # Point scanpy's Leiden to the spatial connectivity graph
    adata.uns["neighbors"] = {
        "connectivities_key": "spatial_connectivities",
        "distances_key": "spatial_distances",
    }

    log.info("Running Leiden clustering (resolution=%.2f)…", resolution)
    sc.tl.leiden(adata, resolution=resolution, key_added=obs_key)

    n_clusters = adata.obs[obs_key].nunique()
    log.info("Found %d spatial clusters.", n_clusters)
    return obs_key
