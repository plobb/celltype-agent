"""Tests for spatial platform detection and clustering utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import anndata as ad

from celltype_agent.spatial import detect_spatial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bare_adata(n_obs: int = 20, n_vars: int = 30) -> ad.AnnData:
    """Minimal AnnData with no spatial metadata."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 100, size=(n_obs, n_vars)).astype(np.float32)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    return ad.AnnData(X=X, obs=obs, var=var)


def _visium_adata(with_scalefactors: bool = True) -> ad.AnnData:
    """Minimal mock Visium AnnData."""
    adata = _bare_adata(n_obs=3000)
    if with_scalefactors:
        adata.uns["spatial"] = {
            "V1_Human_Lymph_Node": {
                "scalefactors": {
                    "spot_diameter_fullres": 89.4,
                    "tissue_hires_scalef": 0.2,
                },
                "images": {},
            }
        }
    else:
        # uns["spatial"] present but no scalefactors key
        adata.uns["spatial"] = {"some_library": {}}
    # Visium spots also typically have spatial coords
    adata.obsm["spatial"] = np.random.default_rng(1).uniform(0, 5000, size=(3000, 2))
    return adata


def _xenium_adata(large: bool = True) -> ad.AnnData:
    """Minimal mock Xenium AnnData."""
    n = 15_000 if large else 5_000
    adata = _bare_adata(n_obs=n)
    adata.obsm["spatial"] = np.random.default_rng(2).uniform(0, 10000, size=(n, 2))
    return adata


def _xenium_adata_with_obs_cols() -> ad.AnnData:
    """Xenium AnnData with characteristic obs columns."""
    adata = _bare_adata(n_obs=500)
    adata.obs["cell_id"] = [f"cid_{i}" for i in range(500)]
    adata.obs["transcript_counts"] = np.random.randint(10, 200, size=500)
    adata.obsm["spatial"] = np.random.default_rng(3).uniform(0, 1000, size=(500, 2))
    return adata


# ---------------------------------------------------------------------------
# detect_spatial
# ---------------------------------------------------------------------------


def test_detect_visium_with_scalefactors():
    adata = _visium_adata(with_scalefactors=True)
    assert detect_spatial(adata) == "visium"


def test_detect_visium_without_scalefactors():
    """uns['spatial'] present but no scalefactors → still Visium."""
    adata = _visium_adata(with_scalefactors=False)
    assert detect_spatial(adata) == "visium"


def test_detect_visium_uns_only():
    """Just uns['spatial'] with a non-dict value should still → visium."""
    adata = _bare_adata()
    adata.uns["spatial"] = {"lib": {"scalefactors": {}}}
    assert detect_spatial(adata) == "visium"


def test_detect_xenium_large_dataset():
    """Large cell count + obsm['spatial'] but no uns['spatial'] → xenium."""
    adata = _xenium_adata(large=True)
    assert detect_spatial(adata) == "xenium"


def test_detect_xenium_obs_columns():
    """Xenium-specific obs columns → xenium (regardless of cell count)."""
    adata = _xenium_adata_with_obs_cols()
    assert detect_spatial(adata) == "xenium"


def test_detect_small_spatial_no_uns():
    """Small dataset with obsm['spatial'] but no uns['spatial'] → visium-like."""
    adata = _bare_adata(n_obs=200)
    adata.obsm["spatial"] = np.random.default_rng(0).uniform(0, 1000, size=(200, 2))
    assert detect_spatial(adata) == "visium"


def test_detect_no_spatial():
    """Plain scRNA-seq AnnData with no spatial metadata → None."""
    adata = _bare_adata()
    assert detect_spatial(adata) is None


def test_detect_visium_takes_priority_over_large_cell_count():
    """uns['spatial'] should always → visium even with many cells."""
    adata = _visium_adata(with_scalefactors=True)
    # Inflate obs count: rebuild with more spots
    big_adata = _bare_adata(n_obs=50_000)
    big_adata.uns["spatial"] = adata.uns["spatial"]
    assert detect_spatial(big_adata) == "visium"


def test_detect_xenium_transcript_counts_col():
    adata = _bare_adata(n_obs=100)
    adata.obs["transcript_counts"] = np.random.randint(1, 100, 100)
    assert detect_spatial(adata) == "xenium"


# ---------------------------------------------------------------------------
# run_spatial_clustering (requires squidpy — skip if not installed)
# ---------------------------------------------------------------------------


squidpy = pytest.importorskip("squidpy", reason="squidpy not installed")


def _spatial_adata_for_clustering(n_obs: int = 200, n_vars: int = 50) -> ad.AnnData:
    """AnnData with spatial coords and PCA-like embedding for clustering."""
    rng = np.random.default_rng(42)
    X = rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = rng.uniform(0, 1000, size=(n_obs, 2))
    return adata


def test_run_spatial_clustering_returns_obs_key():
    from celltype_agent.spatial import run_spatial_clustering

    adata = _spatial_adata_for_clustering()
    obs_key = run_spatial_clustering(adata, resolution=0.5)
    assert obs_key == "spatial_leiden"
    assert obs_key in adata.obs.columns


def test_run_spatial_clustering_labels_are_strings():
    from celltype_agent.spatial import run_spatial_clustering

    adata = _spatial_adata_for_clustering()
    obs_key = run_spatial_clustering(adata, resolution=0.5)
    labels = adata.obs[obs_key]
    assert labels.dtype == object or pd.api.types.is_categorical_dtype(labels)


def test_run_spatial_clustering_at_least_one_cluster():
    from celltype_agent.spatial import run_spatial_clustering

    adata = _spatial_adata_for_clustering()
    obs_key = run_spatial_clustering(adata, resolution=0.5)
    n_clusters = adata.obs[obs_key].nunique()
    assert n_clusters >= 1


def test_run_spatial_clustering_labels_cover_all_cells():
    from celltype_agent.spatial import run_spatial_clustering

    adata = _spatial_adata_for_clustering()
    obs_key = run_spatial_clustering(adata, resolution=0.5)
    assert adata.obs[obs_key].notna().all()
