"""High-level annotate() entry point."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Union

from .agent import annotate_clusters, annotate_topics
from .markers import extract_markers
from .models import AnnotationResult, CellTypeAnnotation, DeconvolutionResult

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
        n_markers=n_markers,
        model_used=model,
    )

    # --- 3. Optionally write back to adata -----------------------------------
    if add_to_obs:
        label_map = result.to_labels()
        adata.obs[obs_key] = adata.obs[cluster_key].astype(str).map(label_map)
        log.info("Wrote cell type labels to adata.obs['%s'].", obs_key)

    return result


def annotate_spatial(
    adata,
    *,
    species: str = "human",
    tissue: Optional[str] = None,
    mode: str = "auto",
    k: Optional[int] = None,
    cluster_key: Optional[str] = None,
    n_markers: int = 10,
    resolution: float = 1.0,
    max_k: int = 20,
    min_k: int = 3,
    n_top_genes: int = 2000,
    api_key: Optional[str] = None,
    model: str = "claude-opus-4-6",
    random_state: int = 42,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Union[AnnotationResult, DeconvolutionResult]:
    """Annotate a spatial transcriptomics dataset.

    Auto-detects the platform (Visium or Xenium) from the AnnData metadata.

    * **Visium** — runs LDA deconvolution to extract gene programs (topics),
      then annotates each topic with Claude, returning a
      :class:`~celltype_agent.models.DeconvolutionResult`.
    * **Xenium** — computes spatial Leiden clusters (via squidpy), then
      uses the standard marker-gene annotation workflow, returning an
      :class:`~celltype_agent.models.AnnotationResult`.

    Parameters
    ----------
    adata:
        AnnData with spatial coordinates.
    species:
        ``'human'`` or ``'mouse'``.
    tissue:
        Optional tissue context.
    mode:
        ``'auto'`` (detect from data), ``'visium'``, or ``'xenium'``.
    k:
        Number of LDA topics (Visium only).  Auto-selected when ``None``.
    cluster_key:
        Pre-computed obs column with cluster labels (Xenium only).  When
        ``None``, :func:`~celltype_agent.spatial.run_spatial_clustering`
        is called automatically.
    n_markers:
        Top markers per cluster for Xenium annotation.
    resolution:
        Leiden resolution for Xenium spatial clustering.
    max_k / min_k:
        Bounds for Visium auto-K search.
    n_top_genes:
        Highly-variable genes to retain before LDA.
    api_key:
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    model:
        Claude model identifier.
    random_state:
        Random seed for LDA / clustering.
    progress_callback:
        Optional callable that receives progress messages as strings.

    Returns
    -------
    :class:`~celltype_agent.models.DeconvolutionResult` for Visium data or
    :class:`~celltype_agent.models.AnnotationResult` for Xenium data.
    """
    from .deconvolution import run_lda  # noqa: PLC0415
    from .spatial import detect_spatial, run_spatial_clustering  # noqa: PLC0415

    def _progress(msg: str) -> None:
        log.info(msg)
        if progress_callback is not None:
            progress_callback(msg)

    species = species.lower().strip()
    if species not in {"human", "mouse"}:
        raise ValueError(f"species must be 'human' or 'mouse', got '{species}'")

    # --- Resolve mode ----------------------------------------------------------
    if mode == "auto":
        detected = detect_spatial(adata)
        mode = detected or "visium"
        _progress(f"Auto-detected platform: {mode}.")

    # ===========================================================================
    # Visium: LDA deconvolution → topic annotation
    # ===========================================================================
    if mode == "visium":
        _progress(f"Running LDA deconvolution (max_k={max_k}, min_k={min_k})…")
        deconv = run_lda(
            adata,
            k=k,
            max_k=max_k,
            min_k=min_k,
            n_top_genes=n_top_genes,
            random_state=random_state,
        )
        n_topics = deconv["n_topics"]
        _progress(
            f"LDA complete — {n_topics} topics. "
            f"Sending to {model} for annotation…"
        )

        topic_annotations = annotate_topics(
            deconv["topic_genes"],
            species=species,
            tissue=tissue,
            api_key=api_key,
            model=model,
        )
        _progress(f"Topic annotation complete — {len(topic_annotations)} topic(s) annotated.")

        return DeconvolutionResult(
            topics=topic_annotations,
            n_topics=n_topics,
            spot_topic_proportions=deconv["spot_topic_proportions"].tolist(),
            coherence_scores={str(k_): v for k_, v in deconv["coherence_scores"].items()},
            species=species,
            tissue=tissue,
            model_used=model,
        )

    # ===========================================================================
    # Xenium: spatial clustering → marker extraction → cluster annotation
    # ===========================================================================
    if mode == "xenium":
        obs_key = cluster_key
        if obs_key is None:
            _progress("Computing spatial clusters (squidpy + Leiden)…")
            obs_key = run_spatial_clustering(adata, resolution=resolution)

        _progress(
            f"Extracting top {n_markers} markers per cluster "
            f"(key='{obs_key}')…"
        )
        markers = extract_markers(adata, cluster_key=obs_key, n_markers=n_markers)
        _progress(
            f"Found {len(markers)} clusters. Sending to {model} for annotation…"
        )

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
            n_markers=n_markers,
            model_used=model,
        )
        return result

    raise ValueError(f"mode must be 'auto', 'visium', or 'xenium'; got '{mode}'")
