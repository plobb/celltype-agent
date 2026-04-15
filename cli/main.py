"""CLI entry point for celltype-agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="celltype-agent",
    help="Automated cell type annotation powered by Claude.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _root() -> None:
    """celltype-agent — Claude-powered single-cell annotation."""


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def annotate(
    h5ad: Path = typer.Argument(..., help="Path to AnnData (.h5ad) file."),
    species: str = typer.Option("human", "--species", "-s", help="'human' or 'mouse'."),
    tissue: Optional[str] = typer.Option(
        None, "--tissue", "-t", help="Tissue context, e.g. 'PBMC', 'lung'."
    ),
    cluster_key: str = typer.Option(
        "leiden", "--cluster-key", "-k", help="obs column with cluster labels."
    ),
    n_markers: int = typer.Option(10, "--n-markers", "-n", help="Top markers per cluster."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save annotated AnnData to this .h5ad path."
    ),
    obs_key: str = typer.Option(
        "cell_type", "--obs-key", help="adata.obs column for cell type labels."
    ),
    model: str = typer.Option("claude-opus-4-6", "--model", help="Claude model to use."),
    report: bool = typer.Option(False, "--report", "-r", help="Print narrative and methods sections after the table."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Annotate cell clusters in an AnnData file and print results."""
    _setup_logging(verbose)

    # Lazy imports so --help is fast
    try:
        import anndata  # noqa: F401
    except ImportError:
        console.print("[red]anndata is not installed.[/red] Run: pip install anndata")
        raise typer.Exit(1)

    from celltype_agent import annotate as _annotate

    console.print(f"[bold]Loading[/bold] {h5ad} …")
    import anndata as ad

    adata = ad.read_h5ad(h5ad)
    console.print(
        f"[green]Loaded[/green] {adata.n_obs:,} cells × {adata.n_vars:,} genes, "
        f"{adata.obs[cluster_key].nunique()} clusters."
    )

    with console.status("Annotating with Claude …"):
        result = _annotate(
            adata,
            species=species,
            tissue=tissue,
            cluster_key=cluster_key,
            n_markers=n_markers,
            model=model,
            add_to_obs=True,
            obs_key=obs_key,
        )

    # Pretty-print results table
    table = Table(title="Cell Type Annotations", show_header=True, header_style="bold cyan")
    table.add_column("Cluster", style="dim")
    table.add_column("Cell Type", style="bold")
    table.add_column("Conf.", justify="right")
    table.add_column("Key Markers")
    table.add_column("DB Support")

    for ann in sorted(result.annotations, key=lambda a: _sort_key(a.cluster_id)):
        db_support = ann.database_support or ""
        if ann.database_markers_matched is not None and ann.database_markers_total:
            db_support = (
                f"{ann.database_markers_matched}/{ann.database_markers_total} matched"
                + (f" — {db_support}" if db_support else "")
            )
        table.add_row(
            ann.cluster_id,
            ann.predicted_type,
            f"{ann.confidence:.2f}",
            ", ".join(ann.markers_used[:4]),
            db_support,
        )

    console.print(table)

    if report:
        with console.status("Generating narrative summary …"):
            narrative = result.to_narrative()
        console.print(Panel(narrative, title="[bold cyan]Narrative Summary[/bold cyan]", border_style="cyan"))
        console.print(Panel(result.to_methods(), title="[bold cyan]Methods[/bold cyan]", border_style="cyan"))

    if output:
        adata.write_h5ad(output)
        console.print(f"[green]Saved[/green] annotated AnnData → {output}")


@app.command()
def spatial(
    h5ad: Path = typer.Argument(..., help="Path to AnnData (.h5ad) file."),
    species: str = typer.Option("human", "--species", "-s", help="'human' or 'mouse'."),
    tissue: Optional[str] = typer.Option(
        None, "--tissue", "-t", help="Tissue context, e.g. 'brain', 'liver'."
    ),
    mode: str = typer.Option(
        "auto", "--mode", "-m",
        help="Platform: 'auto' (detect), 'visium', or 'xenium'.",
    ),
    k: Optional[int] = typer.Option(None, "--k", help="Fixed number of LDA topics (Visium)."),
    min_k: int = typer.Option(3, "--min-k", help="Minimum K for auto-K search (Visium)."),
    max_k: int = typer.Option(20, "--max-k", help="Maximum K for auto-K search (Visium)."),
    resolution: float = typer.Option(1.0, "--resolution", help="Leiden resolution (Xenium)."),
    cluster_key: Optional[str] = typer.Option(
        None, "--cluster-key", "-k2",
        help="Pre-computed obs column with cluster labels (Xenium only).",
    ),
    n_markers: int = typer.Option(10, "--n-markers", "-n", help="Top markers per cluster (Xenium)."),
    model: str = typer.Option("claude-opus-4-6", "--model", help="Claude model to use."),
    report: bool = typer.Option(False, "--report", "-r", help="Print narrative summary after the table."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Annotate a spatial transcriptomics AnnData file (Visium or Xenium)."""
    _setup_logging(verbose)

    try:
        import anndata  # noqa: F401
    except ImportError:
        console.print("[red]anndata is not installed.[/red] Run: pip install anndata")
        raise typer.Exit(1)

    from celltype_agent import annotate_spatial
    from celltype_agent.models import DeconvolutionResult

    console.print(f"[bold]Loading[/bold] {h5ad} …")
    import anndata as ad

    adata = ad.read_h5ad(h5ad)
    console.print(
        f"[green]Loaded[/green] {adata.n_obs:,} cells/spots × {adata.n_vars:,} genes."
    )

    with console.status("Annotating with Claude …"):
        result = annotate_spatial(
            adata,
            species=species,
            tissue=tissue,
            mode=mode,
            k=k,
            cluster_key=cluster_key,
            n_markers=n_markers,
            resolution=resolution,
            max_k=max_k,
            min_k=min_k,
            model=model,
        )

    if isinstance(result, DeconvolutionResult):
        _print_deconvolution_table(result)
    else:
        _print_annotation_table(result)

    if report:
        with console.status("Generating narrative summary …"):
            narrative = result.to_narrative()
        console.print(Panel(narrative, title="[bold cyan]Narrative Summary[/bold cyan]", border_style="cyan"))


def _print_annotation_table(result) -> None:
    """Print a Rich table for AnnotationResult (Xenium clusters)."""
    table = Table(
        title="Spatial Cell Type Annotations",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Cluster", style="dim")
    table.add_column("Cell Type", style="bold")
    table.add_column("Conf.", justify="right")
    table.add_column("Key Markers")

    for ann in sorted(result.annotations, key=lambda a: _sort_key(a.cluster_id)):
        table.add_row(
            ann.cluster_id,
            ann.predicted_type,
            f"{ann.confidence:.2f}",
            ", ".join(ann.markers_used[:4]),
        )
    console.print(table)


def _print_deconvolution_table(result) -> None:
    """Print a Rich table for DeconvolutionResult (Visium topics)."""
    import numpy as np

    table = Table(
        title=f"LDA Topic Annotations ({result.n_topics} topics)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Topic", style="dim", justify="right")
    table.add_column("Annotation", style="bold")
    table.add_column("Category")
    table.add_column("Conf.", justify="right")
    table.add_column("Key Genes")

    _category_style = {
        "cell_type": "blue",
        "cell_state": "magenta",
        "tissue_program": "green",
        "technical": "dim",
        "ambiguous": "yellow",
    }

    for t in sorted(result.topics, key=lambda x: x.topic_id):
        cat_style = _category_style.get(t.category, "")
        table.add_row(
            str(t.topic_id),
            t.annotation,
            f"[{cat_style}]{t.category}[/{cat_style}]",
            f"{t.confidence:.2f}",
            ", ".join(t.key_genes[:4]),
        )
    console.print(table)

    # Per-spot composition summary: dominant topic frequency
    proportions = np.array(result.spot_topic_proportions)
    dominant = proportions.argmax(axis=1)
    console.print("\n[bold]Dominant topic per spot:[/bold]")
    summary_table = Table(show_header=True, header_style="bold")
    summary_table.add_column("Topic")
    summary_table.add_column("Annotation")
    summary_table.add_column("# Spots (dominant)", justify="right")
    summary_table.add_column("% Spots", justify="right")
    n_spots = len(dominant)
    topic_map = {t.topic_id: t for t in result.topics}
    for t_idx in range(result.n_topics):
        n = int((dominant == t_idx).sum())
        ann_label = topic_map[t_idx].annotation if t_idx in topic_map else "—"
        summary_table.add_row(
            str(t_idx),
            ann_label,
            str(n),
            f"{100 * n / n_spots:.1f}%",
        )
    console.print(summary_table)


def _sort_key(label: str):
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)


if __name__ == "__main__":
    app()
