"""CLI entry point for celltype-agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

app = typer.Typer(
    name="celltype-agent",
    help="Automated cell type annotation powered by Claude.",
    add_completion=False,
)
console = Console()


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

    for ann in sorted(result.annotations, key=lambda a: _sort_key(a.cluster_id)):
        table.add_row(
            ann.cluster_id,
            ann.predicted_type,
            f"{ann.confidence:.2f}",
            ", ".join(ann.markers_used[:4]),
        )

    console.print(table)

    if output:
        adata.write_h5ad(output)
        console.print(f"[green]Saved[/green] annotated AnnData → {output}")


def _sort_key(label: str):
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)


if __name__ == "__main__":
    app()
