# celltype-agent — Project Context

## What this is

An automated cell type annotation tool for single-cell and spatial genomics (10x Chromium, Visium, etc.). Given an **AnnData** object with cluster labels, it:

1. Extracts top differentially-expressed marker genes per cluster (`markers.py`)
2. Sends them to **Claude claude-opus-4-6** via tool-use (`agent.py`)
3. Returns structured `CellTypeAnnotation` objects and optionally writes labels back to `adata.obs` (`core.py`)

## Repo layout

```
src/celltype_agent/
    __init__.py      # Public API: annotate(), AnnotationResult, CellTypeAnnotation
    markers.py       # extract_markers() + format helpers
    models.py        # Pydantic models
    agent.py         # Claude API integration (tool-use loop)
    core.py          # annotate() — main entry point
cli/
    main.py          # Typer CLI — `celltype-agent annotate <file.h5ad>`
notebooks/           # Jupyter demos (to be added)
tests/               # pytest unit tests (no API calls)
```

## Key design decisions

- **Tool-use for structured output** — Claude calls `record_cell_type(cluster_id, predicted_type, confidence, markers_used, reasoning)` once per cluster rather than free-text output. This gives us typed Pydantic objects directly.
- **Adaptive thinking** — all Claude API calls use `thinking={"type": "adaptive"}` to let Claude reason before calling tools.
- **Streaming** — uses `.stream()` + `get_final_message()` to avoid HTTP timeouts on large datasets.
- **Lazy scanpy import** — `markers.py` only imports `scanpy` when `rank_genes_groups` needs to be recomputed; the library loads without scanpy installed.
- **Species + tissue context** — passed in the system prompt to steer Claude toward tissue-appropriate subtypes (e.g. "alveolar macrophage" in lung vs "Kupffer cell" in liver).

## Dev setup

```bash
pip install -e ".[dev,scanpy]"
export ANTHROPIC_API_KEY=sk-ant-...
pytest
```

## CLI usage

```bash
celltype-agent annotate data/pbmc.h5ad --species human --tissue PBMC --output pbmc_annotated.h5ad
```

## Python usage

```python
import scanpy as sc
from celltype_agent import annotate

adata = sc.datasets.pbmc3k_processed()
result = annotate(adata, species="human", tissue="PBMC")
print(result.to_labels())
# {'0': 'CD4+ T cell', '1': 'B cell', '2': 'CD14+ Monocyte', ...}
```

## Species support

- `"human"` and `"mouse"` — passed as context in the prompt
- `tissue` is optional but significantly improves specificity

## Phase 2 ideas (not yet implemented)

- Spatial context: pass neighbourhood cell type frequencies for spatially-aware annotation
- Batch mode: annotate multiple AnnData objects via the Anthropic Batches API
- Confidence calibration: compare predictions against a held-out reference dataset
- Marker DB: embed a curated marker gene database for retrieval-augmented annotation
