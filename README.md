# celltype-agent

Automated cell type annotation for single-cell and spatial genomics (10x data) powered by [Claude](https://www.anthropic.com/claude).

## Features

- Works with any **AnnData** object (Scanpy, Squidpy, etc.)
- Handles pre-computed or fresh `rank_genes_groups` marker gene results
- Returns structured `CellTypeAnnotation` objects with confidence scores and reasoning
- Species-aware: human and mouse
- Tissue-aware when tissue context is provided
- CLI and Python API

## Install

```bash
pip install -e ".[scanpy]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quick start

### Python

```python
import scanpy as sc
from celltype_agent import annotate

adata = sc.datasets.pbmc3k_processed()

result = annotate(adata, species="human", tissue="PBMC")

# Dict of cluster_id -> predicted cell type
print(result.to_labels())

# Detailed annotations
for ann in result.annotations:
    print(f"Cluster {ann.cluster_id}: {ann.predicted_type} (conf={ann.confidence:.2f})")
    print(f"  Markers: {', '.join(ann.markers_used)}")
    print(f"  Reasoning: {ann.reasoning}")
```

### CLI

```bash
celltype-agent annotate pbmc.h5ad \
    --species human \
    --tissue PBMC \
    --output pbmc_annotated.h5ad
```

## API reference

### `annotate(adata, *, species, tissue, cluster_key, n_markers, ...)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adata` | — | AnnData object |
| `species` | `"human"` | `"human"` or `"mouse"` |
| `tissue` | `None` | Tissue context (e.g. `"PBMC"`, `"lung"`) |
| `cluster_key` | `"leiden"` | `adata.obs` column with cluster labels |
| `n_markers` | `10` | Top markers to use per cluster |
| `method` | `"wilcoxon"` | DE method if recomputing markers |
| `add_to_obs` | `True` | Write labels to `adata.obs[obs_key]` |
| `obs_key` | `"cell_type"` | Column name for labels |
| `model` | `"claude-opus-4-6"` | Claude model |

## Development

```bash
pip install -e ".[dev,scanpy]"
pytest
```
