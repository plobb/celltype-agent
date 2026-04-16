# celltype-agent

**Claude-powered cell type annotation for single-cell and spatial genomics**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-50%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

celltype-agent automates cell type annotation across three 10x Genomics data modalities.
For **scRNA-seq** (Chromium), it extracts top differentially-expressed marker genes per cluster and sends them to Claude for structured annotation backed by two curated marker gene databases.
For **Visium** spatial data, it runs LDA deconvolution (with automatic K selection via held-out perplexity) to identify gene programs, which Claude annotates with a richer category system that distinguishes cell types from cell states, tissue programs, and technical artifacts.
For **Xenium** single-molecule data, it computes spatially-aware Leiden clusters via Squidpy, then feeds those clusters through the standard marker-gene annotation workflow.
All three modalities return Pydantic-validated structured output with confidence scores, reasoning, and database concordance evidence.

---

## Why it's interesting

**Hypothesise-then-verify agentic loop.** Rather than asking Claude to annotate in a single prompt, the agent follows a three-step workflow per cluster: form a hypothesis from marker genes, call `search_by_celltype` to verify against PanglaoDB + CellMarker, and fall back to `search_by_gene` on individual markers when fewer than 30% of the cluster's genes appear in the database results. This catches incorrect guesses that a single-shot prompt would accept.

**Dual-source database grounding.** PanglaoDB (8,286 marker-gene associations, March 2020) and CellMarker 2.0 (96,075 entries) are queried locally on every annotation — no network calls. Claude sees the raw gene lists from both databases and can form its own opinion about disagreements.

**Topic model for spatial data.** Visium spots are mixtures of cell types; assigning one label per spot ignores that. LDA deconvolution surfaces latent gene programs, and the `category` enum (`cell_type` / `cell_state` / `tissue_program` / `technical` / `ambiguous`) lets Claude express that not every topic is a lineage — some are fibrosis programs, ribosomal artifacts, or activation states overlaid on multiple cell types.

---

## Example output

PBMC3k dataset, 8 Leiden clusters:

```
                    Cell Type Annotations
 Cluster  Cell Type              Conf.  Key Markers              DB Support
 ───────  ─────────────────────  ─────  ───────────────────────  ─────────────────────────
 0        CD4+ T cell            0.94   CD3D, CD3E, IL7R, CD4    8/10 matched — PanglaoDB + CellMarker confirmed
 1        CD14+ Monocyte         0.91   LYZ, CD14, CST3, MS4A7   7/10 matched — both databases agreed
 2        B cell                 0.96   MS4A1, CD79A, CD79B      9/10 matched — strong concordance
 3        CD8+ T cell            0.89   CD8A, CD8B, GZMK, CCL5   8/10 matched — CellMarker confirmed
 4        NK cell                0.82   GNLY, NKG7, PRF1         6/10 matched — PanglaoDB confirmed
 5        CD16+ Monocyte         0.85   FCGR3A, MS4A7, CX3CR1   7/10 matched — both databases agreed
 6        Dendritic cell         0.78   FCER1A, HLA-DQA1         5/10 matched — partial concordance
 7        Megakaryocyte          0.93   PPBP, PF4, GP1BB         9/10 matched — strong concordance
```

---

## Architecture

```
User input (.h5ad)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  core.py: annotate() / annotate_spatial()                │
│                                                          │
│  ┌─────────────┐   ┌──────────────────┐                  │
│  │ markers.py  │   │ deconvolution.py │                  │
│  │ extract_    │   │ run_lda()        │                  │
│  │ markers()   │   │ auto-K (perp.)   │                  │
│  └──────┬──────┘   └────────┬─────────┘                  │
│         │  cluster→genes    │  topic→(gene,weight)       │
│         └──────────┬────────┘                            │
│                    ▼                                      │
│  ┌─────────────────────────────────────┐                 │
│  │  agent.py: annotate_clusters() /    │                 │
│  │            annotate_topics()        │                 │
│  │                                     │                 │
│  │  Claude claude-opus-4-6             │                 │
│  │  + adaptive thinking                │                 │
│  │  + streaming                        │                 │
│  │                                     │                 │
│  │  Tool calls:                        │                 │
│  │  ┌──────────────────────────────┐   │                 │
│  │  │ search_by_celltype(ct,sp)    │◄──┤ local dispatch  │
│  │  │ search_by_gene(gene,sp)      │◄──┤ no network      │
│  │  │ record_cell_type / topic     │   │                 │
│  │  └──────────────────────────────┘   │                 │
│  │           │                         │                 │
│  │           ▼                         │                 │
│  │  knowledge.py                       │                 │
│  │  PanglaoDB (8,286 entries)          │                 │
│  │  CellMarker 2.0 (96,075 entries)    │                 │
│  └─────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────┘
       │
       ▼
AnnotationResult / DeconvolutionResult  (Pydantic)
       │
       ├── to_labels()       dict[cluster→type]
       ├── to_dataframe()    pandas DataFrame
       ├── to_csv()          CSV file
       ├── to_narrative()    LLM-generated methods prose
       ├── to_methods()      citable methods paragraph
       └── _repr_html_()     Jupyter notebook table
```

---

## Quick start

```bash
git clone https://github.com/your-username/celltype-agent
cd celltype-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[scanpy]"

export ANTHROPIC_API_KEY=sk-ant-...
celltype-agent annotate pbmc.h5ad --species human --tissue PBMC
```

---

## Features

### scRNA-seq (Chromium)
- Works with any AnnData object; reuses pre-computed `rank_genes_groups` if present
- Wilcoxon rank-sum marker extraction (configurable method)
- Hypothesise-then-verify loop with up to 30 agent turns
- Dual-database grounding: PanglaoDB + CellMarker 2.0
- `database_markers_matched / total` for per-cluster concordance tracking

### Visium spatial
- LDA deconvolution on HVG-filtered count matrix
- Auto-K selection via held-out perplexity (range configurable)
- Five-category topic system: `cell_type`, `cell_state`, `tissue_program`, `technical`, `ambiguous`
- Per-spot topic proportions stored in `DeconvolutionResult.spot_topic_proportions`
- Sparse and dense input, negative-value safe (clips before LDA)

### Xenium single-molecule
- Auto-detects platform from AnnData metadata (obs columns, `uns["spatial"]` presence, cell count)
- Squidpy spatial neighbourhood graph → Leiden clustering
- Falls through to standard marker-gene annotation

### Shared
- Adaptive thinking on all Claude API calls
- Streaming to avoid HTTP timeouts on large datasets
- Progress callback API for embedding in notebooks or pipelines
- Rich CLI tables with colour-coded confidence
- `to_narrative()` / `to_methods()` for copy-paste into papers
- Jupyter `_repr_html_()` display

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# Core + scRNA-seq support
pip install -e ".[scanpy]"

# Add spatial (LDA + Squidpy)
pip install -e ".[scanpy,spatial]"

# Everything including dev tools
pip install -e ".[all]"
```

Requires `ANTHROPIC_API_KEY` in the environment or passed as `api_key=` at call time.

---

## Python API

### scRNA-seq

```python
import scanpy as sc
from celltype_agent import annotate

adata = sc.datasets.pbmc3k_processed()
result = annotate(adata, species="human", tissue="PBMC")

result.to_labels()          # {"0": "CD4+ T cell", "1": "B cell", ...}
result.to_dataframe()       # pandas DataFrame with confidence + DB support columns
result.to_methods()         # citable methods paragraph
result.to_narrative()       # LLM-generated results prose (makes an API call)
```

`annotate()` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adata` | — | AnnData object |
| `species` | `"human"` | `"human"` or `"mouse"` |
| `tissue` | `None` | Context hint, e.g. `"PBMC"`, `"lung"` |
| `cluster_key` | `"leiden"` | `adata.obs` column with cluster labels |
| `n_markers` | `10` | Top markers per cluster |
| `method` | `"wilcoxon"` | DE method if recomputing markers |
| `model` | `"claude-opus-4-6"` | Claude model |
| `add_to_obs` | `True` | Write labels to `adata.obs[obs_key]` |
| `obs_key` | `"cell_type"` | Target obs column |
| `progress_callback` | `None` | `Callable[[str], None]` for progress messages |

### Spatial

```python
from celltype_agent import annotate_spatial

# Visium → DeconvolutionResult
result = annotate_spatial(adata, species="human", tissue="liver")
result.topics            # list[TopicAnnotation]
result.spot_topic_proportions  # list[list[float]], shape (n_spots, n_topics)
result.dominant_topics() # list[int], dominant topic index per spot

# Xenium → AnnotationResult (same interface as annotate())
result = annotate_spatial(xenium_adata, mode="xenium", species="human")
```

---

## CLI reference

```bash
celltype-agent annotate <file.h5ad> [OPTIONS]
celltype-agent spatial  <file.h5ad> [OPTIONS]
```

### `annotate` subcommand

| Flag | Default | Description |
|------|---------|-------------|
| `--species` / `-s` | `human` | `human` or `mouse` |
| `--tissue` / `-t` | — | Tissue context |
| `--cluster-key` / `-k` | `leiden` | obs column with clusters |
| `--n-markers` / `-n` | `10` | Top markers per cluster |
| `--output` / `-o` | — | Save annotated `.h5ad` |
| `--obs-key` | `cell_type` | obs column for output labels |
| `--model` | `claude-opus-4-6` | Claude model |
| `--report` / `-r` | off | Print narrative + methods after table |
| `--verbose` / `-v` | off | Debug logging |

### `spatial` subcommand

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` / `-m` | `auto` | `auto`, `visium`, or `xenium` |
| `--k` | — | Fixed LDA topic count (Visium) |
| `--min-k` | `3` | Auto-K lower bound (Visium) |
| `--max-k` | `20` | Auto-K upper bound (Visium) |
| `--resolution` | `1.0` | Leiden resolution (Xenium) |
| `--cluster-key` / `-k2` | — | Pre-computed cluster obs column (Xenium) |
| `--n-markers` / `-n` | `10` | Top markers per cluster (Xenium) |
| `--report` / `-r` | off | Print narrative after table |

---

## Further reading

- [Architecture](docs/ARCHITECTURE.md) — system design, tool-use loop, database grounding, spatial pipeline
- [Design decisions](docs/DESIGN_DECISIONS.md) — ADR-style rationale for key choices
- [Roadmap](docs/ROADMAP.md) — known limitations and planned work
