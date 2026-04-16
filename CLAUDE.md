# celltype-agent — Project Context

## What this is

An automated cell type annotation tool for single-cell and spatial genomics (10x Chromium, Visium, Xenium). Given an AnnData object with cluster labels, it:

1. Extracts top differentially-expressed marker genes per cluster (`markers.py`)
2. Sends them to Claude via a tool-use agentic loop (`agent.py`)
3. Verifies hypotheses against PanglaoDB + CellMarker databases (`knowledge.py`)
4. Returns structured Pydantic objects and optionally writes labels back to `adata.obs` (`core.py`)

For spatial data, `annotate_spatial()` in `core.py` auto-detects the platform and routes to either LDA deconvolution (Visium) or Squidpy spatial clustering (Xenium).

## Repo layout

```
src/celltype_agent/
    __init__.py        Public API: annotate(), annotate_spatial(), result models
    core.py            annotate() and annotate_spatial() — main entry points
    agent.py           Claude API integration: tool-use loops, tool definitions, system prompts
    markers.py         extract_markers(), format_markers_for_prompt()
    knowledge.py       PanglaoDB + CellMarker loading, search_by_celltype(), search_by_gene()
    deconvolution.py   run_lda(), _auto_select_k(), format_topics_for_prompt()
    spatial.py         detect_spatial(), run_spatial_clustering() (Squidpy/Leiden)
    models.py          Pydantic models: CellTypeAnnotation, AnnotationResult, TopicAnnotation, DeconvolutionResult

cli/
    main.py            Typer CLI: `celltype-agent annotate` and `celltype-agent spatial` subcommands

tests/
    test_markers.py        format helpers, sort keys
    test_knowledge.py      database lookups (hits real data files)
    test_models.py         Pydantic validation, HTML repr, to_dataframe, to_csv
    test_narrative.py      to_methods() (no API), to_narrative() (mocked Anthropic)
    test_deconvolution.py  run_lda() with synthetic AnnData, auto-K, format_topics_for_prompt
    test_spatial.py        detect_spatial() heuristics, run_spatial_clustering (skipped without squidpy)

data/
    PanglaoDB_markers_27_Mar_2020.tsv
    Cell_marker_All.xlsx

docs/
    ARCHITECTURE.md     System design, tool-use loop, spatial pipeline differences
    DESIGN_DECISIONS.md ADR-style rationale for key choices
    ROADMAP.md          v1.0 path, limitations, production/clinical considerations
```

## Key design patterns

**Hypothesise-then-verify loop** (`agent.py`): Claude follows a 3-step workflow per cluster — form hypothesis → `search_by_celltype` → `search_by_gene` if concordance < 30% → `record_cell_type`. Up to 30 turns; nudge sent to Claude if it stops early with missing clusters.

**Local tool dispatch**: knowledge tools execute in-process. Only tool schemas go to the API. `_dispatch_tool()` in `agent.py` routes calls to `knowledge.py`.

**Adaptive thinking + streaming**: all Claude API calls use `thinking={"type": "adaptive"}` and `.stream()` + `get_final_message()`. This avoids HTTP timeouts on large datasets.

**Lazy imports**: `markers.py`, `deconvolution.py`, `spatial.py`, and `core.py` all guard heavy imports (`scanpy`, `sklearn`, `squidpy`) behind `try/except ImportError`. The library loads without optional deps installed.

**Pydantic boundaries**: `CellTypeAnnotation(**block.input)` validates Claude's tool call input at the entry point. Validation errors are caught and sent back as `is_error: true` tool results so Claude can retry.

**Dual database**: PanglaoDB (8,286 entries) and CellMarker 2.0 (96,075 entries) are loaded once and cached as module-level DataFrames. Fuzzy cell-type matching via exact → substring → word-overlap tiers.

**Category enum for topics**: `TopicAnnotation.category` is a `Literal["cell_type", "cell_state", "tissue_program", "technical", "ambiguous"]`. Forces Claude to classify spatial topics beyond the cell-type / not-cell-type binary.

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,scanpy,spatial]"
export ANTHROPIC_API_KEY=sk-ant-...
pytest
```

Run a quick sanity check (no API call):

```bash
pytest tests/test_markers.py tests/test_models.py tests/test_deconvolution.py -v
```

Run knowledge tests (requires data files to be present):

```bash
pytest tests/test_knowledge.py -v
```

## Running the demo

```bash
# scRNA-seq
celltype-agent annotate pbmc3k.h5ad --species human --tissue PBMC --report

# Spatial (Visium)
celltype-agent spatial visium_test.h5ad --species human --tissue liver --report
```

## CLI usage

```bash
# scRNA-seq
celltype-agent annotate data/pbmc.h5ad \
    --species human --tissue PBMC \
    --output pbmc_annotated.h5ad --report

# Spatial (auto-detects Visium vs Xenium)
celltype-agent spatial data/visium.h5ad \
    --species human --tissue liver \
    --max-k 15 --report
```

## Python API

```python
import scanpy as sc
from celltype_agent import annotate, annotate_spatial

# scRNA-seq
adata = sc.datasets.pbmc3k_processed()
result = annotate(adata, species="human", tissue="PBMC")
print(result.to_labels())      # {"0": "CD4+ T cell", ...}
print(result.to_methods())     # citable methods paragraph
result.to_csv("annotations.csv")

# Spatial
deconv = annotate_spatial(visium_adata, species="human", tissue="liver")
print(deconv.topics)           # list[TopicAnnotation]
print(deconv.dominant_topics()) # dominant topic index per spot
```

## Phase history

- **Phase 1**: Core scRNA-seq pipeline — markers.py, agent.py (tool-use loop), models.py, core.py, CLI
- **Phase 2**: Database grounding — knowledge.py (PanglaoDB + CellMarker), fuzzy matching, dual-source tool calls
- **Phase 3**: Spatial pipeline — deconvolution.py (LDA + auto-K), spatial.py (detect + Squidpy clustering), core.py extended to `annotate_spatial()`
- **Phase 4A**: Rich output — `to_narrative()`, `to_methods()`, `_repr_html_()`, `to_dataframe()`, `to_csv()`
- **Phase 4B**: Narrative tests — test_narrative.py covering to_methods() (unit) and to_narrative() (mocked Anthropic)
- **Phase 5**: Documentation overhaul — this CLAUDE.md, README.md, docs/
