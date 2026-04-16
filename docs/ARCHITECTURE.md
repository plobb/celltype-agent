# Architecture

Technical deep-dive into celltype-agent's design and data flow.

---

## System overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Public API  (src/celltype_agent/__init__.py)                       │
│  annotate()  annotate_spatial()                                     │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │  core.py              │
          │  Orchestrates the     │
          │  scRNA / spatial      │
          │  pipelines            │
          └─┬──────────────────┬──┘
            │                  │
  ┌─────────▼──────┐  ┌────────▼─────────────┐
  │  markers.py    │  │  deconvolution.py     │
  │  extract_      │  │  run_lda()            │
  │  markers()     │  │  _auto_select_k()     │
  │                │  │  format_topics_for_   │
  │  format_       │  │  prompt()             │
  │  markers_for_  │  └────────┬──────────────┘
  │  prompt()      │           │
  └─────────┬──────┘  ┌────────▼──────────────┐
            │         │  spatial.py            │
            │         │  detect_spatial()      │
            │         │  run_spatial_          │
            │         │  clustering()          │
            │         └────────┬──────────────┘
            │                  │
          ┌─▼──────────────────▼──┐
          │  agent.py              │
          │  annotate_clusters()   │
          │  annotate_topics()     │
          │  Agentic tool-use loop │
          └─┬──────────────────┬──┘
            │                  │
  ┌─────────▼──────┐  ┌────────▼──────────────┐
  │  Claude API    │  │  knowledge.py          │
  │  (Anthropic    │  │  search_by_celltype()  │
  │  SDK)          │  │  search_by_gene()      │
  │  claude-opus   │  │  PanglaoDB (in-memory) │
  │  adaptive      │  │  CellMarker (in-memory)│
  │  thinking +    │  └───────────────────────┘
  │  streaming     │
  └────────────────┘
          │
  ┌───────▼────────────────────┐
  │  models.py                 │
  │  CellTypeAnnotation        │
  │  AnnotationResult          │
  │  TopicAnnotation           │
  │  DeconvolutionResult       │
  └────────────────────────────┘
```

---

## The hypothesise-then-verify agentic pattern

### What it is

The annotation loop does not ask Claude to answer in one shot. Instead, the system prompt instructs Claude to follow a three-step workflow for every cluster or topic:

1. **Hypothesise** — form an initial cell type guess from the marker gene list.
2. **Verify** — call `search_by_celltype` with that guess; inspect the returned marker lists from PanglaoDB and CellMarker.
3. **Explore alternatives** — if fewer than 30% of the cluster's own markers appear in the database result, call `search_by_gene` on 2–3 of the most distinctive markers to find what else they're associated with.
4. **Record** — call `record_cell_type` (or `record_topic`) with the final answer and a `database_support` summary.

### Why it beats single-shot annotation

A single-shot prompt gives Claude no way to catch its own errors. If Claude guesses "NK cell" for a cluster that's actually a cytotoxic T cell (CD8A+ but also NKG7+), there's no feedback mechanism to question that call. The tool-use loop forces an explicit verification step: if the database returns zero markers from the cluster's actual gene list, Claude knows the hypothesis is wrong and has a structured way to explore alternatives. This produces more conservative confidence scores and surfaced database disagreements rather than silent errors.

---

## Tool use architecture

### Tool definitions

Three tools are defined in `agent.py` as plain dicts conforming to the Anthropic tool schema:

| Tool | When called | Effect |
|------|-------------|--------|
| `search_by_celltype(cell_type, species)` | After forming a hypothesis | Returns `{panglaodb: [genes], cellmarker: [genes]}` |
| `search_by_gene(gene, species)` | When DB concordance < 30% | Returns `{panglaodb: [{cell_type, organ}], cellmarker: [{cell_type, tissue}]}` |
| `record_cell_type(cluster_id, predicted_type, confidence, markers_used, reasoning, ...)` | Final annotation per cluster | Triggers structured output capture |

For spatial, `record_cell_type` is replaced by `record_topic(topic_id, annotation, category, ...)`.

### Local tool dispatch

The `_dispatch_tool()` function in `agent.py` intercepts every tool call block from the assistant response and executes it locally — no MCP server, no subprocess. Only tool *schemas* are sent to the API. This means:
- No network round-trips for database lookups
- No serialization overhead beyond a JSON string for the result
- Full stack trace visibility when a tool call fails

### The agentic loop

```
messages = [user_message]
for turn in range(max_turns=30):
    response = client.messages.stream(...)
    
    for block in response.content:
        if block.name == "record_cell_type":
            annotations[cluster_id] = CellTypeAnnotation(**block.input)
        elif block.name in ("search_by_celltype", "search_by_gene"):
            result = _dispatch_tool(block.name, block.input)
            tool_results.append({"content": result, ...})
    
    messages.append(assistant_turn)
    messages.append(tool_results)
    
    if all clusters annotated: break
    if end_turn and missing clusters: nudge Claude
```

The loop continues until all clusters have a `record_cell_type` call, or `max_turns` (30) is exhausted. The nudge message explicitly lists missing cluster IDs to prevent Claude from assuming it's done. Partial results are not discarded; any annotations recorded before the turn limit are returned with a warning.

---

## Structured output via Pydantic

Claude's tool call inputs arrive as dicts. `CellTypeAnnotation(**block.input)` immediately validates the incoming data:
- `confidence` is rejected outside `[0, 1]` with a `ValidationError`
- `predicted_type` is stripped of leading/trailing whitespace via a `field_validator`
- Required fields (`cluster_id`, `predicted_type`, `confidence`, `markers_used`, `reasoning`) are enforced at parse time

If validation fails, the loop catches the exception, sends an `is_error: true` tool result back to Claude (with the error message), and lets Claude retry. This means malformed tool calls are self-healing rather than silently dropped.

`AnnotationResult` and `DeconvolutionResult` are similarly strict Pydantic models, which makes them directly serialisable to JSON, CSV, or dataframe without custom marshalling code.

---

## Database grounding strategy

### Databases

**PanglaoDB** (March 2020): 8,286 marker-gene associations across human and mouse. Encoded as a TSV with species flags `"Hs"`, `"Mm"`, or `"Mm Hs"` (both). Loaded once at first use and cached as a module-level DataFrame.

**CellMarker 2.0**: 96,075 deduplicated (species, cell_type, symbol) entries. Loaded from the bundled Excel file.

Both databases are loaded lazily via `_ensure_loaded()` — the first call incurs the I/O cost (roughly 0.5s), subsequent calls are in-memory lookups.

### Fuzzy cell type matching

`_match_cell_types()` applies a three-tier strategy:
1. Exact case-insensitive match
2. Substring match (either direction: "B cell" in "Memory B cell", or "NK" in "NK cell")
3. Word-overlap fallback: any word in the query appears in the database cell type name

This handles the common case where Claude uses a term like "CD4+ helper T cell" while the database has "CD4-positive T cell" or "T helper cell".

### Species normalisation

PanglaoDB uses `"Hs"` / `"Mm"` / `"Mm Hs"` codes. CellMarker uses `"Human"` / `"Mouse"`. Both are normalised to `"human"` / `"mouse"` / `"both"` at load time so the agent API accepts a single uniform `species` string.

---

## Spatial pipeline differences

### Visium: LDA deconvolution

Visium captures ~3,000–6,000 spots, each 55µm in diameter. At that resolution, each spot typically contains contributions from multiple cell types. Treating each spot as a single cell type (as you would in scRNA-seq clustering) discards this mixture information.

LDA decomposes the spot × gene count matrix into `K` latent topics where each topic is a probability distribution over genes, and each spot is a mixture of topics. The resulting topics often correspond to cell type-specific gene programs, but also capture cell states, tissue structural programs, and technical variation — which is why the `category` enum matters (see below).

**Auto-K selection**: fitting LDA for every K in `[min_k, max_k]` (default 3–20), holding out 10–20% of spots as a test set, and selecting the K with lowest perplexity on the held-out data. This is a standard model selection criterion for topic models; it avoids requiring the user to guess K upfront.

**Gene filtering**: HVG selection (`sc.pp.highly_variable_genes`, seurat_v3 flavor) before LDA reduces the feature space from ~30,000 to 2,000 genes. This removes noise genes that don't contribute to biological signal and makes LDA numerically faster and more stable.

### Xenium: Squidpy spatial clustering

Xenium data contains single-cell resolution transcriptomics (typically 10,000–100,000+ cells). Spatial clustering uses Squidpy to build a spatial neighbourhood graph (Delaunay triangulation on cell centroids) and then runs standard Leiden community detection on that graph. This is intentionally different from using a transcriptomic KNN graph: it groups cells that are spatially adjacent, which is relevant for identifying tissue compartments and cell niches.

Once spatial clusters are defined, the annotation follows the same marker-gene extraction → Claude agent loop as scRNA-seq.

### Platform detection

`detect_spatial()` uses a priority-ordered set of heuristics to infer platform:
1. `adata.uns["spatial"]` present → Visium (10x Visium always populates this key)
2. Xenium-specific obs columns (`cell_id`, `transcript_counts`, etc.) → Xenium
3. `adata.obsm["spatial"]` with >10,000 cells → Xenium
4. `adata.obsm["spatial"]` with ≤10,000 cells → Visium-like

The `uns["spatial"]` check takes absolute priority because it's definitive — Visium always writes it and Xenium never does.

---

## Category system for Visium topics

Forcing every LDA topic to be labelled as a "cell type" is wrong. In practice:
- Some topics capture activation states that span multiple cell types (e.g. an interferon response program appears in macrophages and epithelial cells simultaneously)
- Some topics are dominated by ribosomal genes (RPL*, RPS*) or mitochondrial genes (MT-*) — artefacts of technical variation, not biology
- Some topics represent tissue structural programs: ECM remodelling (COL*, FN1, VIM), angiogenesis (VWF, PECAM1), fibrosis

The `category` literal enum in `TopicAnnotation` captures this:

| Category | Meaning | Example |
|----------|---------|---------|
| `cell_type` | Specific cell lineage | "Hepatocyte", "CD8+ T cell" |
| `cell_state` | Activation/stress state overlaid on lineage | "Inflammatory macrophage", "Stress response" |
| `tissue_program` | Structural or signalling program | "ECM/fibrosis", "Angiogenesis" |
| `technical` | Sequencing or cell quality artefact | "Ribosomal", "Mitochondrial stress" |
| `ambiguous` | Mixed or unresolvable gene program | — |

The Claude prompt explicitly lists these categories with examples so Claude knows to assign `technical` to ribosomal topics and `tissue_program` to collagen-heavy topics rather than forcing them into lineage labels.

---

## Confidence derivation

Confidence is not derived algorithmically — it's Claude's self-assessed certainty, calibrated by the system prompt with an explicit anchor: "use 0.9+ only when markers are textbook-perfect." The database verification step provides the evidence Claude uses to calibrate: a cluster with 9/10 markers confirmed by both databases should yield higher confidence than one with 2/10 confirmed by neither.

The `database_markers_matched / database_markers_total` fields in `CellTypeAnnotation` let downstream users apply their own confidence correction or flag clusters for manual review (the `to_narrative()` method already flags annotations with `confidence < 0.6` as requiring review).

---

## Error handling and the agentic loop

The loop is designed to be resilient without hiding failures:

- **Pydantic validation failure on `record_cell_type`**: the error is sent back to Claude as `is_error: true` with the exception message. Claude can retry with corrected input.
- **Knowledge tool failure**: same pattern — the error is surfaced in the tool result and Claude can adapt.
- **Missing annotations at `end_turn`**: a nudge message lists the missing cluster IDs explicitly. This is more reliable than relying on Claude to notice which clusters it skipped.
- **`max_turns` exceeded**: partial results are returned with a `logging.warning`. This is intentional — better to return 7/8 annotated clusters than to raise an exception that discards everything.

---

## Dependencies

| Package | Role | Why this one |
|---------|------|-------------|
| `anthropic>=0.52` | LLM API client | Official SDK; supports streaming, tool_use, adaptive thinking |
| `anndata>=0.10` | Data container | Universal format for single-cell data; h5ad I/O |
| `numpy>=1.24` | Array operations | LDA output arrays, argmax for dominant topics |
| `pydantic>=2.0` | Structured output | Validation at tool-call boundaries; free JSON serialisation |
| `typer>=0.12` | CLI framework | Type-annotated; `--help` is auto-generated |
| `rich>=13` | Terminal output | Tables, status spinners, colour-coded confidence |
| `pandas` | Tabular output | `to_dataframe()`, database loading |
| `scanpy` (optional) | DE testing, HVG | Industry standard for scRNA-seq preprocessing |
| `scikit-learn` (optional) | `LatentDirichletAllocation` | Mature, well-tested LDA implementation |
| `squidpy` (optional) | Spatial graphs | Spatial neighbourhood graphs for Xenium clustering |
