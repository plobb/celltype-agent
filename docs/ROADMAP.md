# Roadmap

What the path to v1.0 looks like, and what would need to change for production or regulated-environment use.

---

## Current state (v0.1)

The library covers the core annotation workflow end-to-end:
- scRNA-seq cluster annotation with hypothesise-then-verify tool-use
- Visium LDA deconvolution with auto-K and Claude topic annotation
- Xenium spatial clustering + annotation
- Dual-database grounding (PanglaoDB + CellMarker)
- Structured Pydantic output with narrative/methods generation
- CLI with Rich tables, Python API, Jupyter display

### Known gaps

| Area | Limitation |
|------|------------|
| Model cost | Opus default is ~4× more expensive than Sonnet. No cost guardrail. |
| Auto-K speed | Grid search over K=3–20 is the dominant runtime for Visium runs |
| Species coverage | Human and mouse only; rat, zebrafish, organoid models not supported |
| Databases | PanglaoDB snapshot is March 2020; no mechanism to update |
| Batch processing | Single AnnData per call; no async batch mode |
| Error recovery | Partial results on `max_turns` exceeded — no checkpoint/resume |
| Validation | No benchmark against ground-truth labelled datasets |

---

## Short-term improvements (toward v0.2)

### Switch default model to Sonnet

Once annotation quality on common cell types (PBMC, bone marrow, lung) is validated as equivalent between Sonnet and Opus, Sonnet becomes the better default — cheaper and nearly as good on well-characterised cell types. A `quality="high"` flag could map to Opus for difficult samples (rare subtypes, poorly annotated tissues).

### Better auto-K selection

The current perplexity minimisation sometimes selects `max_k` on datasets where perplexity decreases monotonically. Two improvements:
1. **Elbow detection**: fit a piecewise linear model to the perplexity curve and find the inflection point — the perplexity gain per additional topic diminishes where the elbow is.
2. **Parallel K fitting**: the K grid search is embarrassingly parallel; using `joblib` to run multiple K fits in parallel would cut wall time proportionally.

### Async API calls

The annotation loop makes synchronous API calls. For large datasets (50+ clusters), async calls with `asyncio` would allow multiple clusters to be processed in parallel — though the rate limiter would cap the speedup.

### Cost estimation before running

A pre-flight `estimate_cost(adata, model=...)` function would estimate token counts from the marker table and return a dollar estimate. Prevents surprise API bills.

---

## Medium-term (toward v1.0)

### Multimodal data: CITE-seq and ATAC

**CITE-seq** (simultaneous RNA + protein surface markers): surface protein expression resolves ambiguities that RNA markers can't (e.g. CD4 vs CD8 when both lineages express similar transcriptomes). The annotation prompt could accept a parallel `protein_markers` dict alongside `gene_markers`.

**ATAC-seq** (chromatin accessibility): open chromatin regions can indicate cell identity through transcription factor binding motifs. Annotating ATAC peaks requires a different marker concept (motif enrichment rather than gene expression) — likely a separate `annotate_atac()` entry point.

### Reference-based annotation integration

The current tool-use loop is reference-free. Integrating a step that queries a cell atlas (Human Cell Atlas, Tabula Sapiens) via their REST APIs would add a third grounding source alongside PanglaoDB and CellMarker. This is architecturally straightforward — add a `search_by_reference_atlas(cell_type, tissue)` tool that hits the HCA API and returns matching cell type hierarchies.

### Hierarchical annotation

Cell type labels exist in a hierarchy (e.g. T cell → CD4+ T cell → Treg). The current model returns flat labels. A hierarchical output model (`CellTypeAnnotation.lineage = ["Lymphoid", "T cell", "CD4+ T cell"]`) would let downstream visualisation tools position cells in a taxonomy and make partial-confidence annotations more informative.

### Database update mechanism

PanglaoDB's March 2020 snapshot will age. A `celltype-agent update-databases` CLI command that pulls fresh snapshots and regenerates the local data files would address this without requiring a library update.

### Cloud deployment

Wrapping the annotation logic as a REST API (FastAPI) would allow:
- Shared annotation service for a research team without requiring every user to have an API key
- Request queuing and rate limiting at the service boundary
- Centralised logging and audit trails

The Pydantic models serialise directly to JSON with `model.model_dump_json()`, so the API response format is already defined.

---

## Longer-term: clinical and regulated-environment readiness

Using LLM-based annotation in a clinical diagnostics context (e.g. characterising a patient biopsy for treatment decisions) would require changes that go beyond software engineering.

### Validation and benchmarking

Clinical software must demonstrate performance against ground truth:
- Benchmark against datasets with manually curated gold-standard labels (e.g. published PBMC atlases with expert annotation)
- Report sensitivity/specificity per cell type, confidence calibration curves, and failure modes on out-of-distribution data
- Document the training data of the underlying LLM and assess potential for systematic bias in underrepresented cell types or species

### Versioning and reproducibility

An annotation result must be reproducible given the same input:
- **Model versioning**: pin the exact model revision (not just family). Claude model weights are updated; `claude-opus-4-6` today may produce different outputs in six months.
- **Database versioning**: include database version in the `AnnotationResult` metadata. PanglaoDB and CellMarker are periodically updated; a result annotated with different database snapshots is not directly comparable.
- **Deterministic output**: LLM outputs are inherently stochastic. For regulated use, this requires either deterministic sampling (temperature=0) or ensemble annotation with majority voting.

### Audit trails for GDPR / IVDR

Under the EU's In Vitro Diagnostic Regulation (IVDR), software used in medical devices must maintain audit trails of decisions. This would require:
- Logging the full Claude conversation (with tool calls and results) alongside each annotation result
- Storing who ran the annotation, when, and with which parameters
- An immutable record that can be reviewed if an annotation is disputed

The current `logging.DEBUG` output captures tool calls — formalising this into a structured audit log (e.g. JSON Lines to an audit store) is the engineering work. The governance framework around data retention, access control, and consent for patient data is an organisational problem.

### Cost at scale

A single PBMC dataset costs ~$0.10–0.40 with Opus. Clinical genomics sequencing centres processing hundreds of samples per day would face $10,000+/month API costs at current pricing. This changes the economics significantly:
- Sonnet or a fine-tuned smaller model becomes necessary
- Caching identical or near-identical marker sets (same cell type appears across many patients) would reduce redundant API calls substantially
- Local model deployment (Claude API on-premises, when/if available) would give cost predictability

---

## Production deployment thoughts

### Containerisation

The library has optional dependencies that are heavy (squidpy pulls in many geospatial packages). A multi-stage Docker build with separate images for `[scanpy]`, `[spatial]`, and `[all]` would give leaner deployments for users who only need one modality.

### API service wrapping

A FastAPI service wrapping `annotate()` would expose:
- `POST /annotate` accepting a serialised AnnData (or pre-extracted marker dict as JSON) and returning an `AnnotationResult`
- `GET /jobs/{id}` for async job status (necessary for large spatial datasets where LDA takes minutes)
- `GET /estimate` for cost estimation before committing to a run

### Scaling the database lookups

At current scale (8,286 + 96,075 entries), the in-memory DataFrames work fine. At 10×–100× scale (if additional databases are added), a proper inverted index (e.g. SQLite FTS, or a lightweight vector store for semantic gene-set search) would be more efficient. The `search_by_celltype` / `search_by_gene` interface is stable and wouldn't change — only the implementation behind it.
