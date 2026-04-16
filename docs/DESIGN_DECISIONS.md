# Design Decisions

ADR-style records of significant technical choices. Format: Context → Options → Decision → Consequences.

---

## 1. Agentic tool-use vs single-shot LLM prompt

**Context**: The core task is mapping a cluster's marker gene list to a cell type name. The simplest approach is a single prompt with all clusters that asks for a JSON response.

**Options considered**:
- Single-shot: send all clusters, ask for JSON array of annotations
- Tool-use loop: define typed tools that Claude calls once per cluster, with intermediate database lookups

**Decision**: Tool-use loop.

**Consequences**:
- (+) Claude can interrogate the databases mid-annotation and revise its hypothesis if they contradict it. A single-shot prompt can't do this.
- (+) `record_cell_type` forces one structured call per cluster — impossible to accidentally annotate a cluster twice or skip one without detection.
- (+) Failed tool calls are surfaced and retried within the same conversation. A single-shot JSON parse failure is unrecoverable.
- (-) More API turns → higher cost and latency. A PBMC3k dataset (8 clusters) typically takes 4–6 turns.
- (-) `max_turns` is a fragile upper bound; pathological inputs can exhaust it and return partial results.

---

## 2. Pydantic for structured output vs ad-hoc JSON parsing

**Context**: Claude's tool call inputs arrive as raw dicts. We need typed objects with validated fields.

**Options considered**:
- `json.loads()` + manual dict access
- `dataclasses` with manual validation
- Pydantic v2 models

**Decision**: Pydantic v2.

**Consequences**:
- (+) Validation happens at the exact boundary where untrusted data enters — Claude's tool call input. A confidence value of `1.7` raises `ValidationError` immediately rather than silently propagating.
- (+) `model.model_dump()`, `.model_dump_json()`, and JSON Schema generation come for free — used by `to_dataframe()` and the tool schema definitions themselves.
- (+) `field_validator` for `predicted_type` whitespace stripping handles a common LLM output quirk without extra code.
- (-) Pydantic is a non-trivial dependency (~1.5MB). For a library, this is worth auditing.
- (-) Pydantic v1/v2 migration is a known industry pain point; pinning `>=2.0` avoids this but means Python 3.9 is the minimum supported version for users who encounter ecosystem conflicts.

---

## 3. Local tool dispatch vs MCP server

**Context**: The database lookup tools (`search_by_celltype`, `search_by_gene`) need to be executable when Claude calls them.

**Options considered**:
- MCP (Model Context Protocol) server: run the databases as a persistent process
- Local dispatch: execute the tool in-process via `_dispatch_tool()`

**Decision**: Local in-process dispatch.

**Consequences**:
- (+) Zero infrastructure overhead — no server to start, port to manage, or Docker container to ship.
- (+) The databases are already in-memory after the first call; tool execution is microseconds.
- (+) Full Python stack traces when a lookup fails; no serialisation boundary to debug across.
- (-) Embedding the tool execution inside the library makes it harder to swap in a different database source later. An MCP server would expose a stable interface that any database implementation could satisfy.
- (-) Not composable with other Claude agent frameworks that expect MCP-style tool servers.
- **Deferred**: if this becomes a hosted API service (see ROADMAP), MCP would be worth revisiting.

---

## 4. Dual databases (PanglaoDB + CellMarker) vs single source

**Context**: One curated marker gene database is simpler to maintain. Two creates potential for conflicting results.

**Options considered**:
- PanglaoDB only (simpler, but dated — March 2020)
- CellMarker 2.0 only (larger, but less structured)
- Both, with Claude adjudicating disagreements

**Decision**: Both.

**Consequences**:
- (+) Disagreement between databases is informative signal. A cluster where PanglaoDB says "T cell" and CellMarker says nothing is less certain than one both databases confirm.
- (+) CellMarker has 11× more entries than PanglaoDB; for rare cell types or mouse data, CellMarker often has coverage where PanglaoDB doesn't.
- (+) Having both prevents over-indexing on either database's curation choices (PanglaoDB is more conservative; CellMarker includes more tissue-specific subtypes).
- (-) Two databases means two loading paths, two fuzzy matchers, and two output keys in every tool result — more code to maintain.
- (-) The databases are not curated against each other. Marker gene sets sometimes contradict.
- **Accepted**: the concordance information is surfaced to Claude and to users (`database_markers_matched/total`); downstream callers can decide how much weight to give it.

---

## 5. LDA for Visium deconvolution vs CARD / RCTD / cell2location

**Context**: Visium spots are cellular mixtures. Reference-based deconvolution methods (CARD, RCTD, cell2location) can estimate precise cell type proportions given a single-cell reference atlas.

**Options considered**:
- Reference-based: CARD, RCTD, cell2location — state of the art for proportion estimation when a reference exists
- Reference-free: LDA, NMF — no reference required

**Decision**: LDA.

**Consequences**:
- (+) No reference atlas required. Many tissues lack a good public single-cell reference, especially non-human or rare tissue types.
- (+) LDA is in scikit-learn — a ubiquitous dependency with no R interop requirement. CARD and RCTD are R packages.
- (+) LDA topics are interpretable gene programs, not just proportion estimates. They surface tissue structural programs (ECM, angiogenesis) that reference-based methods miss because those programs don't correspond to discrete cell types.
- (-) LDA cannot give precise cell type fraction estimates — spot proportions are topic mixtures, not cell type fractions. If precise composition estimation is the goal, reference-based methods are better.
- (-) LDA requires a count matrix; normalised/log-transformed data needs to be rounded back to integers. This is lossy.
- **Accepted**: the use case here is annotation and interpretation, not cell type quantification. LDA is the right tool for that.

---

## 6. Auto-K via perplexity vs fixed K

**Context**: LDA requires specifying the number of topics K upfront.

**Options considered**:
- Fixed K: require the user to choose
- Automatic: grid search over K range with a model selection criterion

**Decision**: Auto-select using held-out perplexity, with fixed K as an override option.

**Consequences**:
- (+) Removes the need for the user to understand LDA model selection. Most biologists should not need to know what perplexity is to annotate their data.
- (+) Perplexity is a principled likelihood-based criterion; it measures how well the model predicts held-out data, which is a reasonable proxy for topic coherence.
- (-) Slow: fitting LDA for K=3 through K=20 with 50 iterations each on 3,000 spots is roughly 17 model fits. On large datasets this is the dominant runtime cost.
- (-) Perplexity monotonically decreases as K increases in some datasets (overfit), which can cause auto-K to select `max_k`. An elbow-finding heuristic would be more robust but harder to implement reliably.
- **Known limitation**: flagged in ROADMAP for improvement (e.g. add elbow detection, parallelise the K search).

---

## 7. Opus default vs Sonnet (cost trade-off)

**Context**: Claude Opus is significantly more capable but costs more per token than Sonnet.

**Options considered**:
- Default to `claude-opus-4-6`: best annotation quality, higher cost
- Default to `claude-sonnet-4-6`: ~4× cheaper per token, slightly lower quality for subtle cell types
- Let the user choose with no default

**Decision**: Default to `claude-opus-4-6`.

**Consequences**:
- (+) For rare or tissue-specific cell types (e.g. "alveolar type II epithelial" vs "lung epithelial"), Opus reliably distinguishes them. Sonnet occasionally collapses to the coarser label.
- (+) Annotation is typically a one-off operation per dataset, not a high-frequency API call. At 8–50 clusters and ~4–6 turns, a single annotation run costs $0.10–$0.40 with Opus. This is acceptable for research.
- (-) For routine use on large datasets (>100 clusters, batch mode), Sonnet would save ~75% in API costs.
- (-) The default choice is sticky — most users won't change it.
- **Known trade-off**: documented in ROADMAP as a candidate for switching to Sonnet default once quality equivalence on common cell types is validated.

---

## 8. Hypothesise-then-verify vs verify-only

**Context**: An alternative workflow would skip the hypothesis step and go straight to querying the databases with the top marker genes.

**Options considered**:
- Verify-only: for each cluster, query `search_by_gene` for every top marker, then ask Claude to synthesise
- Hypothesise-first: Claude forms a hypothesis, then queries with the hypothesis as the cell type name

**Decision**: Hypothesise-first.

**Consequences**:
- (+) `search_by_celltype` is a much more targeted query than running `search_by_gene` on 10 markers and hoping the results converge. The hypothesis guides which database query is most informative.
- (+) Claude's prior knowledge (textbook marker patterns) often gets the cell type right without any database calls for common cell types. The verify step then confirms it efficiently.
- (-) The hypothesis can be wrong and the verification query biased. If Claude hypothesises "NK cell" and queries `search_by_celltype("NK cell")`, it will see a gene list that may look superficially compatible even if the cluster is actually a cytotoxic T cell.
- **Mitigation**: the 30% concordance threshold forces a pivot to `search_by_gene` when the hypothesis-directed lookup doesn't explain the cluster's actual markers.

---

## 9. Category enum for spatial topics

**Context**: Spatial LDA topics represent heterogeneous biological concepts, but the naive approach treats all topics as cell types.

**Options considered**:
- Free text annotation: let Claude describe whatever it finds
- Boolean `is_cell_type` flag
- Multi-class enum: `cell_type`, `cell_state`, `tissue_program`, `technical`, `ambiguous`

**Decision**: Five-class enum.

**Consequences**:
- (+) Downstream code can filter `category == "technical"` to remove ribosomal/mitochondrial noise topics before visualisation.
- (+) `cell_state` vs `cell_type` is a meaningful distinction for biological interpretation: activation states don't map to discrete lineage nodes in cell atlases.
- (+) The enum forces Claude to make a choice rather than hedging with prose. Ambiguous cases get `ambiguous` + low confidence, which is honest and actionable.
- (-) Five categories is a design choice that bakes in assumptions about spatial transcriptomics biology. A different tissue type (e.g. tumour) might warrant different categories.
- (-) The boundary between `cell_state` and `cell_type` is fuzzy (is "regulatory T cell" a type or a state?).

---

## 10. Testing strategy: mocking LLM calls, synthetic AnnData, importorskip

**Context**: Tests that make real Anthropic API calls are slow, expensive, and non-deterministic.

**Options considered**:
- Integration tests with real API (expensive, slow, requires key in CI)
- Pure mocking: mock all Claude responses (fast, fragile — tests what the mock returns, not what Claude returns)
- Test everything except the API call, mock only at the network boundary

**Decision**: Mock only at the `anthropic.Anthropic` class boundary; test all business logic with synthetic data.

**Consequences**:
- (+) `test_narrative.py` uses `unittest.mock.patch("anthropic.Anthropic")` — the mock returns a controlled response, and the test verifies that the result model is populated correctly. The LLM behaviour is not tested (can't be), but the wiring is.
- (+) `test_deconvolution.py` generates structured synthetic AnnData with known block-diagonal gene programs. This tests that LDA recovers approximately the right number of topics on data with real structure.
- (+) `pytest.importorskip("sklearn")` / `pytest.importorskip("squidpy")` means the test suite runs in the core dependency environment without failing on optional deps. CI can run a subset of tests without installing the full `[spatial]` extra.
- (-) LLM output quality cannot be regression-tested without golden fixtures or real API calls. The tests verify structure and wiring, not annotation correctness.
- **Accepted**: annotation quality is validated manually on known datasets (PBMC3k, Human Cell Atlas references). Automated regression testing for LLM quality is a future concern.
