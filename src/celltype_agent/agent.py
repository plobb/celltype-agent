"""Claude-powered cell type annotation agent.

Sends marker genes to Claude claude-opus-4-6 using tool-use to extract structured
CellTypeAnnotation objects for every cluster.

Claude also has access to knowledge-database tools (search_by_celltype,
search_by_gene) that are executed LOCALLY by this module — only the tool schemas
are sent to the API.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import anthropic

from .knowledge import search_by_celltype, search_by_gene
from .markers import format_markers_for_prompt
from .models import CellTypeAnnotation, TopicAnnotation

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_ANNOTATE_TOOL: dict = {
    "name": "record_cell_type",
    "description": (
        "Record the inferred cell type for one cluster. "
        "Call this once for EACH cluster in the table."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "cluster_id": {
                "type": "string",
                "description": "The cluster identifier exactly as it appears in the table.",
            },
            "predicted_type": {
                "type": "string",
                "description": "Concise cell type name, e.g. 'CD4+ T cell', 'B cell', 'Monocyte'.",
            },
            "confidence": {
                "type": "number",
                "description": (
                    "Your confidence in this annotation, 0.0–1.0. "
                    "Use 0.9+ only when markers are textbook-perfect."
                ),
            },
            "markers_used": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Subset of marker genes that most strongly support this call.",
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "2–4 sentence explanation of why these markers indicate this cell type, "
                    "citing specific genes."
                ),
            },
            "database_support": {
                "type": "string",
                "description": (
                    "Brief summary of what PanglaoDB and CellMarker databases confirmed or "
                    "contradicted about this annotation. Include how many of the cluster's "
                    "markers were found in the database results."
                ),
            },
            "database_markers_matched": {
                "type": "integer",
                "description": "Number of this cluster's markers found in database search results.",
            },
            "database_markers_total": {
                "type": "integer",
                "description": "Total number of markers checked for this cluster.",
            },
        },
        "required": [
            "cluster_id",
            "predicted_type",
            "confidence",
            "markers_used",
            "reasoning",
        ],
        "additionalProperties": False,
    },
}

_SEARCH_BY_CELLTYPE_TOOL: dict = {
    "name": "search_by_celltype",
    "description": (
        "Look up known marker genes for a given cell type in PanglaoDB and CellMarker. "
        "Use this to verify your hypothesis against curated databases."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "cell_type": {
                "type": "string",
                "description": "Cell type name to look up, e.g. 'B cell', 'CD4+ T cell'.",
            },
            "species": {
                "type": "string",
                "description": "'human' or 'mouse'.",
            },
        },
        "required": ["cell_type", "species"],
        "additionalProperties": False,
    },
}

_SEARCH_BY_GENE_TOOL: dict = {
    "name": "search_by_gene",
    "description": (
        "Look up which cell types are associated with a given gene symbol in PanglaoDB "
        "and CellMarker. Use this when fewer than 30% of the cluster's markers appeared "
        "in the cell-type search results, to explore alternative annotations."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "gene": {
                "type": "string",
                "description": "Gene symbol to look up, e.g. 'CD79A', 'MS4A1'.",
            },
            "species": {
                "type": "string",
                "description": "'human' or 'mouse'.",
            },
        },
        "required": ["gene", "species"],
        "additionalProperties": False,
    },
}

_ALL_TOOLS = [_SEARCH_BY_CELLTYPE_TOOL, _SEARCH_BY_GENE_TOOL, _ANNOTATE_TOOL]

_RECORD_TOPIC_TOOL: dict = {
    "name": "record_topic",
    "description": (
        "Record the annotation for one LDA topic. "
        "Call this once for EACH topic in the table."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "topic_id": {
                "type": "integer",
                "description": "The topic index (0-based integer) exactly as shown in the table.",
            },
            "annotation": {
                "type": "string",
                "description": (
                    "Concise label for this gene program, e.g. 'CD4+ T cell', "
                    "'Inflammatory macrophage', 'Fibrosis program', 'Ribosomal artifact'."
                ),
            },
            "category": {
                "type": "string",
                "enum": ["cell_type", "cell_state", "tissue_program", "technical", "ambiguous"],
                "description": (
                    "Broad category: 'cell_type' (specific lineage), 'cell_state' "
                    "(activation/stress state), 'tissue_program' (ECM, angiogenesis, etc.), "
                    "'technical' (ribosomal, mitochondrial), 'ambiguous' (cannot resolve)."
                ),
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in this annotation, 0.0–1.0.",
            },
            "key_genes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Top genes that most strongly define this topic.",
            },
            "reasoning": {
                "type": "string",
                "description": "2–4 sentence explanation citing specific genes.",
            },
            "database_support": {
                "type": "string",
                "description": (
                    "Summary of what PanglaoDB and CellMarker confirmed or contradicted. "
                    "Omit if no database searches were performed."
                ),
            },
        },
        "required": [
            "topic_id",
            "annotation",
            "category",
            "confidence",
            "key_genes",
            "reasoning",
        ],
        "additionalProperties": False,
    },
}

_TOPIC_TOOLS = [_SEARCH_BY_CELLTYPE_TOOL, _SEARCH_BY_GENE_TOOL, _RECORD_TOPIC_TOOL]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_TOPIC_SYSTEM_PROMPT = """\
You are annotating topics from LDA decomposition of Visium spatial transcriptomics data.

Each topic is a gene program derived from co-expression patterns across tissue spots.
Topics may represent:
- A specific cell type (e.g. "T cell", "Hepatocyte")
- A cell state or activation program (e.g. "Inflammatory macrophage", "Stressed epithelial")
- A tissue structure program (e.g. "Extracellular matrix", "Angiogenesis")
- A technical artifact (e.g. "Ribosomal", "Mitochondrial stress")
- An ambiguous mixture that cannot be clearly resolved

For each topic, follow this workflow:

1. **Form a hypothesis** based on the top-weighted genes.
2. **Call `search_by_celltype`** with your hypothesised cell type to verify against databases.
3. **If fewer than 30 % of the top genes** appear in database results, call `search_by_gene`
   on 2–3 of the most distinctive genes to explore alternatives.
4. **Call `record_topic`** with your final annotation and the appropriate `category`.

Additional guidelines
---------------------
* Ribosomal programs (RPS*, RPL*) and mitochondrial programs (MT-*) are usually technical.
* ECM genes (COL*, FN1, VIM) typically indicate stromal / fibroblast programs.
* Assign "ambiguous" and lower confidence for mixed or unresolvable gene programs.
* Call `record_topic` EXACTLY ONCE per topic — do not skip any.
* Do NOT call `record_topic` more than once for the same topic_id.
"""

_SYSTEM_PROMPT = """\
You are a cell type annotation expert specialising in single-cell and spatial
transcriptomics.

For each cluster in the table, follow this "hypothesise then verify" workflow:

1. **Form a hypothesis** based on the top differentially-expressed marker genes.
2. **Call `search_by_celltype`** with your hypothesised cell type to verify it
   against PanglaoDB and CellMarker databases.
3. **If fewer than 30 % of the cluster's markers** appear in the database results,
   call `search_by_gene` on 2–3 of the most distinctive markers to explore
   alternative cell type identities.
4. **Call `record_cell_type`** with your final annotation, including a
   `database_support` summary that states how many markers were confirmed and
   whether the databases agreed with your call.

Additional guidelines
---------------------
* Use well-established marker gene knowledge (e.g. CD3D/CD3E for T cells,
  MS4A1/CD19 for B cells, LYZ/CD14 for monocytes, PPBP/PF4 for platelets).
* When tissue context is provided, prefer tissue-specific subtypes where
  markers clearly support them.
* For ambiguous clusters, assign the most parsimonious cell type and lower your
  confidence score.
* Call `record_cell_type` EXACTLY ONCE per cluster — do not skip any.
* Do NOT call `record_cell_type` more than once for the same cluster_id.
"""


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def _dispatch_tool(name: str, inputs: dict) -> str:
    """Execute a knowledge-database tool locally and return JSON result."""
    if name == "search_by_celltype":
        result = search_by_celltype(
            cell_type=inputs["cell_type"],
            species=inputs["species"],
        )
    elif name == "search_by_gene":
        result = search_by_gene(
            gene=inputs["gene"],
            species=inputs["species"],
        )
    else:
        raise ValueError(f"Unknown knowledge tool: {name!r}")
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def annotate_clusters(
    markers: dict[str, list[str]],
    species: str = "human",
    tissue: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "claude-opus-4-6",
) -> list[CellTypeAnnotation]:
    """Query Claude to annotate clusters from their marker genes.

    Parameters
    ----------
    markers:
        Mapping of cluster_id → list of top marker gene names.
    species:
        Biological species ('human' or 'mouse').
    tissue:
        Optional tissue/organ context (e.g. 'PBMC', 'lung', 'bone marrow').
    api_key:
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY`` env var.
    model:
        Claude model to use.

    Returns
    -------
    List of :class:`CellTypeAnnotation` objects, one per cluster.
    """
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    marker_table = format_markers_for_prompt(markers, species=species, tissue=tissue)
    n_clusters = len(markers)

    user_message = (
        f"Please annotate all {n_clusters} cluster(s) shown below. "
        f"For each cluster: hypothesise the cell type, verify via search_by_celltype, "
        f"check search_by_gene if needed, then call record_cell_type.\n\n"
        f"{marker_table}"
    )

    log.info("Sending %d clusters to %s for annotation…", n_clusters, model)

    messages: list[dict] = [{"role": "user", "content": user_message}]
    annotations: dict[str, CellTypeAnnotation] = {}

    # Agentic loop — Claude may need several turns to call the tool for all clusters
    max_turns = 30
    for turn in range(max_turns):
        with client.messages.stream(
            model=model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=_SYSTEM_PROMPT,
            tools=_ALL_TOOLS,
            tool_choice={"type": "auto"},
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        # Collect tool calls from this turn
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        tool_results = []

        for block in tool_use_blocks:
            if block.name == "record_cell_type":
                try:
                    ann = CellTypeAnnotation(**block.input)
                    annotations[ann.cluster_id] = ann
                    log.debug(
                        "Annotated cluster %s → %s", ann.cluster_id, ann.predicted_type
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Recorded.",
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "Failed to parse annotation for block %s: %s", block.id, exc
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "is_error": True,
                            "content": str(exc),
                        }
                    )

            elif block.name in ("search_by_celltype", "search_by_gene"):
                try:
                    result_json = _dispatch_tool(block.name, block.input)
                    log.debug("Tool %s(%s) → %s", block.name, block.input, result_json[:120])
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_json,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("Tool %s failed: %s", block.name, exc)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "is_error": True,
                            "content": str(exc),
                        }
                    )

        # Append the assistant turn + tool results
        messages.append({"role": "assistant", "content": response.content})
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Check if we have all clusters
        missing = set(markers.keys()) - set(annotations.keys())
        if not missing:
            log.info("All %d clusters annotated in %d turn(s).", n_clusters, turn + 1)
            break

        # If Claude stopped without tool calls and clusters are still missing, nudge it
        if response.stop_reason == "end_turn" and missing:
            if turn < max_turns - 1:
                log.debug("Nudging Claude to annotate remaining clusters: %s", missing)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Please also call `record_cell_type` for the remaining "
                            f"clusters: {sorted(missing)}"
                        ),
                    }
                )
            continue

        if response.stop_reason != "tool_use":
            break
    else:
        missing = set(markers.keys()) - set(annotations.keys())
        if missing:
            log.warning(
                "Reached max turns (%d) with %d cluster(s) still unannotated: %s",
                max_turns,
                len(missing),
                sorted(missing),
            )

    return list(annotations.values())


def annotate_topics(
    topic_genes: dict[int, list[tuple[str, float]]],
    species: str = "human",
    tissue: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "claude-opus-4-6",
) -> list[TopicAnnotation]:
    """Query Claude to annotate LDA topics from their top-weighted genes.

    Parameters
    ----------
    topic_genes:
        Mapping of topic_id → list of (gene, weight) tuples, as returned by
        :func:`~celltype_agent.deconvolution.run_lda`.
    species:
        Biological species (``'human'`` or ``'mouse'``).
    tissue:
        Optional tissue/organ context.
    api_key:
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    model:
        Claude model to use.

    Returns
    -------
    List of :class:`~celltype_agent.models.TopicAnnotation` objects, one per topic.
    """
    from .deconvolution import format_topics_for_prompt  # noqa: PLC0415

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    topic_table = format_topics_for_prompt(topic_genes)
    n_topics = len(topic_genes)
    tissue_str = f" in {tissue}" if tissue else ""

    user_message = (
        f"Please annotate all {n_topics} LDA topic(s) from this {species} "
        f"spatial transcriptomics dataset{tissue_str}. "
        f"For each topic: hypothesise what it represents, verify via search_by_celltype, "
        f"check search_by_gene if needed, then call record_topic.\n\n"
        f"{topic_table}"
    )

    log.info("Sending %d topics to %s for annotation…", n_topics, model)

    messages: list[dict] = [{"role": "user", "content": user_message}]
    annotations: dict[int, TopicAnnotation] = {}

    max_turns = 30
    for turn in range(max_turns):
        with client.messages.stream(
            model=model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=_TOPIC_SYSTEM_PROMPT,
            tools=_TOPIC_TOOLS,
            tool_choice={"type": "auto"},
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        tool_results = []

        for block in tool_use_blocks:
            if block.name == "record_topic":
                try:
                    ann = TopicAnnotation(**block.input)
                    annotations[ann.topic_id] = ann
                    log.debug("Annotated topic %d → %s", ann.topic_id, ann.annotation)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Recorded.",
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "Failed to parse topic annotation for block %s: %s", block.id, exc
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "is_error": True,
                            "content": str(exc),
                        }
                    )

            elif block.name in ("search_by_celltype", "search_by_gene"):
                try:
                    result_json = _dispatch_tool(block.name, block.input)
                    log.debug("Tool %s(%s) → %s", block.name, block.input, result_json[:120])
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_json,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("Tool %s failed: %s", block.name, exc)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "is_error": True,
                            "content": str(exc),
                        }
                    )

        messages.append({"role": "assistant", "content": response.content})
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        missing = set(topic_genes.keys()) - set(annotations.keys())
        if not missing:
            log.info("All %d topics annotated in %d turn(s).", n_topics, turn + 1)
            break

        if response.stop_reason == "end_turn" and missing:
            if turn < max_turns - 1:
                log.debug("Nudging Claude for remaining topics: %s", sorted(missing))
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Please also call `record_topic` for the remaining "
                            f"topic IDs: {sorted(missing)}"
                        ),
                    }
                )
            continue

        if response.stop_reason != "tool_use":
            break
    else:
        missing = set(topic_genes.keys()) - set(annotations.keys())
        if missing:
            log.warning(
                "Reached max turns (%d) with %d topic(s) still unannotated: %s",
                max_turns,
                len(missing),
                sorted(missing),
            )

    return list(annotations.values())
