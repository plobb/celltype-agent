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
from .models import CellTypeAnnotation

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

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

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
