"""Claude-powered cell type annotation agent.

Sends marker genes to Claude claude-opus-4-6 using tool-use to extract structured
CellTypeAnnotation objects for every cluster.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import anthropic

from .markers import format_markers_for_prompt
from .models import CellTypeAnnotation

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definition
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

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert computational biologist specialising in single-cell and spatial
transcriptomics.  Your task is to annotate cell clusters based on their top
differentially-expressed marker genes.

Guidelines
----------
* Use well-established marker gene knowledge (e.g. CD3D/CD3E for T cells,
  MS4A1/CD19 for B cells, LYZ/CD14 for monocytes, PPBP/PF4 for platelets, etc.).
* When tissue context is provided, prefer tissue-specific subtypes where the
  markers clearly support them.
* For ambiguous clusters, assign the most parsimonious cell type and lower your
  confidence score.
* Call `record_cell_type` EXACTLY ONCE per cluster — do not skip any.
* Do NOT call `record_cell_type` more than once for the same cluster_id.
"""


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
        f"Call `record_cell_type` exactly once per cluster.\n\n"
        f"{marker_table}"
    )

    log.info("Sending %d clusters to %s for annotation…", n_clusters, model)

    messages: list[dict] = [{"role": "user", "content": user_message}]
    annotations: dict[str, CellTypeAnnotation] = {}

    # Agentic loop — Claude may need several turns to call the tool for all clusters
    max_turns = 10
    for turn in range(max_turns):
        with client.messages.stream(
            model=model,
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=_SYSTEM_PROMPT,
            tools=[_ANNOTATE_TOOL],
            tool_choice={"type": "auto"},
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        # Collect tool calls from this turn
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        tool_results = []

        for block in tool_use_blocks:
            if block.name != "record_cell_type":
                continue

            try:
                ann = CellTypeAnnotation(**block.input)
                annotations[ann.cluster_id] = ann
                log.debug("Annotated cluster %s → %s", ann.cluster_id, ann.predicted_type)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Recorded.",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to parse annotation for block %s: %s", block.id, exc)
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
