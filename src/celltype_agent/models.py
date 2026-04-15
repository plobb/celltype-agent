"""Pydantic models for cell type annotation results."""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    import pandas as pd


class CellTypeAnnotation(BaseModel):
    """Structured cell type annotation for a single cluster."""

    cluster_id: str = Field(description="Cluster identifier (e.g. '0', '1', 'leiden_3')")
    predicted_type: str = Field(description="Predicted cell type name")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    markers_used: list[str] = Field(
        description="Marker genes that drove this annotation"
    )
    reasoning: str = Field(
        description="Step-by-step reasoning for the cell type assignment"
    )
    database_support: Optional[str] = Field(
        default=None,
        description="Summary of which databases agreed or disagreed with the annotation",
    )
    database_markers_matched: Optional[int] = Field(
        default=None,
        description="Number of cluster markers found in database results",
    )
    database_markers_total: Optional[int] = Field(
        default=None,
        description="Total number of cluster markers checked against databases",
    )

    @field_validator("predicted_type")
    @classmethod
    def strip_predicted_type(cls, v: str) -> str:
        return v.strip()


class AnnotationResult(BaseModel):
    """Full annotation result for an AnnData object."""

    annotations: list[CellTypeAnnotation] = Field(default_factory=list)
    species: str
    tissue: Optional[str] = None
    n_clusters: int
    model_used: str = "claude-opus-4-6"

    def as_dict(self) -> dict[str, CellTypeAnnotation]:
        """Return cluster_id -> CellTypeAnnotation mapping."""
        return {ann.cluster_id: ann for ann in self.annotations}

    def to_labels(self) -> dict[str, str]:
        """Return cluster_id -> predicted_type mapping (for adata.obs)."""
        return {ann.cluster_id: ann.predicted_type for ann in self.annotations}

    def to_dataframe(self) -> pd.DataFrame:
        """Return annotations as a :class:`pandas.DataFrame`.

        Columns: ``cluster_id``, ``predicted_type``, ``confidence``,
        ``markers_used``, ``reasoning``, ``database_support``,
        ``database_markers_matched``, ``database_markers_total``.
        """
        import pandas as pd  # noqa: PLC0415

        rows = []
        for ann in self._sorted_annotations():
            rows.append(
                {
                    "cluster_id": ann.cluster_id,
                    "predicted_type": ann.predicted_type,
                    "confidence": ann.confidence,
                    "markers_used": ", ".join(ann.markers_used),
                    "reasoning": ann.reasoning,
                    "database_support": ann.database_support,
                    "database_markers_matched": ann.database_markers_matched,
                    "database_markers_total": ann.database_markers_total,
                }
            )
        return pd.DataFrame(rows)

    def to_csv(self, path: Union[str, Path]) -> None:
        """Write annotations to *path* as a CSV file."""
        self.to_dataframe().to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Jupyter display
    # ------------------------------------------------------------------

    def _repr_html_(self) -> str:
        """Render a styled HTML table for Jupyter notebook display."""
        context_parts = [f"<b>Species:</b> {html.escape(self.species)}"]
        if self.tissue:
            context_parts.append(f"<b>Tissue:</b> {html.escape(self.tissue)}")
        context_parts.append(f"<b>Model:</b> {html.escape(self.model_used)}")
        context_parts.append(f"<b>Clusters:</b> {self.n_clusters}")
        context_line = " &nbsp;|&nbsp; ".join(context_parts)

        header = (
            "<tr>"
            "<th>Cluster</th>"
            "<th>Cell Type</th>"
            "<th>Confidence</th>"
            "<th>Key Markers</th>"
            "<th>DB Support</th>"
            "</tr>"
        )

        rows_html = []
        for ann in self._sorted_annotations():
            conf_color = _confidence_color(ann.confidence)
            conf_bg = _confidence_bg(ann.confidence)

            db_cell = html.escape(ann.database_support or "—")
            if ann.database_markers_matched is not None and ann.database_markers_total:
                frac = f"{ann.database_markers_matched}/{ann.database_markers_total} matched"
                if ann.database_support:
                    db_cell = f"{frac} — {html.escape(ann.database_support)}"
                else:
                    db_cell = frac

            markers_str = html.escape(", ".join(ann.markers_used[:5]))

            rows_html.append(
                f"<tr>"
                f'<td style="text-align:center;font-weight:bold;">'
                f"{html.escape(ann.cluster_id)}</td>"
                f'<td style="font-weight:bold;">{html.escape(ann.predicted_type)}</td>'
                f'<td style="text-align:center;color:{conf_color};'
                f'background:{conf_bg};border-radius:4px;font-weight:bold;">'
                f"{ann.confidence:.2f}</td>"
                f"<td><code>{markers_str}</code></td>"
                f'<td style="font-size:0.85em;color:#555;">{db_cell}</td>'
                f"</tr>"
            )

        table_style = (
            "border-collapse:collapse;width:100%;font-family:sans-serif;"
            "font-size:0.9em;"
        )
        th_style = (
            "background:#2c3e50;color:white;padding:8px 12px;"
            "text-align:left;border:1px solid #ddd;"
        )
        td_style = "padding:7px 12px;border:1px solid #ddd;vertical-align:top;"

        # Inject td style via a <style> block scoped to a wrapper div
        return f"""
<div style="font-family:sans-serif;">
  <p style="margin:4px 0 8px 0;font-size:0.85em;color:#666;">{context_line}</p>
  <style>
    .cta-result th {{ {th_style} }}
    .cta-result td {{ {td_style} }}
    .cta-result tr:nth-child(even) td {{ background:#f8f9fa; }}
  </style>
  <table class="cta-result" style="{table_style}">
    <thead>{header}</thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</div>
""".strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sorted_annotations(self) -> list[CellTypeAnnotation]:
        def _key(ann: CellTypeAnnotation):
            try:
                return (0, int(ann.cluster_id))
            except ValueError:
                return (1, ann.cluster_id)

        return sorted(self.annotations, key=_key)


def _confidence_color(conf: float) -> str:
    if conf >= 0.8:
        return "#1a7f37"
    if conf >= 0.6:
        return "#9a6700"
    return "#cf222e"


def _confidence_bg(conf: float) -> str:
    if conf >= 0.8:
        return "#d4edda"
    if conf >= 0.6:
        return "#fff3cd"
    return "#f8d7da"
