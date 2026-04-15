"""Pydantic models for cell type annotation results."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


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
