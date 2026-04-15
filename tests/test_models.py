"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from celltype_agent.models import AnnotationResult, CellTypeAnnotation


def _make_ann(cluster_id="0", predicted_type="T cell", confidence=0.9):
    return CellTypeAnnotation(
        cluster_id=cluster_id,
        predicted_type=predicted_type,
        confidence=confidence,
        markers_used=["CD3D", "CD3E"],
        reasoning="CD3D and CD3E are canonical T cell markers.",
    )


def test_annotation_valid():
    ann = _make_ann()
    assert ann.cluster_id == "0"
    assert ann.predicted_type == "T cell"
    assert ann.confidence == 0.9


def test_annotation_strips_whitespace():
    ann = _make_ann(predicted_type="  B cell  ")
    assert ann.predicted_type == "B cell"


def test_annotation_confidence_bounds():
    with pytest.raises(ValidationError):
        _make_ann(confidence=1.5)
    with pytest.raises(ValidationError):
        _make_ann(confidence=-0.1)


def test_result_as_dict():
    anns = [_make_ann("0", "T cell"), _make_ann("1", "B cell")]
    result = AnnotationResult(
        annotations=anns, species="human", n_clusters=2
    )
    d = result.as_dict()
    assert d["0"].predicted_type == "T cell"
    assert d["1"].predicted_type == "B cell"


def test_result_to_labels():
    anns = [_make_ann("0", "T cell"), _make_ann("1", "B cell")]
    result = AnnotationResult(
        annotations=anns, species="human", n_clusters=2
    )
    assert result.to_labels() == {"0": "T cell", "1": "B cell"}
