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


# ---------------------------------------------------------------------------
# _repr_html_
# ---------------------------------------------------------------------------


def _make_result(**kwargs):
    anns = [
        _make_ann("0", "T cell", confidence=0.9),
        CellTypeAnnotation(
            cluster_id="1",
            predicted_type="B cell",
            confidence=0.75,
            markers_used=["MS4A1", "CD79A"],
            reasoning="MS4A1 and CD79A are canonical B cell markers.",
            database_support="Both databases confirmed B cell",
            database_markers_matched=8,
            database_markers_total=10,
        ),
        _make_ann("2", "Monocyte", confidence=0.5),
    ]
    return AnnotationResult(
        annotations=anns, species="human", tissue="PBMC", n_clusters=3, **kwargs
    )


def test_repr_html_returns_string():
    result = _make_result()
    html = result._repr_html_()
    assert isinstance(html, str)


def test_repr_html_contains_cell_types():
    result = _make_result()
    html = result._repr_html_()
    assert "T cell" in html
    assert "B cell" in html
    assert "Monocyte" in html


def test_repr_html_contains_confidence_values():
    result = _make_result()
    html = result._repr_html_()
    assert "0.90" in html
    assert "0.75" in html
    assert "0.50" in html


def test_repr_html_contains_context():
    result = _make_result()
    html = result._repr_html_()
    assert "human" in html
    assert "PBMC" in html


def test_repr_html_shows_db_support():
    result = _make_result()
    html = result._repr_html_()
    # Cluster 1 has database_markers_matched/total set
    assert "8/10 matched" in html


def test_repr_html_confidence_colors():
    result = _make_result()
    html = result._repr_html_()
    # High confidence (0.9) → green tones
    assert "#1a7f37" in html or "#d4edda" in html
    # Low confidence (0.5) → red tones
    assert "#cf222e" in html or "#f8d7da" in html


def test_repr_html_clusters_sorted():
    # Numeric cluster IDs should appear in order 0, 1, 2
    result = _make_result()
    html = result._repr_html_()
    pos_0 = html.index(">0<")
    pos_1 = html.index(">1<")
    pos_2 = html.index(">2<")
    assert pos_0 < pos_1 < pos_2


# ---------------------------------------------------------------------------
# to_dataframe
# ---------------------------------------------------------------------------


def test_to_dataframe_shape():
    pytest.importorskip("pandas")
    result = _make_result()
    df = result.to_dataframe()
    assert len(df) == 3
    assert list(df["cluster_id"]) == ["0", "1", "2"]


def test_to_dataframe_columns():
    pytest.importorskip("pandas")
    result = _make_result()
    df = result.to_dataframe()
    expected_cols = {
        "cluster_id", "predicted_type", "confidence",
        "markers_used", "reasoning", "database_support",
        "database_markers_matched", "database_markers_total",
    }
    assert expected_cols.issubset(set(df.columns))


def test_to_dataframe_values():
    pytest.importorskip("pandas")
    result = _make_result()
    df = result.to_dataframe()
    row1 = df[df["cluster_id"] == "1"].iloc[0]
    assert row1["predicted_type"] == "B cell"
    assert row1["confidence"] == pytest.approx(0.75)
    assert row1["database_markers_matched"] == 8


def test_to_dataframe_markers_joined():
    pytest.importorskip("pandas")
    result = _make_result()
    df = result.to_dataframe()
    row0 = df[df["cluster_id"] == "0"].iloc[0]
    assert "CD3D" in row0["markers_used"]
    assert "CD3E" in row0["markers_used"]


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------


def test_to_csv(tmp_path):
    pytest.importorskip("pandas")
    result = _make_result()
    out = tmp_path / "annotations.csv"
    result.to_csv(out)
    assert out.exists()
    import pandas as pd
    df = pd.read_csv(out)
    assert len(df) == 3
    assert "predicted_type" in df.columns
