"""Tests for markers.py — no API calls required."""

import numpy as np
import pytest

from celltype_agent.markers import _sort_key, format_markers_for_prompt


def test_sort_key_numeric():
    labels = ["2", "10", "1", "0"]
    assert sorted(labels, key=_sort_key) == ["0", "1", "2", "10"]


def test_sort_key_mixed():
    labels = ["B_cell", "0", "1"]
    sorted_labels = sorted(labels, key=_sort_key)
    # numeric ones come first
    assert sorted_labels[:2] == ["0", "1"]
    assert sorted_labels[2] == "B_cell"


def test_format_markers_for_prompt_basic():
    markers = {
        "0": ["CD3D", "CD3E", "IL7R"],
        "1": ["MS4A1", "CD19", "CD79A"],
    }
    result = format_markers_for_prompt(markers, species="human")
    assert "CD3D" in result
    assert "MS4A1" in result
    assert "human" in result
    assert "| 0 |" in result
    assert "| 1 |" in result


def test_format_markers_for_prompt_with_tissue():
    markers = {"0": ["LYZ", "CD14"]}
    result = format_markers_for_prompt(markers, species="mouse", tissue="bone marrow")
    assert "bone marrow" in result
    assert "mouse" in result
