"""Tests for knowledge.py — database lookups, no API calls required."""

from __future__ import annotations

import pytest

from celltype_agent.knowledge import search_by_celltype, search_by_gene


class TestSearchByGene:
    def test_cd79a_returns_b_cell(self):
        result = search_by_gene("CD79A", "human")
        all_cell_types = (
            [r["cell_type"].lower() for r in result["panglaodb"]]
            + [r["cell_type"].lower() for r in result["cellmarker"]]
        )
        assert any("b cell" in ct or "b-cell" in ct for ct in all_cell_types), (
            f"Expected a B cell entry for CD79A but got: {all_cell_types[:10]}"
        )

    def test_cd3d_returns_t_cell(self):
        result = search_by_gene("CD3D", "human")
        all_cell_types = (
            [r["cell_type"].lower() for r in result["panglaodb"]]
            + [r["cell_type"].lower() for r in result["cellmarker"]]
        )
        assert any("t cell" in ct or "t-cell" in ct for ct in all_cell_types), (
            f"Expected a T cell entry for CD3D but got: {all_cell_types[:10]}"
        )

    def test_returns_both_databases(self):
        result = search_by_gene("CD14", "human")
        assert "panglaodb" in result
        assert "cellmarker" in result
        assert isinstance(result["panglaodb"], list)
        assert isinstance(result["cellmarker"], list)

    def test_each_hit_has_cell_type(self):
        result = search_by_gene("LYZ", "human")
        for hit in result["panglaodb"]:
            assert "cell_type" in hit
            assert "organ" in hit
        for hit in result["cellmarker"]:
            assert "cell_type" in hit
            assert "tissue" in hit

    def test_mouse_species(self):
        result = search_by_gene("Cd3d", "mouse")
        # Gene lookup is case-insensitive via .upper()
        all_cell_types = (
            [r["cell_type"].lower() for r in result["panglaodb"]]
            + [r["cell_type"].lower() for r in result["cellmarker"]]
        )
        assert any("t cell" in ct or "t-cell" in ct for ct in all_cell_types), (
            f"Expected T cell hits for Cd3d (mouse) but got: {all_cell_types[:10]}"
        )

    def test_unknown_gene_returns_empty_lists(self):
        result = search_by_gene("NOTAREALGENEXYZ123", "human")
        assert result["panglaodb"] == []
        assert result["cellmarker"] == []


class TestSearchByCelltype:
    def test_b_cell_returns_cd79a(self):
        result = search_by_celltype("B cell", "human")
        all_genes = set(result["panglaodb"]) | set(result["cellmarker"])
        assert "CD79A" in all_genes, (
            f"Expected CD79A among B cell markers but got: {sorted(all_genes)[:20]}"
        )

    def test_b_cell_returns_ms4a1(self):
        result = search_by_celltype("B cell", "human")
        all_genes = set(result["panglaodb"]) | set(result["cellmarker"])
        assert "MS4A1" in all_genes, (
            f"Expected MS4A1 (CD20) among B cell markers but got: {sorted(all_genes)[:20]}"
        )

    def test_t_cell_returns_cd3_genes(self):
        result = search_by_celltype("T cell", "human")
        all_genes = set(result["panglaodb"]) | set(result["cellmarker"])
        t_cell_markers = {"CD3D", "CD3E", "CD3G"}
        assert all_genes & t_cell_markers, (
            f"Expected at least one CD3 gene for T cells but got: {sorted(all_genes)[:20]}"
        )

    def test_returns_both_database_keys(self):
        result = search_by_celltype("Monocyte", "human")
        assert "panglaodb" in result
        assert "cellmarker" in result
        assert isinstance(result["panglaodb"], list)
        assert isinstance(result["cellmarker"], list)

    def test_case_insensitive(self):
        result_upper = search_by_celltype("B CELL", "human")
        result_lower = search_by_celltype("b cell", "human")
        assert set(result_upper["panglaodb"]) == set(result_lower["panglaodb"])

    def test_partial_match(self):
        # "NK" should match "NK cell", "Natural killer cell", etc.
        result = search_by_celltype("NK cell", "human")
        all_genes = set(result["panglaodb"]) | set(result["cellmarker"])
        assert len(all_genes) > 0, "Expected NK cell markers from at least one database"

    def test_unknown_celltype_returns_empty_lists(self):
        result = search_by_celltype("NOTAREALCELLTYPEXYZ123", "human")
        assert result["panglaodb"] == []
        assert result["cellmarker"] == []
