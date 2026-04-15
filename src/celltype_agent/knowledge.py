"""Marker gene knowledge databases.

Loads PanglaoDB and CellMarker databases on first use and exposes two query
functions that the annotation agent can call locally when Claude requests them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Module-level caches; populated by _ensure_loaded()
_panglaodb: pd.DataFrame | None = None
_cellmarker: pd.DataFrame | None = None

_DATA_DIR = Path(__file__).parent.parent.parent / "data"

_PANGLAO_PATH = _DATA_DIR / "PanglaoDB_markers_27_Mar_2020.tsv"
_CELLMARKER_PATH = _DATA_DIR / "Cell_marker_All.xlsx"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_panglaodb(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # Keep only the columns we need
    df = df[["species", "official gene symbol", "cell type", "organ"]].copy()
    df.columns = ["species_raw", "symbol", "cell_type", "organ"]
    df = df.dropna(subset=["symbol", "cell_type"])
    df["symbol"] = df["symbol"].str.strip()
    df["cell_type"] = df["cell_type"].str.strip()
    df["organ"] = df["organ"].fillna("").str.strip()
    # Derive normalised species: "human", "mouse", or "both"
    # PanglaoDB encodes: "Mm Hs" = both, "Hs" = human only, "Mm" = mouse only
    def _norm_species(s: str) -> str:
        s = str(s).strip()
        if s == "Hs":
            return "human"
        if s == "Mm":
            return "mouse"
        return "both"  # "Mm Hs"

    df["species"] = df["species_raw"].apply(_norm_species)
    return df


def _load_cellmarker(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df = df[["species", "tissue_type", "cell_name", "Symbol"]].copy()
    df.columns = ["species_raw", "tissue_type", "cell_type", "symbol"]
    df = df.dropna(subset=["symbol", "cell_type"])
    df["symbol"] = df["symbol"].str.strip()
    df["cell_type"] = df["cell_type"].str.strip()
    df["tissue_type"] = df["tissue_type"].fillna("").str.strip()
    # Normalise species: "Human" -> "human", "Mouse" -> "mouse"
    df["species"] = df["species_raw"].str.strip().str.lower()
    # Deduplicate by (species, cell_type, symbol)
    df = df.drop_duplicates(subset=["species", "cell_type", "symbol"])
    return df


def _ensure_loaded() -> None:
    global _panglaodb, _cellmarker
    if _panglaodb is None:
        log.info("Loading PanglaoDB from %s …", _PANGLAO_PATH)
        _panglaodb = _load_panglaodb(_PANGLAO_PATH)
        log.info("PanglaoDB loaded: %d rows.", len(_panglaodb))
    if _cellmarker is None:
        log.info("Loading CellMarker from %s …", _CELLMARKER_PATH)
        _cellmarker = _load_cellmarker(_CELLMARKER_PATH)
        log.info("CellMarker loaded: %d rows.", len(_cellmarker))


# ---------------------------------------------------------------------------
# Species filter helpers
# ---------------------------------------------------------------------------


def _panglao_species_mask(df: pd.DataFrame, species: str) -> pd.Series:
    """Return a boolean mask for rows matching *species* in PanglaoDB."""
    norm = species.lower().strip()
    if norm == "human":
        return df["species"].isin(["human", "both"])
    if norm == "mouse":
        return df["species"].isin(["mouse", "both"])
    # Unknown species: return all
    return pd.Series([True] * len(df), index=df.index)


def _cellmarker_species_mask(df: pd.DataFrame, species: str) -> pd.Series:
    norm = species.lower().strip()
    return df["species"] == norm


# ---------------------------------------------------------------------------
# Fuzzy cell-type matching
# ---------------------------------------------------------------------------


def _match_cell_types(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Return rows whose cell_type fuzzy-matches *query* (case-insensitive)."""
    q = query.lower().strip()
    ct_lower = df["cell_type"].str.lower()

    # 1. Exact match
    mask = ct_lower == q
    if mask.any():
        return df[mask]

    # 2. Substring match (query in cell_type OR cell_type in query)
    mask = ct_lower.str.contains(q, regex=False) | pd.Series(
        [q in ct for ct in ct_lower], index=df.index
    )
    if mask.any():
        return df[mask]

    # 3. Word-overlap: any word in the query appears in the cell_type
    words = set(q.split())
    mask = ct_lower.apply(lambda ct: bool(words & set(ct.split())))
    return df[mask]


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------


def search_by_celltype(cell_type: str, species: str) -> dict:
    """Return known marker genes for *cell_type* from both databases.

    Parameters
    ----------
    cell_type:
        Cell type name to look up (case-insensitive fuzzy match).
    species:
        ``"human"`` or ``"mouse"``.

    Returns
    -------
    dict with keys ``"panglaodb"`` and ``"cellmarker"``, each a list of gene
    symbol strings.
    """
    _ensure_loaded()

    # PanglaoDB
    pdb = _panglaodb[_panglao_species_mask(_panglaodb, species)]
    pdb_matches = _match_cell_types(pdb, cell_type)
    panglaodb_genes = sorted(pdb_matches["symbol"].unique().tolist())

    # CellMarker
    cdb = _cellmarker[_cellmarker_species_mask(_cellmarker, species)]
    cdb_matches = _match_cell_types(cdb, cell_type)
    cellmarker_genes = sorted(cdb_matches["symbol"].unique().tolist())

    log.debug(
        "search_by_celltype(%r, %r): panglaodb=%d genes, cellmarker=%d genes",
        cell_type, species, len(panglaodb_genes), len(cellmarker_genes),
    )
    return {"panglaodb": panglaodb_genes, "cellmarker": cellmarker_genes}


def search_by_gene(gene: str, species: str) -> dict:
    """Return cell types associated with *gene* from both databases.

    Parameters
    ----------
    gene:
        Gene symbol to look up (case-insensitive exact match).
    species:
        ``"human"`` or ``"mouse"``.

    Returns
    -------
    dict with keys ``"panglaodb"`` and ``"cellmarker"``, each a list of dicts
    with ``"cell_type"`` and ``"organ"``/``"tissue"`` keys.
    """
    _ensure_loaded()

    g = gene.strip().upper()

    # PanglaoDB
    pdb = _panglaodb[_panglao_species_mask(_panglaodb, species)]
    pdb_hits = pdb[pdb["symbol"].str.upper() == g]
    panglaodb_results = [
        {"cell_type": row["cell_type"], "organ": row["organ"]}
        for _, row in pdb_hits.drop_duplicates(subset=["cell_type", "organ"]).iterrows()
    ]

    # CellMarker
    cdb = _cellmarker[_cellmarker_species_mask(_cellmarker, species)]
    cdb_hits = cdb[cdb["symbol"].str.upper() == g]
    cellmarker_results = [
        {"cell_type": row["cell_type"], "tissue": row["tissue_type"]}
        for _, row in cdb_hits.drop_duplicates(subset=["cell_type", "tissue_type"]).iterrows()
    ]

    log.debug(
        "search_by_gene(%r, %r): panglaodb=%d hits, cellmarker=%d hits",
        gene, species, len(panglaodb_results), len(cellmarker_results),
    )
    return {"panglaodb": panglaodb_results, "cellmarker": cellmarker_results}
