"""
data/dataset.py — LEAP Benchmark Corpus

Defines the nine-city climate policy corpus used in the LEAP benchmark.
Each city entry maps to:
  - a location key  (used as the evaluation identifier)
  - a source document  (markdown-converted climate policy PDF)
  - ground-truth policies  (GENIUS-annotated rows from the master CSV)

Ground-truth CSV
----------------
Produced by the GENIUS pipeline (see ../../../GENIUS/). Each row is one
policy with fields: policy_statement, role, primary_category,
parent_statement, is_financial_instrument, secondary_category, verbatim_text.

Adding a new city
-----------------
1. Add a markdown document to docs/cities/.
2. Add a row to CORPUS below.
3. Ensure ground-truth rows exist in GT_CSV with a matching ``city`` value.
"""

from __future__ import annotations

import csv
import pathlib
from typing import Any

_HERE     = pathlib.Path(__file__).resolve().parent          # benchmark/data/
_BENCH    = _HERE.parent                                     # benchmark/
_OPT      = _BENCH.parent / "optimization"                  # optimization/
_DOCS     = _OPT / "docs" / "cities"

GT_CSV: pathlib.Path = (
    _BENCH.parent.parent                                     # UROP_Climate_SP26/
    / "GENIUS" / "notebooks" / "outputs"
    / "all_cities_kept_classified_policies_final.csv"
)


# ---------------------------------------------------------------------------
# Corpus definition
# ---------------------------------------------------------------------------

CORPUS: dict[str, dict[str, Any]] = {
    "Austin": {
        "location_key": "Austin_US",
        "document":     _DOCS / "austin.md",
    },
    "Chicago": {
        "location_key": "Chicago_US",
        "document":     _DOCS / "chicago.md",
    },
    "Dakar": {
        "location_key": "Dakar_SN",
        "document":     _DOCS / "dakar.md",
    },
    "Geneva": {
        "location_key": "Geneva_CH",
        "document":     _DOCS / "geneva.md",
    },
    "Hiroshima": {
        "location_key": "Hiroshima_JP",
        "document":     _DOCS / "Hiroshima.md",
    },
    "Kuwait": {
        "location_key": "Kuwait_KW",
        "document":     _DOCS / "kuwait.md",
    },
    "Miami_Dade": {
        "location_key": "Miami_Dade_US",
        "document":     _DOCS / "miami_markdown.md",
    },
    "Portugal": {
        "location_key": "Portugal_PT",
        "document":     _DOCS / "Portugal.md",
    },
    "Seattle": {
        "location_key": "Seattle_US",
        "document":     _DOCS / "seattle_markdown.md",
    },
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ground_truth(city: str) -> list[dict[str, Any]]:
    """Return ground-truth policy rows for one city.

    Args:
        city: Key from ``CORPUS``, e.g. ``"Seattle"``.

    Returns:
        List of dicts (one per policy row), filtered to non-empty
        ``policy_statement`` values.
    """
    if city not in CORPUS:
        raise ValueError(f"Unknown city {city!r}. Valid: {sorted(CORPUS)}")
    rows: list[dict[str, Any]] = []
    with open(GT_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("city") == city and row.get("policy_statement", "").strip():
                rows.append(row)
    return rows


def load_ground_truth_all() -> dict[str, list[dict[str, Any]]]:
    """Return ground-truth rows for all corpus cities, grouped by city name."""
    gt: dict[str, list[dict[str, Any]]] = {c: [] for c in CORPUS}
    with open(GT_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            city = row.get("city", "")
            if city in gt and row.get("policy_statement", "").strip():
                gt[city].append(row)
    return gt
