"""
data/dataset.py — LEAP Benchmark Corpus

All data files are colocated in this directory (self-contained):
  all_cities_kept_classified_policies_final.csv  — ground-truth policies
  hex_location_to_final_run_csv.csv              — authoritative hex -> city_key mapping
  pdf_markdown_cache/{hex}.md                    — pre-converted PDF markdown

The CORPUS is derived from hex_location_to_final_run_csv.csv. Only entries
whose city_key appears in the GT CSV are included. Three entries are excluded:
  - Budapest_Hungary_final_artemis_report_hungary  (no GT rows)
  - Geneva_Switzerland                             (no GT rows)
  - Los_Angeles_San_Diego_California_United_States (no GT rows)

Adding a new city
-----------------
1. Add the converted markdown to pdf_markdown_cache/.
2. Add a row to hex_location_to_final_run_csv.csv.
3. Ensure ground-truth rows exist in GT_CSV with a matching ``city`` value.
"""

from __future__ import annotations

import csv
import json
import pathlib
from typing import Any

_HERE  = pathlib.Path(__file__).resolve().parent   # benchmark/data/
_CACHE = _HERE / "pdf_markdown_cache"

GT_CSV:      pathlib.Path = _HERE / "all_cities_kept_classified_policies_final.csv"
HEX_MAP_CSV: pathlib.Path = _HERE / "hex_location_to_final_run_csv.csv"


# ---------------------------------------------------------------------------
# Corpus definition
# ---------------------------------------------------------------------------

CORPUS: dict[str, dict[str, Any]] = {
    "Armenia": {
        "location_key": "Armenia_AM",
        "document":     _CACHE / "463494f6142f3c95.md",
        "gt_city":      ["Armenia"],
    },
    "Austin": {
        "location_key": "Austin_US",
        "document":     _CACHE / "863f59e62488663e.md",
        "gt_city":      ["Austin_Texas_United_States"],
    },
    "Bahrain": {
        "location_key": "Bahrain_BH",
        "document":     _CACHE / "6d8e5d823a4a9453.md",
        "gt_city":      ["Bahrain"],
    },
    "Beirut": {
        "location_key": "Beirut_LB",
        "document":     _CACHE / "65771b51b5230de2.md",
        "gt_city":      ["Beirut_Lebanon"],
    },
    "Brussels": {
        "location_key": "Brussels_BE",
        "document":     _CACHE / "833fc42e36139927.md",
        "gt_city":      ["Brussles_Belgium"],
    },
    "Budapest": {
        "location_key": "Budapest_HU",
        "document":     _CACHE / "55b806dece1683a6.md",
        "gt_city":      ["Budapest_Hungary"],
    },
    "California": {
        "location_key": "California_US",
        "document":     _CACHE / "f764cece0453ddf2.md",
        "gt_city":      ["California_United_States"],
    },
    "Chicago": {
        "location_key": "Chicago_US",
        "document":     _CACHE / "3d7c0632daeecab3.md",
        "gt_city":      ["Chicago_IL_US"],
    },
    "Chile": {
        "location_key": "Chile_CL",
        "document":     _CACHE / "d174943ca7611d54.md",
        "gt_city":      ["Chile"],
    },
    "Colorado_Springs": {
        "location_key": "Colorado_Springs_US",
        "document":     _CACHE / "7c80389d0b18db12.md",
        "gt_city":      ["Colorado_Springs_CO_United_States"],
    },
    "Columbus": {
        "location_key": "Columbus_US",
        "document":     _CACHE / "8fbd410b4410c12d.md",
        "gt_city":      ["Columbus_Ohio_United_States"],
    },
    "Egypt": {
        "location_key": "Egypt_EG",
        "document":     _CACHE / "b40b752fab51914c.md",
        "gt_city":      ["Egypt"],
    },
    "Fiji": {
        "location_key": "Fiji_FJ",
        "document":     _CACHE / "346b4256a7281c1e.md",
        "gt_city":      ["Fiji"],
    },
    "Helsinki": {
        "location_key": "Helsinki_FI",
        "document":     _CACHE / "310520867abd5fef.md",
        "gt_city":      ["Helsinki_Finland"],
    },
    "Hiroshima": {
        "location_key": "Hiroshima_JP",
        "document":     _CACHE / "a27d2619f09918ac.md",
        "gt_city":      ["Hiroshima_Japan"],
    },
    "Hong_Kong": {
        "location_key": "Hong_Kong_HK",
        "document":     _CACHE / "baf5c3a49a7ca728.md",
        "gt_city":      ["Hong_Kong_China"],
    },
    "Indonesia": {
        "location_key": "Indonesia_ID",
        "document":     _CACHE / "91f9a7e4d651792b.md",
        "gt_city":      ["Indonesia"],
    },
    "Italy": {
        "location_key": "Italy_IT",
        "document":     _CACHE / "7360a9b818f89f3e.md",
        "gt_city":      ["Italy"],
    },
    "Japan": {
        "location_key": "Japan_JP",
        "document":     _CACHE / "6e1ff8569e096155.md",
        "gt_city":      ["Japan"],
    },
    "Kuwait": {
        "location_key": "Kuwait_KW",
        "document":     _CACHE / "5e582fc2e4eef1a5.md",
        "gt_city":      ["Kuwait"],
    },
    "Las_Vegas": {
        "location_key": "Las_Vegas_US",
        "document":     _CACHE / "122052fa04cd0942.md",
        "gt_city":      ["Las_Vegas_Nevada_United_States"],
    },
    "Liberia": {
        "location_key": "Liberia_LR",
        "document":     _CACHE / "ff4fb26c29f23769.md",
        "gt_city":      ["Liberia"],
    },
    "London": {
        "location_key": "London_GB",
        "document":     _CACHE / "e0bd5e822b42e7a2.md",
        "gt_city":      ["London_England"],
    },
    "Los_Angeles": {
        "location_key": "Los_Angeles_US",
        "document":     _CACHE / "5d09e0b1b86e4a5e.md",
        "gt_city":      ["Los_Angeles_California_United_States"],
    },
    "Luxor": {
        "location_key": "Luxor_EG",
        "document":     _CACHE / "48348062028a742b.md",
        "gt_city":      ["Luxor_Egypt"],
    },
    "Memphis": {
        "location_key": "Memphis_US",
        "document":     _CACHE / "a4aef7ca58f961cf.md",
        "gt_city":      ["Memphis_Tenesse_United_States"],
    },
    "Miami_Dade": {
        "location_key": "Miami_Dade_US",
        "document":     _CACHE / "a0e3986face5e158.md",
        "gt_city":      ["Miami_Dade_Florida_United_States"],
    },
    "Namibia": {
        "location_key": "Namibia_NA",
        "document":     _CACHE / "2b8c363d2c01e9f1.md",
        "gt_city":      ["Nambia"],
    },
    "New_York": {
        "location_key": "New_York_US",
        "document":     _CACHE / "eb1ff5864431e20c.md",
        "gt_city":      ["New_York_United_States"],
    },
    "New_Zealand": {
        "location_key": "New_Zealand_NZ",
        "document":     _CACHE / "67cabbe9e748b945.md",
        "gt_city":      ["New_Zealand"],
    },
    "Perugia": {
        "location_key": "Perugia_IT",
        "document":     _CACHE / "a174f2c0fc5e2d68.md",
        "gt_city":      ["Perugia_Italy"],
    },
    "Portugal": {
        "location_key": "Portugal_PT",
        "document":     _CACHE / "891bfd541d2bf60f.md",
        "gt_city":      ["Portugal"],
    },
    "Qatar": {
        "location_key": "Qatar_QA",
        "document":     _CACHE / "b2d85c3458f830e2.md",
        "gt_city":      ["Qatar"],
    },
    "San_Francisco": {
        "location_key": "San_Francisco_US",
        "document":     _CACHE / "3ef128ce4832a0e7.md",
        "gt_city":      ["San_Francisco_California_United_States"],
    },
    "Santiago": {
        "location_key": "Santiago_CL",
        "document":     _CACHE / "5912b78282296af3.md",
        "gt_city":      ["Santiago_Chile"],
    },
    "Seattle": {
        "location_key": "Seattle_US",
        "document":     _CACHE / "cd660feb69d3dfd0.md",
        "gt_city":      ["Seattle_WA_US"],
    },
    "Singapore": {
        "location_key": "Singapore_SG",
        "document":     _CACHE / "9feb9d6f872b5cb4.md",
        "gt_city":      ["Singapore"],
    },
    "Stockholm": {
        "location_key": "Stockholm_SE",
        "document":     _CACHE / "79ce08f80f170146.md",
        "gt_city":      ["Stockholm_Sweden"],
    },
    "Tokyo": {
        "location_key": "Tokyo_JP",
        "document":     _CACHE / "d661f9335059b9e1.md",
        "gt_city":      ["Tokyo_Japan"],
    },
    "Utah": {
        "location_key": "Utah_US",
        "document":     _CACHE / "275de8caf42e3a48.md",
        "gt_city":      ["Utah_United_States"],
    },
    "Waterloo": {
        "location_key": "Waterloo_CA",
        "document":     _CACHE / "50b11d4c74c02321.md",
        "gt_city":      ["Waterloo_Ontario_Canada"],
    },
    "Yanbu": {
        "location_key": "Yanbu_SA",
        "document":     _CACHE / "24e188c799ad7fd3.md",
        "gt_city":      ["Yanbu_Saudi_Arabia"],
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
    gt_cities = set(CORPUS[city]["gt_city"])
    rows: list[dict[str, Any]] = []
    with open(GT_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("city") in gt_cities and row.get("policy_statement", "").strip():
                rows.append(row)
    return rows


def load_ground_truth_all() -> dict[str, list[dict[str, Any]]]:
    """Return ground-truth rows for all corpus cities, grouped by corpus key."""
    gt_city_to_corpus: dict[str, str] = {}
    for corpus_key, cfg in CORPUS.items():
        for gt_city in cfg["gt_city"]:
            gt_city_to_corpus[gt_city] = corpus_key

    gt: dict[str, list[dict[str, Any]]] = {c: [] for c in CORPUS}
    with open(GT_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            city = row.get("city", "")
            corpus_key = gt_city_to_corpus.get(city)
            if corpus_key and row.get("policy_statement", "").strip():
                gt[corpus_key].append(row)
    return gt
