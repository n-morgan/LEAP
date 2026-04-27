"""Tests for the verbatim ``location_column`` filter in ``LocationConfig``.

A single shared multi-city CSV must be split by exact-string match against
the YAML ``name``. Missing column or no matching rows must raise loudly.
"""

import csv
import pathlib

import pytest

from dev_test_split import LocationConfig, load_locations_from_yaml


def _write_csv(tmp_path: pathlib.Path, rows: list[dict], fieldnames: list[str]) -> pathlib.Path:
    p = tmp_path / "shared.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return p


def test_filter_returns_only_matching_rows(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        rows=[
            {"location": "Seattle_US",  "policy_statement": "S1"},
            {"location": "Seattle_US",  "policy_statement": "S2"},
            {"location": "Boston_US",   "policy_statement": "B1"},
            {"location": "Portland_US", "policy_statement": "P1"},
        ],
        fieldnames=["location", "policy_statement"],
    )

    cfg = LocationConfig(name="Seattle_US", ground_truth_csv=csv_path)
    rows = cfg.load_ground_truth()
    assert [r["policy_statement"] for r in rows] == ["S1", "S2"]


def test_filter_is_case_sensitive(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        rows=[{"location": "Seattle_US", "policy_statement": "X"}],
        fieldnames=["location", "policy_statement"],
    )
    cfg = LocationConfig(name="seattle_us", ground_truth_csv=csv_path)
    with pytest.raises(ValueError, match="No ground-truth rows found"):
        cfg.load_ground_truth()


def test_skips_rows_with_empty_policy_statement(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        rows=[
            {"location": "X", "policy_statement": "ok"},
            {"location": "X", "policy_statement": "  "},
            {"location": "X", "policy_statement": ""},
        ],
        fieldnames=["location", "policy_statement"],
    )
    cfg = LocationConfig(name="X", ground_truth_csv=csv_path)
    rows = cfg.load_ground_truth()
    assert len(rows) == 1


def test_missing_location_column_raises(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        rows=[{"policy_statement": "X"}],
        fieldnames=["policy_statement"],
    )
    cfg = LocationConfig(name="anything", ground_truth_csv=csv_path)
    with pytest.raises(ValueError, match="missing the 'location' column"):
        cfg.load_ground_truth()


def test_custom_location_column(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        rows=[
            {"city": "Seattle_US", "policy_statement": "S"},
            {"city": "Boston_US",  "policy_statement": "B"},
        ],
        fieldnames=["city", "policy_statement"],
    )
    cfg = LocationConfig(name="Boston_US", ground_truth_csv=csv_path, location_column="city")
    rows = cfg.load_ground_truth()
    assert [r["policy_statement"] for r in rows] == ["B"]


def test_load_locations_from_yaml_propagates_location_column(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        rows=[
            {"city": "Seattle_US", "policy_statement": "S"},
            {"city": "Boston_US",  "policy_statement": "B"},
        ],
        fieldnames=["city", "policy_statement"],
    )
    yaml_path = tmp_path / "locations.yaml"
    yaml_path.write_text(
        "location_column: city\n"
        "locations:\n"
        f"  - name: Seattle_US\n"
        f"    ground_truth_csv: {csv_path.name}\n"
        f"    split: dev\n"
        f"  - name: Boston_US\n"
        f"    ground_truth_csv: {csv_path.name}\n"
        f"    split: test\n",
        encoding="utf-8",
    )
    locset = load_locations_from_yaml(yaml_path)
    assert [c.name for c in locset.dev] == ["Seattle_US"]
    assert [c.name for c in locset.test] == ["Boston_US"]
    assert locset.dev[0].location_column == "city"
    assert [r["policy_statement"] for r in locset.dev[0].load_ground_truth()] == ["S"]
    assert [r["policy_statement"] for r in locset.test[0].load_ground_truth()] == ["B"]
