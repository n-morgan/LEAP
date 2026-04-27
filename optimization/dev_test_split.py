"""dev_test_split.py — location set abstraction for the redesigned LEAP loop.

A ``LocationSet`` carries the dev locations used for acceptance decisions and
the held-out test locations evaluated once at loop end. Each location owns its
own ground-truth CSV + source-document path so the loop can run a candidate
prompt across multiple cities and aggregate the resulting bundles.

Locations are described in a small YAML file at ``optimization/locations.yaml``
with the schema:

    location_column: location          # optional, default "location"
    locations:
      - name: Seattle_US
        ground_truth_csv: organized_outputs/structured_policies.csv
        source_document_md: ../GENIUS/docs/cities/seattle_markdown.md
        split: dev
      - name: Boston_US
        ground_truth_csv: organized_outputs/structured_policies.csv
        source_document_md: ../GENIUS/docs/cities/boston_markdown.md
        split: test

Each location's ``name`` is matched **verbatim** against the
``location_column`` value of every row in ``ground_truth_csv``. Only matching
rows are loaded, so a single shared multi-city CSV can drive multiple
``LocationConfig`` entries.

Paths are resolved relative to the YAML file's parent directory.
"""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional


# Default name of the column in the ground-truth CSV whose value identifies
# the city / location. Override per-LocationSet by passing a different value
# in locations.yaml's top-level ``location_column`` key.
DEFAULT_LOCATION_COLUMN: str = "location"


@dataclass
class LocationConfig:
    name: str
    ground_truth_csv: pathlib.Path
    source_document_md: Optional[pathlib.Path] = None
    split: Literal["dev", "test"] = "dev"
    location_column: str = DEFAULT_LOCATION_COLUMN

    def load_ground_truth(self) -> list[dict]:
        """Load GT rows for this location, filtering by exact-string match on
        ``location_column`` against ``self.name``.

        If the CSV has no ``location_column``, raises a clear error so the
        config issue is loud rather than silently degrading to all-rows.
        """
        with open(self.ground_truth_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or self.location_column not in reader.fieldnames:
                raise ValueError(
                    f"Ground-truth CSV {self.ground_truth_csv} is missing the "
                    f"'{self.location_column}' column required to filter by "
                    f"location='{self.name}'. Available columns: {reader.fieldnames}."
                )
            rows = [
                row for row in reader
                if row.get(self.location_column, "") == self.name
                and (row.get("policy_statement") or "").strip()
            ]
        if not rows:
            raise ValueError(
                f"No ground-truth rows found in {self.ground_truth_csv} where "
                f"'{self.location_column}' == '{self.name}'. Check the YAML "
                f"name matches the CSV value exactly (case-sensitive)."
            )
        return rows


@dataclass
class LocationSet:
    dev: list[LocationConfig] = field(default_factory=list)
    test: list[LocationConfig] = field(default_factory=list)

    @classmethod
    def single(
        cls,
        name: str,
        ground_truth_csv: pathlib.Path | str,
        source_document_md: Optional[pathlib.Path | str] = None,
        location_column: str = DEFAULT_LOCATION_COLUMN,
    ) -> "LocationSet":
        """Convenience constructor for a one-location dev/no-test setup."""
        return cls(
            dev=[
                LocationConfig(
                    name=name,
                    ground_truth_csv=pathlib.Path(ground_truth_csv),
                    source_document_md=(
                        pathlib.Path(source_document_md)
                        if source_document_md is not None else None
                    ),
                    split="dev",
                    location_column=location_column,
                )
            ],
            test=[],
        )

    def all(self) -> Iterable[LocationConfig]:
        yield from self.dev
        yield from self.test


def load_locations_from_yaml(path: pathlib.Path | str) -> LocationSet:
    """Load a LocationSet from a YAML file.

    Falls back to a tiny manual YAML parser when ``pyyaml`` is not installed,
    so the dependency footprint remains zero (per the implementation plan).
    """
    path = pathlib.Path(path)
    base = path.parent
    text = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
    except ImportError:
        data = _parse_minimal_yaml(text)

    if not isinstance(data, dict):
        return LocationSet(dev=[], test=[])

    location_column = data.get("location_column") or DEFAULT_LOCATION_COLUMN
    raw = data.get("locations", [])
    dev: list[LocationConfig] = []
    test: list[LocationConfig] = []
    for entry in raw:
        gt_path = (base / entry["ground_truth_csv"]).resolve()
        src_path = (
            (base / entry["source_document_md"]).resolve()
            if entry.get("source_document_md") else None
        )
        cfg = LocationConfig(
            name=entry["name"],
            ground_truth_csv=gt_path,
            source_document_md=src_path,
            split=entry.get("split", "dev"),
            location_column=entry.get("location_column", location_column),
        )
        (dev if cfg.split == "dev" else test).append(cfg)
    return LocationSet(dev=dev, test=test)


def _parse_minimal_yaml(text: str) -> dict:
    """Tiny YAML subset parser sufficient for the locations.yaml schema.

    Supports:
      * a top-level scalar ``location_column: <value>``
      * a top-level key ``locations:`` containing a list of mappings with
        simple string values.
    No nested structures, no anchors, no flow-style.
    """
    result: dict = {"locations": []}
    current: Optional[dict] = None
    in_locations = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("locations:"):
            in_locations = True
            continue
        # Top-level scalars (no leading whitespace, not inside locations list)
        if not in_locations and not line.startswith(" ") and ":" in line:
            k, _, v = line.partition(":")
            result[k.strip()] = v.strip()
            continue
        stripped = line.lstrip()
        if stripped.startswith("- "):
            current = {}
            result["locations"].append(current)
            kv = stripped[2:]
            if ":" in kv:
                k, _, v = kv.partition(":")
                current[k.strip()] = v.strip()
        elif current is not None and ":" in stripped:
            k, _, v = stripped.partition(":")
            current[k.strip()] = v.strip()
    return result
