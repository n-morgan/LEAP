"""Per-prong trigger logic — eight scenarios across {extraction, hierarchy,
classification} x {worsened, stagnant-bad, stagnant-good, improved}.

Plus a few sanity tests on iteration-1 bootstrap and ordering.
"""

from typing import Optional

import pytest

from metrics import (
    ClassificationBundle,
    EvaluationBundle,
    ExtractionBundle,
    HierarchyBundle,
    MatchingResult,
    QualityBundle,
)
from prompt_optimizer import PrognTarget, _build_resample_user_v2, _triggered_prongs, ordered_triggered


def _bundle(f1=0.7, role=0.85, parent_attr=0.85, primary=0.85) -> EvaluationBundle:
    return EvaluationBundle(
        location="test",
        matching=MatchingResult(
            matched=[], unmatched_extracted=[], unmatched_gt=[], similarity_threshold=0.55,
        ),
        extraction=ExtractionBundle(
            f1=f1, precision=f1, recall=f1,
            per_category_recall={}, category_distribution_jsd=0.0, plus_one_coverage=0.0,
        ),
        hierarchy=HierarchyBundle(role_agreement=role, parent_attribution_accuracy=parent_attr),
        classification=ClassificationBundle(
            primary_category_agreement=primary, confusion_matrix={},
            financial_instrument_agreement=1.0, secondary_category_agreement=1.0,
        ),
        quality=QualityBundle(
            plus_one_coverage=0.0, grade_distribution={"+1": 0.0, "0": 0.0, "-1": 0.0},
            mean_grade=0.0,
        ),
        composite_score=0.0,
    )


def test_iteration_1_bootstraps_all_prongs():
    triggered = _triggered_prongs(_bundle(), None, PrognTarget())
    assert triggered == {"extraction", "hierarchy", "classification"}


def test_extraction_worsened():
    cur = _bundle(f1=0.6)
    prev = _bundle(f1=0.8)
    triggered = _triggered_prongs(cur, prev, PrognTarget())
    assert "extraction" in triggered


def test_extraction_stagnant_bad():
    """Same value but below target → still triggered."""
    cur = _bundle(f1=0.5)
    prev = _bundle(f1=0.5)
    triggered = _triggered_prongs(cur, prev, PrognTarget())
    assert "extraction" in triggered


def test_extraction_stagnant_good_not_triggered():
    cur = _bundle(f1=0.9)
    prev = _bundle(f1=0.9)
    triggered = _triggered_prongs(cur, prev, PrognTarget())
    assert "extraction" not in triggered


def test_extraction_improved_above_target_not_triggered():
    cur = _bundle(f1=0.92)
    prev = _bundle(f1=0.85)
    triggered = _triggered_prongs(cur, prev, PrognTarget())
    assert "extraction" not in triggered


def test_hierarchy_worsened_role():
    cur = _bundle(role=0.6)
    prev = _bundle(role=0.9)
    triggered = _triggered_prongs(cur, prev, PrognTarget())
    assert "hierarchy" in triggered


def test_hierarchy_stagnant_bad():
    cur = _bundle(role=0.5, parent_attr=0.5)
    prev = _bundle(role=0.5, parent_attr=0.5)
    triggered = _triggered_prongs(cur, prev, PrognTarget())
    assert "hierarchy" in triggered


def test_hierarchy_stagnant_good_not_triggered():
    cur = _bundle(role=0.95, parent_attr=0.95)
    prev = _bundle(role=0.95, parent_attr=0.95)
    triggered = _triggered_prongs(cur, prev, PrognTarget())
    assert "hierarchy" not in triggered


def test_classification_worsened():
    cur = _bundle(primary=0.5)
    prev = _bundle(primary=0.9)
    triggered = _triggered_prongs(cur, prev, PrognTarget())
    assert "classification" in triggered


def test_classification_stagnant_bad():
    cur = _bundle(primary=0.6)
    prev = _bundle(primary=0.6)
    triggered = _triggered_prongs(cur, prev, PrognTarget())
    assert "classification" in triggered


def test_classification_stagnant_good_not_triggered():
    cur = _bundle(primary=0.95)
    prev = _bundle(primary=0.95)
    triggered = _triggered_prongs(cur, prev, PrognTarget())
    assert "classification" not in triggered


def test_no_triggers_when_all_above_targets_and_stable():
    cur = _bundle(f1=0.92, role=0.92, parent_attr=0.92, primary=0.92)
    prev = _bundle(f1=0.92, role=0.92, parent_attr=0.92, primary=0.92)
    assert _triggered_prongs(cur, prev, PrognTarget()) == set()


def test_ordered_triggered_priority():
    assert ordered_triggered({"classification", "extraction"}) == ["extraction", "classification"]
    assert ordered_triggered({"hierarchy"}) == ["hierarchy"]
    assert ordered_triggered(set()) == []
    assert ordered_triggered({"classification", "hierarchy", "extraction"}) == [
        "extraction", "hierarchy", "classification",
    ]


def test_build_resample_user_v2_allows_braces_in_user_content():
    """Regression: str.format on a template could KeyError on {n} or {json}."""
    msg = _build_resample_user_v2(
        scope="extraction",
        headline_name="extraction.f1",
        headline_value=0.5,
        target=0.7,
        delta=-0.1,
        supporting="x",
        section='Output JSON like {"n": 1} and TeX {n} here.',
        failure_examples="MISSED:\n  - {weird: brace}",
    )
    assert '{"n": 1}' in msg
    assert "{weird: brace}" in msg
    assert "PRONG: extraction" in msg
