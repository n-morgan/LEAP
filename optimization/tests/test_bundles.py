"""Hand-computed unit tests for each bundle-computation pure function.

Every input is set by hand and every expected output is computed by hand —
these tests must pass without any LLM, embedding, or fixture file.
"""

import math

import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon

from metrics import (
    ClassificationBundle,
    ExtractionBundle,
    HierarchyBundle,
    MatchedPair,
    MatchingResult,
    QualityBundle,
    compute_classification_bundle,
    compute_composite_score,
    compute_extraction_bundle,
    compute_hierarchy_bundle,
    compute_quality_bundle,
)


def _gt(stmt, cat="Mitigation", role="individual", parent_statement="",
        financial_instrument="", secondary_category=""):
    return {
        "policy_statement": stmt,
        "primary_category": cat,
        "role": role,
        "parent_statement": parent_statement,
        "financial_instrument": financial_instrument,
        "secondary_category": secondary_category,
    }


# ---------------------------------------------------------------------------
# Extraction bundle
# ---------------------------------------------------------------------------


def test_extraction_bundle_hand_computed():
    """4 GT, 3 extracted, 2 matched, 1 spurious extracted, 2 missed GT.

    TP=2, FP=1, FN=2 → precision=2/3, recall=2/4=0.5, F1=2*(0.667)*(0.5)/(0.667+0.5)
    Per-cat recall: Mitigation 1/2=0.5, Adaptation 1/2=0.5
    +1 coverage: 1/4 = 0.25
    """
    gt = [
        _gt("g1", "Mitigation"),
        _gt("g2", "Mitigation"),
        _gt("g3", "Adaptation"),
        _gt("g4", "Adaptation"),
    ]
    matched = [
        MatchedPair(
            extracted={"policy_statement": "e1", "primary_category": "Mitigation"},
            ground_truth=gt[0],
            similarity=0.9,
            grade=1,
        ),
        MatchedPair(
            extracted={"policy_statement": "e2", "primary_category": "Adaptation"},
            ground_truth=gt[2],
            similarity=0.8,
            grade=0,
        ),
    ]
    matching = MatchingResult(
        matched=matched,
        unmatched_extracted=[{"policy_statement": "spurious", "primary_category": "Mitigation"}],
        unmatched_gt=[gt[1], gt[3]],
        similarity_threshold=0.55,
    )

    bundle = compute_extraction_bundle(matching, gt)

    assert bundle.precision == pytest.approx(2 / 3)
    assert bundle.recall == pytest.approx(0.5)
    expected_f1 = 2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5)
    assert bundle.f1 == pytest.approx(expected_f1)

    assert bundle.per_category_recall["Mitigation"] == pytest.approx(0.5)
    assert bundle.per_category_recall["Adaptation"] == pytest.approx(0.5)

    # JSD: ext distribution (Mitigation=2/3, Adaptation=1/3)
    #      gt  distribution (Mitigation=2/4, Adaptation=2/4)
    expected_jsd = float(
        jensenshannon([2 / 3, 1 / 3], [0.5, 0.5], base=2)
    )
    assert bundle.category_distribution_jsd == pytest.approx(expected_jsd)

    assert bundle.plus_one_coverage == pytest.approx(1 / 4)


def test_extraction_bundle_jsd_extremes():
    """Identical distributions → 0; disjoint distributions → 1."""
    # Identical
    gt = [_gt("g1", "Mitigation"), _gt("g2", "Adaptation")]
    matched = [
        MatchedPair(
            extracted={"policy_statement": "e1", "primary_category": "Mitigation"},
            ground_truth=gt[0], similarity=0.9, grade=1,
        ),
        MatchedPair(
            extracted={"policy_statement": "e2", "primary_category": "Adaptation"},
            ground_truth=gt[1], similarity=0.9, grade=1,
        ),
    ]
    matching = MatchingResult(
        matched=matched, unmatched_extracted=[], unmatched_gt=[], similarity_threshold=0.55,
    )
    b = compute_extraction_bundle(matching, gt)
    assert b.category_distribution_jsd == pytest.approx(0.0, abs=1e-9)

    # Disjoint: ext is all Adaptation, GT is all Mitigation
    gt = [_gt("g1", "Mitigation")]
    matching = MatchingResult(
        matched=[],
        unmatched_extracted=[{"policy_statement": "e1", "primary_category": "Adaptation"}],
        unmatched_gt=gt,
        similarity_threshold=0.55,
    )
    b = compute_extraction_bundle(matching, gt)
    assert b.category_distribution_jsd == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Hierarchy bundle
# ---------------------------------------------------------------------------


def test_hierarchy_bundle_role_and_parent_attribution():
    """3 matched pairs:
        pair 0: parent (gt) - parent (ext)             role match
        pair 1: sub (gt, parent=g_par) - sub (ext, parent=e_par_correct)  role match,
                                                       and parent attributes correctly
        pair 2: sub (gt, parent=g_par) - individual (ext)  role MISMATCH

    role_agreement = 2/3
    parent_attribution: only pair 1 is eligible (both sides sub, gt parent in M)
                        → 1/1 = 1.0
    pair 2 excluded (extracted role != sub).
    """
    parent_pair_gt = {"policy_statement": "g_par", "role": "parent"}
    parent_pair_ext = {"policy_statement": "e_par", "role": "parent"}

    sub_pair_1_gt = {"policy_statement": "g_sub_1", "role": "sub", "parent_statement": "g_par"}
    sub_pair_1_ext = {"policy_statement": "e_sub_1", "role": "sub", "parent_statement": "e_par"}

    sub_pair_2_gt = {"policy_statement": "g_sub_2", "role": "sub", "parent_statement": "g_par"}
    sub_pair_2_ext = {"policy_statement": "e_sub_2", "role": "individual", "parent_statement": ""}

    matched = [
        MatchedPair(extracted=parent_pair_ext, ground_truth=parent_pair_gt, similarity=0.9),
        MatchedPair(extracted=sub_pair_1_ext, ground_truth=sub_pair_1_gt, similarity=0.9),
        MatchedPair(extracted=sub_pair_2_ext, ground_truth=sub_pair_2_gt, similarity=0.9),
    ]
    matching = MatchingResult(
        matched=matched, unmatched_extracted=[], unmatched_gt=[], similarity_threshold=0.55,
    )

    b = compute_hierarchy_bundle(matching)
    assert b.role_agreement == pytest.approx(2 / 3)
    assert b.parent_attribution_accuracy == pytest.approx(1.0)


def test_hierarchy_bundle_empty_match_defaults():
    matching = MatchingResult(
        matched=[], unmatched_extracted=[], unmatched_gt=[], similarity_threshold=0.55,
    )
    b = compute_hierarchy_bundle(matching)
    assert b.role_agreement == 1.0
    assert b.parent_attribution_accuracy == 1.0


# ---------------------------------------------------------------------------
# Classification bundle
# ---------------------------------------------------------------------------


def test_classification_bundle_hand_computed():
    """4 matched pairs:
       0: gt=Mitigation,  ext=Mitigation       primary correct
       1: gt=Mitigation,  ext=Adaptation       wrong
       2: gt=Adaptation,  ext=Adaptation       primary correct
       3: gt=Resource Efficiency, ext=Mitigation   wrong

    primary_category_agreement = 2/4 = 0.5
    confusion_matrix:
        Mitigation: {Mitigation: 1, Adaptation: 1}
        Adaptation: {Adaptation: 1}
        Resource Efficiency: {Mitigation: 1}
    """
    matched = [
        MatchedPair(
            extracted={"policy_statement": f"e{i}", "primary_category": ext, "financial_instrument": "yes", "secondary_category": "x"},
            ground_truth={"policy_statement": f"g{i}", "primary_category": gt, "financial_instrument": "yes", "secondary_category": "x"},
            similarity=0.9,
        )
        for i, (gt, ext) in enumerate([
            ("Mitigation", "Mitigation"),
            ("Mitigation", "Adaptation"),
            ("Adaptation", "Adaptation"),
            ("Resource Efficiency", "Mitigation"),
        ])
    ]
    matching = MatchingResult(
        matched=matched, unmatched_extracted=[], unmatched_gt=[], similarity_threshold=0.55,
    )

    b = compute_classification_bundle(matching)

    assert b.primary_category_agreement == pytest.approx(0.5)
    assert b.financial_instrument_agreement == pytest.approx(1.0)
    assert b.secondary_category_agreement == pytest.approx(1.0)

    cm = b.confusion_matrix
    assert cm["Mitigation"]["Mitigation"] == 1
    assert cm["Mitigation"]["Adaptation"] == 1
    assert cm["Adaptation"]["Adaptation"] == 1
    assert cm["Resource Efficiency"]["Mitigation"] == 1


# ---------------------------------------------------------------------------
# Quality bundle
# ---------------------------------------------------------------------------


def test_quality_bundle_hand_computed():
    """5 GT, 3 matched with grades [+1, +1, -1].
    +1 coverage = 2/5
    grade distribution: +1=2/3, 0=0, -1=1/3
    mean_grade = (1+1-1)/3 = 1/3
    """
    matched = [
        MatchedPair(extracted={}, ground_truth={}, similarity=0.9, grade=1),
        MatchedPair(extracted={}, ground_truth={}, similarity=0.9, grade=1),
        MatchedPair(extracted={}, ground_truth={}, similarity=0.9, grade=-1),
    ]
    matching = MatchingResult(
        matched=matched, unmatched_extracted=[], unmatched_gt=[], similarity_threshold=0.55,
    )
    gt = [{"policy_statement": f"g{i}"} for i in range(5)]

    b = compute_quality_bundle(matching, gt)
    assert b.plus_one_coverage == pytest.approx(2 / 5)
    assert b.grade_distribution["+1"] == pytest.approx(2 / 3)
    assert b.grade_distribution["0"] == pytest.approx(0.0)
    assert b.grade_distribution["-1"] == pytest.approx(1 / 3)
    assert b.mean_grade == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


def test_composite_score_default_weights():
    e = ExtractionBundle(
        f1=0.6, precision=0.7, recall=0.5,
        per_category_recall={}, category_distribution_jsd=0.1, plus_one_coverage=0.3,
    )
    h = HierarchyBundle(role_agreement=0.8, parent_attribution_accuracy=0.4)
    c = ClassificationBundle(
        primary_category_agreement=0.7, confusion_matrix={},
        financial_instrument_agreement=0.9, secondary_category_agreement=0.6,
    )
    q = QualityBundle(
        plus_one_coverage=0.4,
        grade_distribution={"+1": 0.4, "0": 0.4, "-1": 0.2},
        mean_grade=0.2,
    )
    expected = 0.40 * 0.6 + 0.25 * ((0.8 + 0.4) / 2) + 0.25 * 0.7 + 0.10 * 0.4
    assert compute_composite_score(e, h, c, q) == pytest.approx(expected)
