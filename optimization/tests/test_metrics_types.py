"""Round-trip tests for the metric Pydantic models defined in metrics.py."""

from metrics import (
    ClassificationBundle,
    EvaluationBundle,
    ExtractionBundle,
    HierarchyBundle,
    MatchedPair,
    MatchingResult,
    QualityBundle,
)


def _example_matched_pair() -> MatchedPair:
    return MatchedPair(
        extracted={"policy_statement": "ext stmt", "primary_category": "Mitigation"},
        ground_truth={"policy_statement": "gt stmt", "primary_category": "Mitigation"},
        similarity=0.91,
        grade=1,
        reasoning="solid match",
        statement_match="match",
        role_match=True,
        category_match=True,
    )


def _example_matching_result() -> MatchingResult:
    return MatchingResult(
        matched=[_example_matched_pair()],
        unmatched_extracted=[{"policy_statement": "spurious"}],
        unmatched_gt=[{"policy_statement": "missed"}],
        similarity_threshold=0.55,
    )


def test_matched_pair_roundtrip():
    pair = _example_matched_pair()
    raw = pair.model_dump_json()
    rehydrated = MatchedPair.model_validate_json(raw)
    assert rehydrated == pair


def test_matching_result_roundtrip():
    mr = _example_matching_result()
    raw = mr.model_dump_json()
    rehydrated = MatchingResult.model_validate_json(raw)
    assert rehydrated == mr


def test_extraction_bundle_roundtrip():
    eb = ExtractionBundle(
        f1=0.6,
        precision=0.7,
        recall=0.5,
        per_category_recall={"Mitigation": 0.5, "Adaptation": 0.5},
        category_distribution_jsd=0.1,
        plus_one_coverage=0.3,
    )
    raw = eb.model_dump_json()
    assert ExtractionBundle.model_validate_json(raw) == eb


def test_hierarchy_bundle_roundtrip():
    hb = HierarchyBundle(role_agreement=0.8, parent_attribution_accuracy=0.7)
    raw = hb.model_dump_json()
    assert HierarchyBundle.model_validate_json(raw) == hb


def test_classification_bundle_roundtrip():
    cb = ClassificationBundle(
        primary_category_agreement=0.85,
        confusion_matrix={"Mitigation": {"Mitigation": 3, "Adaptation": 1}},
        financial_instrument_agreement=0.9,
        secondary_category_agreement=0.6,
    )
    raw = cb.model_dump_json()
    assert ClassificationBundle.model_validate_json(raw) == cb


def test_quality_bundle_roundtrip():
    qb = QualityBundle(
        plus_one_coverage=0.4,
        grade_distribution={"+1": 0.5, "0": 0.3, "-1": 0.2},
        mean_grade=0.3,
    )
    raw = qb.model_dump_json()
    assert QualityBundle.model_validate_json(raw) == qb


def test_evaluation_bundle_roundtrip():
    eb = EvaluationBundle(
        location="Seattle_US",
        matching=_example_matching_result(),
        extraction=ExtractionBundle(
            f1=0.6, precision=0.7, recall=0.5,
            per_category_recall={}, category_distribution_jsd=0.1,
            plus_one_coverage=0.3,
        ),
        hierarchy=HierarchyBundle(role_agreement=0.8, parent_attribution_accuracy=0.7),
        classification=ClassificationBundle(
            primary_category_agreement=0.85, confusion_matrix={},
            financial_instrument_agreement=0.9, secondary_category_agreement=0.6,
        ),
        quality=QualityBundle(
            plus_one_coverage=0.4,
            grade_distribution={"+1": 0.5, "0": 0.3, "-1": 0.2},
            mean_grade=0.3,
        ),
        composite_score=0.61,
    )
    raw = eb.model_dump_json()
    assert EvaluationBundle.model_validate_json(raw) == eb
