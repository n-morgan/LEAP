"""
metrics.py — orthogonal metric bundles for the redesigned LEAP optimizer.

Defines:
  * MatchedPair / MatchingResult           — output of category-agnostic Hungarian
                                             matching with a similarity threshold.
  * ExtractionBundle / HierarchyBundle /
    ClassificationBundle / QualityBundle   — three prong-owned bundles plus one
                                             cross-cutting quality bundle.
  * EvaluationBundle                       — the full per-location result object.
  * compute_*_bundle / compute_composite_score — pure functions that turn a
    MatchingResult (plus the original GT list) into the bundles above.

Design notes
------------
The bundles are deliberately decoupled from any LLM or embedding I/O so they can
be unit-tested with hand-built fixtures (see optimization/tests/test_bundles.py).
The matching stage owns all similarity / Hungarian work; the bundles only see
the resulting MatchedPair lists.

Each bundle's *headline* signal is the single number the owning prong's trigger
reads. The supporting metrics in each bundle are kept for diagnostics and
logging only.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial.distance import jensenshannon


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MatchedPair(BaseModel):
    """One extracted/GT pair with its similarity and (optional) grader output."""

    extracted: dict
    ground_truth: dict
    similarity: float
    grade: Optional[Literal[-1, 0, 1]] = None
    reasoning: Optional[str] = None
    statement_match: Optional[Literal["match", "partial", "mismatch"]] = None
    role_match: Optional[bool] = None
    category_match: Optional[bool] = None


class MatchingResult(BaseModel):
    """Output of category-agnostic Hungarian matching with similarity threshold."""

    matched: list[MatchedPair]
    unmatched_extracted: list[dict]
    unmatched_gt: list[dict]
    similarity_threshold: float


class ExtractionBundle(BaseModel):
    """Headline: micro-F1 on M vs. GT."""

    f1: float
    precision: float
    recall: float
    per_category_recall: dict[str, float]
    category_distribution_jsd: float
    plus_one_coverage: float


class HierarchyBundle(BaseModel):
    """Headline: role-agreement rate on M (or mean of role + parent attribution)."""

    role_agreement: float
    parent_attribution_accuracy: float


class ClassificationBundle(BaseModel):
    """Headline: primary-category agreement on M."""

    primary_category_agreement: float
    confusion_matrix: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="gt_primary_category -> extracted_primary_category -> count",
    )
    financial_instrument_agreement: float
    secondary_category_agreement: float


class QualityBundle(BaseModel):
    """Cross-cutting quality bundle (used in composite, not prong-owned)."""

    plus_one_coverage: float
    grade_distribution: dict[str, float] = Field(
        default_factory=dict,
        description='"+1" / "0" / "-1" -> fraction of matched pairs',
    )
    mean_grade: float


class EvaluationBundle(BaseModel):
    """Full evaluation result for one location."""

    location: str
    matching: MatchingResult
    extraction: ExtractionBundle
    hierarchy: HierarchyBundle
    classification: ClassificationBundle
    quality: QualityBundle
    composite_score: float


# ---------------------------------------------------------------------------
# Composite-score weights (defaults)
# ---------------------------------------------------------------------------

DEFAULT_COMPOSITE_WEIGHTS: dict[str, float] = {
    "extraction_f1": 0.40,
    "hierarchy": 0.25,
    "classification": 0.25,
    "plus_one_coverage": 0.10,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den > 0 else default


def _category_histogram(items: list[dict], key: str = "primary_category") -> dict[str, float]:
    """Return a normalised category histogram for the given list of dicts."""
    if not items:
        return {}
    counts = Counter((p.get(key) or "Unknown") for p in items)
    total = sum(counts.values())
    return {cat: c / total for cat, c in counts.items()}


def _aligned_distributions(
    a: dict[str, float], b: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned probability vectors over the union of categories."""
    keys = sorted(set(a.keys()) | set(b.keys()))
    if not keys:
        return np.array([1.0]), np.array([1.0])
    va = np.array([a.get(k, 0.0) for k in keys], dtype=np.float64)
    vb = np.array([b.get(k, 0.0) for k in keys], dtype=np.float64)
    return va, vb


# ---------------------------------------------------------------------------
# Bundle computation
# ---------------------------------------------------------------------------


def compute_extraction_bundle(
    matching: MatchingResult, ground_truth: list[dict]
) -> ExtractionBundle:
    """Compute the extraction bundle.

    Headline: micro-F1 on the matched set vs. ground truth.

    A "match" here means a pair survived the global Hungarian + threshold gate.
    Therefore:
        TP = |M|
        FP = |U_E|   (extracted with no GT counterpart above threshold)
        FN = |U_G|   (GT with no extracted counterpart above threshold)
    """
    tp = len(matching.matched)
    fp = len(matching.unmatched_extracted)
    fn = len(matching.unmatched_gt)

    precision = _safe_div(tp, tp + fp, default=0.0)
    recall = _safe_div(tp, tp + fn, default=0.0)
    f1 = _safe_div(2 * precision * recall, precision + recall, default=0.0)

    # Per-category recall on GT
    gt_by_cat: dict[str, list[dict]] = {}
    for p in ground_truth:
        gt_by_cat.setdefault(p.get("primary_category") or "Unknown", []).append(p)

    matched_gt_stmts = {
        (pair.ground_truth.get("policy_statement") or "").strip()
        for pair in matching.matched
    }
    per_category_recall: dict[str, float] = {}
    for cat, items in gt_by_cat.items():
        m = sum(
            1 for p in items
            if (p.get("policy_statement") or "").strip() in matched_gt_stmts
        )
        per_category_recall[cat] = _safe_div(m, len(items), default=0.0)

    # Category distribution JSD: extracted set (matched + unmatched_extracted)
    # vs. GT set.
    extracted_all = (
        [pair.extracted for pair in matching.matched] + list(matching.unmatched_extracted)
    )
    ext_hist = _category_histogram(extracted_all)
    gt_hist = _category_histogram(ground_truth)
    va, vb = _aligned_distributions(ext_hist, gt_hist)
    jsd = float(jensenshannon(va, vb, base=2)) if va.sum() > 0 and vb.sum() > 0 else 1.0
    if np.isnan(jsd):
        jsd = 0.0

    plus_one = sum(1 for pair in matching.matched if pair.grade == 1)
    plus_one_coverage = _safe_div(plus_one, len(ground_truth), default=0.0)

    return ExtractionBundle(
        f1=f1,
        precision=precision,
        recall=recall,
        per_category_recall=per_category_recall,
        category_distribution_jsd=jsd,
        plus_one_coverage=plus_one_coverage,
    )


def compute_hierarchy_bundle(matching: MatchingResult) -> HierarchyBundle:
    """Compute the hierarchy bundle.

    Two metrics, both restricted to the matched set M:
      * role_agreement
            fraction of M whose extracted role equals GT role.
      * parent_attribution_accuracy
            for subs in M whose GT counterpart is also a sub *and* whose GT
            parent has a counterpart in M, fraction whose extracted
            parent_statement resolves to that same matched-pair counterpart.
    """
    if not matching.matched:
        return HierarchyBundle(role_agreement=1.0, parent_attribution_accuracy=1.0)

    # Role agreement
    role_correct = sum(
        1
        for pair in matching.matched
        if (pair.extracted.get("role") or "individual")
        == (pair.ground_truth.get("role") or "individual")
    )
    role_agreement = role_correct / len(matching.matched)

    # Parent attribution accuracy
    # Build maps from GT and extracted policy_statement -> matched_pair index.
    ext_to_pair: dict[str, int] = {}
    gt_to_pair: dict[str, int] = {}
    for i, pair in enumerate(matching.matched):
        ext_stmt = (pair.extracted.get("policy_statement") or "").strip()
        gt_stmt = (pair.ground_truth.get("policy_statement") or "").strip()
        if ext_stmt:
            ext_to_pair[ext_stmt] = i
        if gt_stmt:
            gt_to_pair[gt_stmt] = i

    eligible = 0
    correct = 0
    for pair in matching.matched:
        gt_role = pair.ground_truth.get("role") or "individual"
        ext_role = pair.extracted.get("role") or "individual"
        if gt_role != "sub" or ext_role != "sub":
            continue

        gt_parent = (pair.ground_truth.get("parent_statement") or "").strip()
        ext_parent = (pair.extracted.get("parent_statement") or "").strip()
        if not gt_parent:
            continue

        gt_parent_pair = gt_to_pair.get(gt_parent)
        if gt_parent_pair is None:
            # GT parent isn't in M — exclude so extraction failure doesn't leak.
            continue

        eligible += 1
        ext_parent_pair = ext_to_pair.get(ext_parent)
        if ext_parent_pair is not None and ext_parent_pair == gt_parent_pair:
            correct += 1

    parent_attribution_accuracy = (
        correct / eligible if eligible > 0 else 1.0
    )

    return HierarchyBundle(
        role_agreement=role_agreement,
        parent_attribution_accuracy=parent_attribution_accuracy,
    )


def compute_classification_bundle(matching: MatchingResult) -> ClassificationBundle:
    """Compute the classification bundle on the matched set.

    Headline: primary_category agreement on M.
    Plus a 4x4 confusion matrix and the two secondary-field agreements.
    """
    if not matching.matched:
        return ClassificationBundle(
            primary_category_agreement=1.0,
            confusion_matrix={},
            financial_instrument_agreement=1.0,
            secondary_category_agreement=1.0,
        )

    primary_correct = 0
    fin_correct = 0
    sec_correct = 0
    confusion: dict[str, dict[str, int]] = {}

    for pair in matching.matched:
        gt_cat = pair.ground_truth.get("primary_category") or "Unknown"
        ext_cat = pair.extracted.get("primary_category") or "Unknown"
        confusion.setdefault(gt_cat, {}).setdefault(ext_cat, 0)
        confusion[gt_cat][ext_cat] += 1
        if gt_cat == ext_cat:
            primary_correct += 1

        if (pair.ground_truth.get("financial_instrument") or "") == (
            pair.extracted.get("financial_instrument") or ""
        ):
            fin_correct += 1

        gt_sec = pair.ground_truth.get("secondary_category") or pair.ground_truth.get(
            "secondary_categories"
        ) or ""
        ext_sec = pair.extracted.get("secondary_category") or pair.extracted.get(
            "secondary_categories"
        ) or ""
        if gt_sec == ext_sec:
            sec_correct += 1

    n = len(matching.matched)
    return ClassificationBundle(
        primary_category_agreement=primary_correct / n,
        confusion_matrix=confusion,
        financial_instrument_agreement=fin_correct / n,
        secondary_category_agreement=sec_correct / n,
    )


def compute_quality_bundle(
    matching: MatchingResult, ground_truth: list[dict]
) -> QualityBundle:
    """Compute the cross-cutting quality bundle.

    Headline: +1-coverage = |{m in M with grade=+1}| / |GT|.
    """
    if not matching.matched:
        return QualityBundle(
            plus_one_coverage=0.0,
            grade_distribution={"+1": 0.0, "0": 0.0, "-1": 0.0},
            mean_grade=0.0,
        )

    grades = [pair.grade for pair in matching.matched if pair.grade is not None]
    n = len(grades)
    if n == 0:
        return QualityBundle(
            plus_one_coverage=0.0,
            grade_distribution={"+1": 0.0, "0": 0.0, "-1": 0.0},
            mean_grade=0.0,
        )

    plus_one = sum(1 for g in grades if g == 1)
    zero = sum(1 for g in grades if g == 0)
    minus_one = sum(1 for g in grades if g == -1)

    return QualityBundle(
        plus_one_coverage=_safe_div(plus_one, len(ground_truth), default=0.0),
        grade_distribution={
            "+1": plus_one / n,
            "0": zero / n,
            "-1": minus_one / n,
        },
        mean_grade=float(np.mean(grades)),
    )


def compute_composite_score(
    extraction: ExtractionBundle,
    hierarchy: HierarchyBundle,
    classification: ClassificationBundle,
    quality: QualityBundle,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """Weighted composite over each prong's headline + +1-coverage."""
    w = weights or DEFAULT_COMPOSITE_WEIGHTS
    hierarchy_headline = (
        hierarchy.role_agreement + hierarchy.parent_attribution_accuracy
    ) / 2.0
    return (
        w.get("extraction_f1", 0.0) * extraction.f1
        + w.get("hierarchy", 0.0) * hierarchy_headline
        + w.get("classification", 0.0) * classification.primary_category_agreement
        + w.get("plus_one_coverage", 0.0) * quality.plus_one_coverage
    )


# ---------------------------------------------------------------------------
# Aggregation across locations / seeds
# ---------------------------------------------------------------------------


def aggregate_bundles(
    bundles: list[EvaluationBundle],
    weights: Optional[dict[str, float]] = None,
    location_label: str = "aggregate",
) -> tuple[EvaluationBundle, dict[str, float]]:
    """Mean-reduce a list of EvaluationBundles into one.

    Returns:
        (aggregate_bundle, std_per_metric) where ``std_per_metric`` carries
        per-metric standard deviations across the input bundles for variance
        reporting.

    The returned bundle's ``matching`` is a synthetic stub holding all matched
    pairs concatenated — useful for downstream feedback assembly without
    pretending the matchings were produced jointly.
    """
    if not bundles:
        raise ValueError("aggregate_bundles called with no bundles")

    f1s = np.array([b.extraction.f1 for b in bundles])
    precs = np.array([b.extraction.precision for b in bundles])
    recs = np.array([b.extraction.recall for b in bundles])
    jsds = np.array([b.extraction.category_distribution_jsd for b in bundles])
    plus_one_ext = np.array([b.extraction.plus_one_coverage for b in bundles])
    role_ag = np.array([b.hierarchy.role_agreement for b in bundles])
    par_attr = np.array([b.hierarchy.parent_attribution_accuracy for b in bundles])
    primary = np.array([b.classification.primary_category_agreement for b in bundles])
    fin = np.array([b.classification.financial_instrument_agreement for b in bundles])
    sec = np.array([b.classification.secondary_category_agreement for b in bundles])
    plus_one_q = np.array([b.quality.plus_one_coverage for b in bundles])
    mean_grade = np.array([b.quality.mean_grade for b in bundles])
    composite = np.array([b.composite_score for b in bundles])

    # Per-category recall: union over bundles, mean over those that report each cat.
    per_cat_keys: set[str] = set()
    for b in bundles:
        per_cat_keys.update(b.extraction.per_category_recall.keys())
    per_category_recall: dict[str, float] = {}
    for k in per_cat_keys:
        vals = [b.extraction.per_category_recall.get(k) for b in bundles]
        vals = [v for v in vals if v is not None]
        per_category_recall[k] = float(np.mean(vals)) if vals else 0.0

    # Confusion matrix: sum across bundles
    confusion: dict[str, dict[str, int]] = {}
    for b in bundles:
        for gt_cat, row in b.classification.confusion_matrix.items():
            for ext_cat, count in row.items():
                confusion.setdefault(gt_cat, {}).setdefault(ext_cat, 0)
                confusion[gt_cat][ext_cat] += count

    extraction = ExtractionBundle(
        f1=float(f1s.mean()),
        precision=float(precs.mean()),
        recall=float(recs.mean()),
        per_category_recall=per_category_recall,
        category_distribution_jsd=float(jsds.mean()),
        plus_one_coverage=float(plus_one_ext.mean()),
    )
    hierarchy = HierarchyBundle(
        role_agreement=float(role_ag.mean()),
        parent_attribution_accuracy=float(par_attr.mean()),
    )
    classification = ClassificationBundle(
        primary_category_agreement=float(primary.mean()),
        confusion_matrix=confusion,
        financial_instrument_agreement=float(fin.mean()),
        secondary_category_agreement=float(sec.mean()),
    )
    quality = QualityBundle(
        plus_one_coverage=float(plus_one_q.mean()),
        grade_distribution={"+1": 0.0, "0": 0.0, "-1": 0.0},
        mean_grade=float(mean_grade.mean()),
    )

    # Concatenate matched pairs so downstream feedback has a non-empty corpus.
    all_matched = [pair for b in bundles for pair in b.matching.matched]
    all_unmatched_e = [p for b in bundles for p in b.matching.unmatched_extracted]
    all_unmatched_g = [p for b in bundles for p in b.matching.unmatched_gt]
    matching = MatchingResult(
        matched=all_matched,
        unmatched_extracted=all_unmatched_e,
        unmatched_gt=all_unmatched_g,
        similarity_threshold=bundles[0].matching.similarity_threshold,
    )

    aggregate = EvaluationBundle(
        location=location_label,
        matching=matching,
        extraction=extraction,
        hierarchy=hierarchy,
        classification=classification,
        quality=quality,
        composite_score=float(composite.mean()),
    )

    std_per_metric = {
        "extraction.f1": float(f1s.std(ddof=0)),
        "hierarchy.role_agreement": float(role_ag.std(ddof=0)),
        "hierarchy.parent_attribution_accuracy": float(par_attr.std(ddof=0)),
        "classification.primary_category_agreement": float(primary.std(ddof=0)),
        "quality.plus_one_coverage": float(plus_one_q.std(ddof=0)),
        "composite_score": float(composite.std(ddof=0)),
    }
    return aggregate, std_per_metric
