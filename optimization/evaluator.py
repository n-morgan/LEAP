"""
evaluator.py — LEAP Evaluator (Algorithm 1)

Grades RLM-extracted policies against ground-truth (GENIUS) policies for one
location. Matching is **global**: embed every extracted and every GT
``policy_statement``, run Hungarian assignment on cosine similarity, then drop
pairs below ``similarity_threshold`` (default 0.55). Accepted pairs are graded
(+1 / 0 / -1) via the grader LLM, or via RLM when ``source_document_path`` is
set so the full document can be used.

Algorithm:
    1. Embed every extracted ``policy_statement`` and every GT ``policy_statement``
       (same embedding model for both sides).
    2. Build a cosine-similarity matrix (extracted × ground_truth).
    3. Run Hungarian matching once over that matrix for optimal 1:1 assignment
       (rectangular: each GT maps to at most one extraction and vice versa, up to
       min(n_extracted, n_gt) pairs).
    4. Keep only assignments whose similarity is ``>= similarity_threshold``;
       weaker assignments count as unmatched on both sides.
    5. For each accepted pair, call the grader (+1 / 0 / -1): Chat Completions
       parse when no document path is provided; RLM over full markdown when
       ``source_document_path`` is set.
    6. Aggregate headline metrics on accepted pairs: extraction precision/recall/F1,
       role agreement, parent attribution (sub/sub ``parent_statement`` match),
       primary category agreement, financial and secondary field agreement where
       defined, plus-one coverage (# grade +1 pairs / n_gt), and composite score.
    7. Backfill legacy **per-category** ``scores``, ``recall``, and ``fpr`` from the
       same global matching (GT-side and extracted-side category buckets, unmatched
       rows penalized with -1 where applicable).

See ``EvaluationOutput`` for field-level definitions.

C = { Mitigation, Adaptation, Resource Efficiency, Nature-Based Solutions }

Usage:
    evaluator = LEAPEvaluator(similarity_threshold=0.55)
    result = evaluator.evaluate(
        location="Seattle_US",
        extracted_policies=rlm_output,
        ground_truth_policies=genius_output,
        rubric="Grade on specificity, commitment, and mechanism...",
        source_document_path=path_to_markdown,  # optional; omit for grader-only
    )
    print(result.scores)
    print(result.composite_score, result.extraction_f1)
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import tempfile
import time
from typing import Any, Literal, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment

load_dotenv()

CATEGORIES: list[str] = [
    "Mitigation",
    "Adaptation",
    "Resource Efficiency",
    "Nature-Based Solutions",
]


# ---------------------------------------------------------------------------
# Financial instrument normalization (GT vs extracted field names)
# ---------------------------------------------------------------------------


def get_financial_instrument(policy: dict[str, Any]) -> Optional[str]:
    """
    Return a normalized financial flag: 'yes' | 'no' | None if unknown.

    Checks `financial_instrument` first, then `is_financial_instrument`
    (boolean or string).
    """
    if "financial_instrument" in policy and policy["financial_instrument"] not in (
        None,
        "",
    ):
        v = policy["financial_instrument"]
        if isinstance(v, bool):
            return "yes" if v else "no"
        s = str(v).strip().lower()
        if s in ("yes", "y", "true", "1"):
            return "yes"
        if s in ("no", "n", "false", "0"):
            return "no"
        return s if s else None
    raw = policy.get("is_financial_instrument")
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        return "yes" if raw else "no"
    s = str(raw).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return "yes"
    if s in ("no", "n", "false", "0"):
        return "no"
    return None


def _normalize_parent(s: Optional[str]) -> str:
    if not s:
        return ""
    return " ".join(str(s).strip().lower().split())


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def _compute_composite(
    extraction_f1: float,
    role_agreement: float,
    parent_attribution_accuracy: float,
    primary_category_agreement: float,
    plus_one_coverage: float,
) -> float:
    """Blend headline metrics; weights are tunable in one place."""
    hierarchy_headline = _safe_div(
        role_agreement + parent_attribution_accuracy,
        2.0,
    )
    return (
        0.40 * extraction_f1
        + 0.25 * hierarchy_headline
        + 0.25 * primary_category_agreement
        + 0.10 * plus_one_coverage
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PolicyGrade(BaseModel):
    """Grade for a single extracted policy relative to its matched ground-truth."""

    policy_id: str = Field(
        description="Truncated policy_statement used as a stable identifier."
    )
    grade: Literal[-1, 0, 1] = Field(
        description=(
            "+1: matches ground truth in scope, commitment, and specificity. "
            " 0: directionally correct but vague or imprecise. "
            "-1: no match, hallucinated, or contradicts document intent."
        )
    )
    reasoning: str = Field(
        description="Step-by-step justification grounded in the source text."
    )
    similarity: Optional[float] = Field(
        default=None,
        description="Cosine similarity of policy_statement embeddings for this pair.",
    )
    statement_match: Optional[bool] = Field(
        default=None,
        description="True if grade is +1 (full semantic match per grader).",
    )
    role_match: Optional[bool] = Field(
        default=None,
        description="True if extracted role equals ground-truth role.",
    )
    category_match: Optional[bool] = Field(
        default=None,
        description="True if primary_category equals ground-truth primary_category.",
    )


class _GraderOutput(BaseModel):
    """Structured output enforced on the grader LLM call."""

    grade: Literal[-1, 0, 1]
    reasoning: str


class EvaluationOutput(BaseModel):
    """Full evaluation result for one location.

    **Legacy (per-category, kept for prompt_optimizer):** ``scores`` are mean
    grades including ``-1`` for unmatched GT/extracted within each category;
    ``recall`` / ``fpr`` are matched GT / unmatched extracted ratios **within**
    each category bucket; ``hierarchy_accuracy`` mirrors ``role_agreement``.

    **Headline (global matching):** After Hungarian + similarity filtering,
    ``extraction_precision`` = accepted_pairs / n_extracted,
    ``extraction_recall`` = accepted_pairs / n_gt,
    ``extraction_f1`` = harmonic mean of those.
    ``role_agreement`` / ``primary_category_agreement`` / field agreements are
    computed only on **accepted** pairs; ``parent_attribution_accuracy`` only on
    pairs where both roles are ``sub`` (else trivially 1.0 when no subs).
    ``plus_one_coverage`` = (# accepted pairs with grade +1) / n_gt.

    ``composite_score`` = 0.40 * extraction_f1 + 0.25 * hierarchy_headline +
    0.25 * primary_category_agreement + 0.10 * plus_one_coverage, where
    hierarchy_headline = (role_agreement + parent_attribution_accuracy) / 2.
    """

    location: str

    # Legacy per-category metrics (prompt_optimizer compatibility)
    scores: dict[str, float] = Field(
        description="Mean grade per category (includes -1 penalties)."
    )
    recall: dict[str, float] = Field(
        description="Matched GT count / total GT count per category."
    )
    fpr: dict[str, float] = Field(
        description="Unmatched extracted count / total extracted count per category."
    )
    grades: dict[str, PolicyGrade] = Field(
        description=(
            "Per accepted pair: keys look like "
            "'{GT_primary_category}::gt{j}_ext{i}_{policy_id_stub}'. "
            "See PolicyGrade for similarity and role/category flags."
        ),
    )
    hierarchy_accuracy: float = Field(
        default=1.0,
        description="Same as role_agreement; kept for legacy callers.",
    )

    # Headline metrics — formulas summarized in class docstring above
    composite_score: float = Field(default=0.0, description="Weighted headline score.")
    extraction_precision: float = Field(
        default=0.0,
        description="accepted_pairs / n_extracted.",
    )
    extraction_recall: float = Field(
        default=0.0,
        description="accepted_pairs / n_gt.",
    )
    extraction_f1: float = Field(
        default=0.0,
        description="Harmonic mean of precision and recall.",
    )
    role_agreement: float = Field(
        default=0.0,
        description="Fraction of accepted pairs with matching role.",
    )
    parent_attribution_accuracy: float = Field(
        default=1.0,
        description="Sub/sub pairs with matching normalized parent_statement.",
    )
    primary_category_agreement: float = Field(
        default=0.0,
        description="Fraction of accepted pairs with matching primary_category.",
    )
    financial_instrument_agreement: float = Field(
        default=0.0,
        description="Agreement where both sides resolve via get_financial_instrument().",
    )
    secondary_category_agreement: float = Field(
        default=0.0,
        description="Agreement when at least one side sets secondary_category.",
    )
    plus_one_coverage: float = Field(
        default=0.0,
        description="(# accepted pairs with grade +1) / n_gt.",
    )

    matched_count: int = Field(
        default=0,
        description="Pairs passing Hungarian assignment and similarity_threshold.",
    )
    unmatched_extracted_count: int = Field(
        default=0,
        description="Extracted rows without an accepted pair.",
    )
    unmatched_ground_truth_count: int = Field(
        default=0,
        description="GT rows without an accepted pair.",
    )


# ---------------------------------------------------------------------------
# Grader prompts
# ---------------------------------------------------------------------------

_GRADER_SYSTEM = """\
You are an expert unbiased evaluator grading an extracted climate policy against a
ground-truth policy produced by a reference expert system.

SCORING GUIDE:
  +1  The extracted policy matches the ground-truth in scope, commitment, and specificity.
      The core commitment, target, and delivery mechanism are all captured correctly.
   0  Directionally correct but vague or imprecise. The general intent is right but
      key details such as targets, deadlines, or mechanisms are missing or softened.
  -1  No meaningful match, hallucinated content, or the extraction contradicts the
      document intent.

Ground your grade in the source document text when provided.
Return only the structured output — no preamble.
"""

_GRADER_USER = """\
RUBRIC:
{rubric}

SOURCE DOCUMENT (excerpt):
{source_document}

GROUND-TRUTH POLICY:
{ground_truth}

EXTRACTED POLICY:
{extracted}
"""

# RLM grader templates — used when source_document_path is set so the model can
# traverse the full markdown document before grading.

_RLM_GRADER_SYSTEM = """\
You are an expert unbiased evaluator grading an extracted climate policy against a
ground-truth policy. You have access to the full source document — read through it
to find the passage(s) that ground your evaluation before deciding on a grade.

SCORING GUIDE:
  +1  The extracted policy matches the ground-truth in scope, commitment, and
      specificity. Core commitment, target, and delivery mechanism all correct.
   0  Directionally correct but vague or imprecise. Intent is right but key
      details (targets, deadlines, mechanisms) are missing or softened.
  -1  No meaningful match, hallucinated content, or contradicts document intent.

After reading the document, return ONLY a JSON object with exactly two keys:
  "grade":     -1, 0, or 1
  "reasoning": step-by-step justification grounded in the source text

No preamble. No trailing text.
"""

_RLM_GRADER_USER = """\
RUBRIC:
{rubric}

GROUND-TRUTH POLICY:
{ground_truth}

EXTRACTED POLICY:
{extracted}

SOURCE DOCUMENT:
{document}
"""


# ---------------------------------------------------------------------------
# LEAPEvaluator
# ---------------------------------------------------------------------------


class LEAPEvaluator:
    """LEAP Evaluator — Algorithm 1 with global statement matching.

    Older implementations grouped by category and role before embedding + Hungarian.
    This version matches **all** extracted statements to **all** GT statements at
    once (still per ``location``), thresholds cosine similarity, then grades pairs.
    See ``EvaluationOutput`` for headline vs legacy fields.
    """

    def __init__(
        self,
        model: str = "gpt-5.4",
        embedding_model: str = "text-embedding-3-small",
        similarity_threshold: float = 0.55,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) float32 embedding matrix."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        response = self._get_client().embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return np.array([d.embedding for d in response.data], dtype=np.float32)

    def _cosine_sim(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Return (N, M) cosine similarity matrix between rows of A and B."""
        norm_a = np.linalg.norm(A, axis=1, keepdims=True) + 1e-10
        norm_b = np.linalg.norm(B, axis=1, keepdims=True) + 1e-10
        return (A / norm_a) @ (B / norm_b).T

    def _hungarian_match(self, sim: np.ndarray) -> list[tuple[int, int]]:
        """
        Optimal 1:1 matching maximising total cosine similarity.
        sim shape (n_ext, n_gt). Returns pairs (ext_idx, gt_idx).
        """
        if sim.size == 0:
            return []
        cost = -sim.astype(np.float64)
        row_ind, col_ind = linear_sum_assignment(cost)
        return list(zip(row_ind.tolist(), col_ind.tolist()))

    def _grade_pair_rlm(
        self,
        extracted: dict[str, Any],
        ground_truth: dict[str, Any],
        rubric: str,
        document_text: str,
    ) -> _GraderOutput:
        """Grade one pair via RLM so the full source document can be traversed."""
        from rlm import RLM
        from rlm.logger import RLMLogger

        log_dir = tempfile.mkdtemp(prefix="rlm_grade_")
        logger = RLMLogger(log_dir=log_dir)
        rlm = RLM(
            backend="openai",
            backend_kwargs={
                "model_name": self.model,
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            other_backends=["openai"],
            other_backend_kwargs=[
                {"model_name": self.model, "api_key": os.getenv("OPENAI_API_KEY")},
            ],
            environment="local",
            environment_kwargs={},
            max_depth=1,
            max_iterations=20,
            custom_system_prompt=_RLM_GRADER_SYSTEM,
            logger=logger,
            verbose=False,
        )
        prompt = _RLM_GRADER_USER.format(
            rubric=rubric,
            ground_truth=ground_truth.get("policy_statement", str(ground_truth)),
            extracted=extracted.get("policy_statement", str(extracted)),
            document=document_text,
        )
        result = rlm.completion(
            prompt=prompt,
            root_prompt="Grade the extracted policy against the ground truth using the source document.",
        )
        raw = result.response.strip()
        match = re.search(r'\{[^{}]*"grade"[^{}]*\}', raw, re.DOTALL)
        data = json.loads(match.group() if match else raw)

        print(data["reasoning"])
        grade = max(-1, min(1, int(data["grade"])))
        return _GraderOutput(grade=grade, reasoning=data.get("reasoning", ""))

    def _grade_pair(
        self,
        extracted: dict[str, Any],
        ground_truth: dict[str, Any],
        rubric: str,
        source_document: str,
    ) -> _GraderOutput:
        """Grade one matched pair via RLM (full doc) or Chat Completions parse."""
        if source_document:
            return self._grade_pair_rlm(extracted, ground_truth, rubric, source_document)

        user_msg = _GRADER_USER.format(
            rubric=rubric,
            source_document="Not provided.",
            ground_truth=ground_truth.get("policy_statement", str(ground_truth)),
            extracted=extracted.get("policy_statement", str(extracted)),
        )
        response = self._get_client().beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": _GRADER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=_GraderOutput,
        )

        print(response)
        time.sleep(5)
        return response.choices[0].message.parsed

    def evaluate(
        self,
        location: str,
        extracted_policies: list[dict[str, Any]],
        ground_truth_policies: list[dict[str, Any]],
        rubric: str,
        source_document_path: Optional[pathlib.Path | str] = None,
    ) -> EvaluationOutput:
        """Run global matching, grading, and aggregation for one location.

        Args:
            location: Location key, e.g. ``Seattle_US``.
            extracted_policies: Policy dicts from the RLM run (need ``policy_statement``).
            ground_truth_policies: Reference dicts (GENIUS / structured CSV rows).
            rubric: Free-form grading instructions for the grader LLM.
            source_document_path: Optional markdown path; if set, grading uses RLM
                over the full document; if omitted, the grader API runs without doc text.

        Returns:
            ``EvaluationOutput`` with headline metrics and legacy per-category fields.
        """
        doc_text = (
            pathlib.Path(source_document_path).read_text(encoding="utf-8")
            if source_document_path is not None
            else ""
        )

        n_ext = len(extracted_policies)
        n_gt = len(ground_truth_policies)

        # Empty inputs
        if n_ext == 0 and n_gt == 0:
            empty_scores = {c: 0.0 for c in CATEGORIES}
            empty_recall = {c: 1.0 for c in CATEGORIES}
            empty_fpr = {c: 0.0 for c in CATEGORIES}
            return EvaluationOutput(
                location=location,
                scores=empty_scores,
                recall=empty_recall,
                fpr=empty_fpr,
                grades={},
                hierarchy_accuracy=1.0,
                composite_score=0.0,
                extraction_precision=0.0,
                extraction_recall=0.0,
                extraction_f1=0.0,
                role_agreement=0.0,
                parent_attribution_accuracy=1.0,
                primary_category_agreement=0.0,
                financial_instrument_agreement=0.0,
                secondary_category_agreement=0.0,
                plus_one_coverage=0.0,
                matched_count=0,
                unmatched_extracted_count=n_ext,
                unmatched_ground_truth_count=n_gt,
            )

        # One side empty: no pairs possible
        if n_ext == 0 or n_gt == 0:
            scores_d: dict[str, float] = {}
            recall_d: dict[str, float] = {}
            fpr_d: dict[str, float] = {}
            for cat in CATEGORIES:
                gt_ic = [
                    j
                    for j in range(n_gt)
                    if ground_truth_policies[j].get("primary_category") == cat
                ]
                ext_ic = [
                    i
                    for i in range(n_ext)
                    if extracted_policies[i].get("primary_category") == cat
                ]
                contrib: list[float] = []
                for _j in gt_ic:
                    contrib.append(-1.0)
                for _i in ext_ic:
                    contrib.append(-1.0)
                scores_d[cat] = float(np.mean(contrib)) if contrib else 0.0
                recall_d[cat] = (
                    0.0
                    if gt_ic
                    else 1.0
                )
                fpr_d[cat] = (
                    1.0
                    if ext_ic
                    else 0.0
                )
            return EvaluationOutput(
                location=location,
                scores=scores_d,
                recall=recall_d,
                fpr=fpr_d,
                grades={},
                hierarchy_accuracy=1.0 if n_gt == 0 else 0.0,
                composite_score=0.0,
                extraction_precision=0.0,
                extraction_recall=0.0,
                extraction_f1=0.0,
                role_agreement=0.0,
                parent_attribution_accuracy=1.0,
                primary_category_agreement=0.0,
                financial_instrument_agreement=0.0,
                secondary_category_agreement=0.0,
                plus_one_coverage=0.0,
                matched_count=0,
                unmatched_extracted_count=n_ext,
                unmatched_ground_truth_count=n_gt,
            )

        ext_texts = [p.get("policy_statement", "") or "" for p in extracted_policies]
        gt_texts = [
            p.get("policy_statement", "") or "" for p in ground_truth_policies
        ]

        ext_emb = self._embed(ext_texts)
        gt_emb = self._embed(gt_texts)
        sim = self._cosine_sim(ext_emb, gt_emb)

        raw_pairs = self._hungarian_match(sim)

        accepted: list[tuple[int, int, float]] = []
        matched_ext_idx: set[int] = set()
        matched_gt_idx: set[int] = set()

        thr = self.similarity_threshold
        for ei, gj in raw_pairs:
            s_ij = float(sim[ei, gj])
            if s_ij >= thr:
                accepted.append((ei, gj, s_ij))
                matched_ext_idx.add(ei)
                matched_gt_idx.add(gj)

        grades: dict[str, PolicyGrade] = {}
        pair_grades: dict[tuple[int, int], int] = {}

        role_matches = 0
        cat_matches = 0
        fi_matches = 0
        fi_total = 0
        sec_matches = 0
        sec_total = 0
        parent_correct = 0
        parent_total = 0

        plus_one_gt_count = 0

        for ei, gj, s_ij in accepted:
            ext = extracted_policies[ei]
            gt = ground_truth_policies[gj]

            graded = self._grade_pair(ext, gt, rubric, doc_text)

            ext_role = ext.get("role", "individual")
            gt_role = gt.get("role", "individual")
            rm = ext_role == gt_role

            ext_cat = ext.get("primary_category")
            gt_cat = gt.get("primary_category")
            cm = ext_cat == gt_cat

            policy_id = (ext.get("policy_statement") or f"ext_{ei}")[:80]
            gt_primary = gt_cat if isinstance(gt_cat, str) else "Unknown"
            key = f"{gt_primary}::gt{gj}_ext{ei}_{policy_id}"

            stmt_match = graded.grade == 1

            pair_grades[(ei, gj)] = graded.grade

            grades[key] = PolicyGrade(
                policy_id=policy_id,
                grade=graded.grade,
                reasoning=graded.reasoning,
                similarity=s_ij,
                statement_match=stmt_match,
                role_match=rm,
                category_match=cm,
            )

            if rm:
                role_matches += 1
            if cm:
                cat_matches += 1

            if gt_role == "sub" and ext_role == "sub":
                parent_total += 1
                if _normalize_parent(ext.get("parent_statement")) == _normalize_parent(
                    gt.get("parent_statement")
                ):
                    parent_correct += 1

            fi_e = get_financial_instrument(ext)
            fi_g = get_financial_instrument(gt)
            if fi_e is not None and fi_g is not None:
                fi_total += 1
                if fi_e == fi_g:
                    fi_matches += 1

            se = ext.get("secondary_category")
            sg = gt.get("secondary_category")
            if se not in (None, "") or sg not in (None, ""):
                sec_total += 1
                if se == sg:
                    sec_matches += 1

            if graded.grade == 1:
                plus_one_gt_count += 1

        matched_n = len(accepted)
        unmatched_ext = n_ext - len(matched_ext_idx)
        unmatched_gt = n_gt - len(matched_gt_idx)

        extraction_precision = _safe_div(matched_n, n_ext)
        extraction_recall = _safe_div(matched_n, n_gt)
        p, r = extraction_precision, extraction_recall
        extraction_f1 = _safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0

        role_agreement = _safe_div(role_matches, matched_n) if matched_n else 0.0
        primary_category_agreement = (
            _safe_div(cat_matches, matched_n) if matched_n else 0.0
        )
        financial_instrument_agreement = (
            _safe_div(fi_matches, fi_total) if fi_total else 0.0
        )
        secondary_category_agreement = (
            _safe_div(sec_matches, sec_total) if sec_total else 0.0
        )

        if parent_total > 0:
            parent_attribution_accuracy = _safe_div(parent_correct, parent_total)
        else:
            parent_attribution_accuracy = 1.0

        plus_one_coverage = _safe_div(plus_one_gt_count, n_gt)

        composite_score = _compute_composite(
            extraction_f1,
            role_agreement,
            parent_attribution_accuracy,
            primary_category_agreement,
            plus_one_coverage,
        )

        hierarchy_accuracy = role_agreement

        # Legacy per-category scores / recall / fpr (compat with prompt_optimizer)
        scores: dict[str, float] = {}
        recall: dict[str, float] = {}
        fpr: dict[str, float] = {}

        for cat in CATEGORIES:
            gt_indices_c = [
                j
                for j in range(n_gt)
                if ground_truth_policies[j].get("primary_category") == cat
            ]
            ext_indices_c = [
                i
                for i in range(n_ext)
                if extracted_policies[i].get("primary_category") == cat
            ]

            contrib: list[float] = []

            for j in gt_indices_c:
                if j not in matched_gt_idx:
                    contrib.append(-1.0)
                else:
                    ei_m = next(e for e, g, _ in accepted if g == j)
                    gpair = pair_grades.get((ei_m, j))
                    contrib.append(float(gpair) if gpair is not None else -1.0)

            for i in ext_indices_c:
                if i not in matched_ext_idx:
                    contrib.append(-1.0)

            scores[cat] = float(np.mean(contrib)) if contrib else 0.0

            recall[cat] = (
                _safe_div(
                    sum(1 for j in gt_indices_c if j in matched_gt_idx),
                    len(gt_indices_c),
                )
                if gt_indices_c
                else 1.0
            )
            fpr[cat] = (
                _safe_div(
                    sum(1 for i in ext_indices_c if i not in matched_ext_idx),
                    len(ext_indices_c),
                )
                if ext_indices_c
                else 0.0
            )

        return EvaluationOutput(
            location=location,
            scores=scores,
            recall=recall,
            fpr=fpr,
            grades=grades,
            hierarchy_accuracy=hierarchy_accuracy,
            composite_score=composite_score,
            extraction_precision=extraction_precision,
            extraction_recall=extraction_recall,
            extraction_f1=extraction_f1,
            role_agreement=role_agreement,
            parent_attribution_accuracy=parent_attribution_accuracy,
            primary_category_agreement=primary_category_agreement,
            financial_instrument_agreement=financial_instrument_agreement,
            secondary_category_agreement=secondary_category_agreement,
            plus_one_coverage=plus_one_coverage,
            matched_count=matched_n,
            unmatched_extracted_count=unmatched_ext,
            unmatched_ground_truth_count=unmatched_gt,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import csv
    import pathlib

    OUTPUTS_DIR = pathlib.Path(__file__).parent / "organized_outputs"

    def load_policies(path: pathlib.Path) -> list[dict]:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [row for row in reader if row.get("policy_statement", "").strip()]

    rlm_policies = load_policies(OUTPUTS_DIR / "rlm_seattle_policies.csv")
    structured_policies = load_policies(OUTPUTS_DIR / "structured_policies.csv")

    print(f"Loaded {len(rlm_policies)} RLM policies and "
          f"{len(structured_policies)} structured (ground-truth) policies.")

    DEFAULT_RUBRIC = (
        "Grade on specificity (quantified targets, deadlines, mechanisms), "
        "commitment strength (binding vs aspirational language), "
        "and accuracy relative to the source document."
    )

    evaluator = LEAPEvaluator(similarity_threshold=0.55)
    result = evaluator.evaluate(
        location="Seattle_US",
        extracted_policies=rlm_policies,
        ground_truth_policies=structured_policies,
        rubric=DEFAULT_RUBRIC,
    )

    print("\n=== RLM vs Structured System — Seattle_US ===")
    print(f"composite_score={result.composite_score:.4f}")
    print(f"extraction_precision={result.extraction_precision:.4f}  "
          f"recall={result.extraction_recall:.4f}  f1={result.extraction_f1:.4f}")
    print(f"role_agreement={result.role_agreement:.4f}  "
          f"parent_attr={result.parent_attribution_accuracy:.4f}")
    print(f"primary_cat_agreement={result.primary_category_agreement:.4f}  "
          f"+1_cov={result.plus_one_coverage:.4f}")
    print(f"matched={result.matched_count}  unmatched_ext={result.unmatched_extracted_count}  "
          f"unmatched_gt={result.unmatched_ground_truth_count}")
    print(f"\n{'Category':<25} {'Score':>7}  {'Recall':>7}  {'FPR':>7}")
    print("-" * 52)
    for cat in CATEGORIES:
        print(
            f"{cat:<25} {result.scores.get(cat, 0.0):>7.3f}"
            f"  {result.recall.get(cat, 0.0):>7.3f}"
            f"  {result.fpr.get(cat, 0.0):>7.3f}"
        )
    overall = (
        sum(result.scores.values()) / len(result.scores) if result.scores else 0.0
    )
    print("-" * 52)
    print(f"{'Overall mean score':<25} {overall:>7.3f}")
