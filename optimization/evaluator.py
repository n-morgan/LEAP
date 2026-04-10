"""
evaluator.py — LEAP Evaluator (Algorithm 1)

Grades RLM-extracted policies against ground-truth (GENIUS) policies for a given
location. Produces per-category scores, recall, and FPR, plus per-policy grade
reasoning keyed by (category, policy_id) for downstream prompt optimization.

Algorithm:
    For each category c in C:
        1. Group extracted and ground-truth policies by role (parent / sub / individual)
        2. Embed policy statements and compute pairwise cosine similarity
        3. Run Hungarian matching for optimal 1:1 alignment within each role group
        4. Grade each matched pair via GraderLLM (+1 / 0 / -1)
        5. Penalize unmatched GT policies (-1, hurts recall)
        6. Penalize unmatched extracted policies (-1, hurts FPR)
        7. Aggregate into S[l, c], R[l, c], F[l, c]

C = { Mitigation, Adaptation, Resource Efficiency, Nature-Based Solutions }

Usage:
    evaluator = LEAPEvaluator()
    result = evaluator.evaluate(
        location="Seattle_US",
        extracted_policies=rlm_output,
        ground_truth_policies=genius_output,
        rubric="Grade on specificity, commitment, and mechanism...",
        source_document=markdown_text,
    )
    print(result.scores)   # {"Mitigation": 0.4, "Adaptation": -0.2, ...}
    print(result.recall)   # {"Mitigation": 0.8, ...}
"""

import os
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


class _GraderOutput(BaseModel):
    """Structured output enforced on the grader LLM call."""

    grade: Literal[-1, 0, 1]
    reasoning: str


class EvaluationOutput(BaseModel):
    """Full evaluation result for one location."""

    location: str
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
        description="Key format: '{category}::{policy_id}'. One entry per matched pair."
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


# ---------------------------------------------------------------------------
# LEAPEvaluator
# ---------------------------------------------------------------------------


class LEAPEvaluator:
    """
    Implements Algorithm 1: LEAP Evaluator.

    Groups policies by category and role, runs optimal 1:1 Hungarian matching
    on embedding similarity, grades each matched pair via an LLM using the
    expert rubric, and penalizes unmatched policies on both sides.
    """

    def __init__(
        self,
        model: str = "gpt-5",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self._client: Optional[OpenAI] = None

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    # ------------------------------------------------------------------
    # Embedding and matching
    # ------------------------------------------------------------------

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
        Returns a list of (extracted_idx, ground_truth_idx) pairs.
        """
        if sim.size == 0:
            return []
        row_ind, col_ind = linear_sum_assignment(-sim)
        return list(zip(row_ind.tolist(), col_ind.tolist()))

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------

    def _grade_pair(
        self,
        extracted: dict[str, Any],
        ground_truth: dict[str, Any],
        rubric: str,
        source_document: str,
    ) -> _GraderOutput:
        """Call GraderLLM on one matched (extracted, ground_truth) pair."""
        user_msg = _GRADER_USER.format(
            rubric=rubric,
            source_document=source_document[:3000] if source_document else "Not provided.",
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
        return response.choices[0].message.parsed

    # ------------------------------------------------------------------
    # Cell evaluation
    # ------------------------------------------------------------------

    def _evaluate_cell(
        self,
        category: str,
        extracted: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]],
        rubric: str,
        source_document: str,
    ) -> tuple[float, float, float, dict[str, PolicyGrade]]:
        """
        Evaluate one (location, category) cell.

        Separates policies by role (parent / sub / individual) to enforce
        hierarchy-aware matching (parent to parent, child to child), then
        runs Hungarian matching within each role group independently.

        Returns:
            score   — mean grade across all scores including -1 penalties
            recall  — matched GT / total GT
            fpr     — unmatched extracted / total extracted
            grades  — dict of PolicyGrade keyed by '{category}::{policy_id}'
        """
        grades: dict[str, PolicyGrade] = {}
        all_scores: list[int] = []

        if not extracted and not ground_truth:
            return 0.0, 1.0, 0.0, grades

        # Group by role for hierarchy-aware matching
        def by_role(policies: list[dict]) -> dict[str, list[dict]]:
            groups: dict[str, list[dict]] = {
                "parent": [], "sub": [], "individual": []
            }
            for p in policies:
                role = p.get("role", "individual")
                groups.setdefault(role, []).append(p)
            return groups

        ext_by_role = by_role(extracted)
        gt_by_role = by_role(ground_truth)

        # Track global indices into ext_flat / gt_flat for unmatched detection
        ext_flat: list[dict] = []
        gt_flat: list[dict] = []
        matched_ext: set[int] = set()
        matched_gt: set[int] = set()

        for role in ("parent", "sub", "individual"):
            ext_group = ext_by_role.get(role, [])
            gt_group = gt_by_role.get(role, [])

            ext_offset = len(ext_flat)
            gt_offset = len(gt_flat)
            ext_flat.extend(ext_group)
            gt_flat.extend(gt_group)

            if not ext_group or not gt_group:
                continue

            # Embed and match
            ext_emb = self._embed([p.get("policy_statement", "") for p in ext_group])
            gt_emb = self._embed([p.get("policy_statement", "") for p in gt_group])
            sim = self._cosine_sim(ext_emb, gt_emb)
            pairs = self._hungarian_match(sim)

            for local_ei, local_gi in pairs:
                global_ei = ext_offset + local_ei
                global_gi = gt_offset + local_gi
                matched_ext.add(global_ei)
                matched_gt.add(global_gi)

                graded = self._grade_pair(
                    ext_group[local_ei],
                    gt_group[local_gi],
                    rubric,
                    source_document,
                )
                policy_id = ext_group[local_ei].get("policy_statement", f"{role}_{local_ei}")[:80]
                grades[f"{category}::{policy_id}"] = PolicyGrade(
                    policy_id=policy_id,
                    grade=graded.grade,
                    reasoning=graded.reasoning,
                )
                all_scores.append(graded.grade)

        # Unmatched GT: RLM missed them — penalize recall
        for gi in range(len(gt_flat)):
            if gi not in matched_gt:
                all_scores.append(-1)

        # Unmatched extracted: no GT counterpart — penalize FPR
        for ei in range(len(ext_flat)):
            if ei not in matched_ext:
                all_scores.append(-1)

        score = float(np.mean(all_scores)) if all_scores else 0.0
        recall = len(matched_gt) / len(gt_flat) if gt_flat else 1.0
        fpr = (len(ext_flat) - len(matched_ext)) / len(ext_flat) if ext_flat else 0.0

        return score, recall, fpr, grades

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        location: str,
        extracted_policies: list[dict[str, Any]],
        ground_truth_policies: list[dict[str, Any]],
        rubric: str,
        source_document: str = "",
    ) -> EvaluationOutput:
        """
        Run Algorithm 1 for a single location.

        Args:
            location:               location key, e.g. "Seattle_US"
            extracted_policies:     policy dicts from the RLM run
            ground_truth_policies:  policy dicts from GENIUS (Ziyad model)
            rubric:                 freeform grading guidelines string
            source_document:        markdown source text for richer grading context

        Returns:
            EvaluationOutput with per-category scores, recall, FPR, and grades.
        """
        scores: dict[str, float] = {}
        recall: dict[str, float] = {}
        fpr: dict[str, float] = {}
        all_grades: dict[str, PolicyGrade] = {}

        for category in CATEGORIES:
            ext_cat = [
                p for p in extracted_policies
                if p.get("primary_category") == category
            ]
            gt_cat = [
                p for p in ground_truth_policies
                if p.get("primary_category") == category
            ]

            s, r, f, cell_grades = self._evaluate_cell(
                category, ext_cat, gt_cat, rubric, source_document
            )
            scores[category] = s
            recall[category] = r
            fpr[category] = f
            all_grades.update(cell_grades)

        return EvaluationOutput(
            location=location,
            scores=scores,
            recall=recall,
            fpr=fpr,
            grades=all_grades,
        )
