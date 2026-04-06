"""
RLM prompt optimizer.

Accepts already-extracted policies (from an external RLM run) and provides:
  - evaluate()      — grade each policy against ground truth
  - improve_prompt() — rewrite the system prompt to fix identified failures
  - step()          — one full evaluate-then-improve cycle

RLM extraction is intentionally kept out of this module; run it in a
separate file and pass the resulting policy list in.
"""

import json
import os
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PolicyGrade(BaseModel):
    """Evaluation grade for a single extracted policy."""

    policy_statement: str = Field(
        description="The policy statement being graded (matches the input policy)"
    )
    grade: Literal[-1, 0, 1] = Field(
        description="+1 (great match), 0 (acceptable), -1 (poor/missing)"
    )
    reasoning: str = Field(
        description="Explanation for the grade relative to the ground-truth policy"
    )


class _RawEvaluation(BaseModel):
    """
    Schema enforced on the LLM via JSON schema mode.

    Uses a list rather than a dict so OpenAI structured output can validate
    it (additionalProperties / dynamic dict keys are not supported).
    """

    per_policy_eval: list[PolicyGrade] = Field(
        description="One entry per input policy"
    )


class EvaluationResult(BaseModel):
    """Aggregate evaluation output across all policies for one run."""

    aggregate_grade: float = Field(
        description="Mean grade across all per-policy evaluations"
    )
    per_policy_eval: dict[str, PolicyGrade] = Field(
        description="Policy statement → PolicyGrade mapping"
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_EVALUATION_PROMPT = """\
YOU ARE AN EXPERT UNBIASED EVALUATOR GRADING EXTRACTED CLIMATE POLICIES.

SCORING GUIDE
  +1 (great)  — policy closely matches a ground-truth entry in scope, commitment, and specificity
   0 (okay)   — policy is loosely related to a ground-truth entry but lacks detail or precision
  -1 (bad)    — policy has no ground-truth match, is vague, or contradicts the document intent

CRITERIA: {criteria}

GROUND TRUTH POLICIES:
{ground_truth_policies}

INPUT POLICIES TO GRADE:
{policies}
"""

_EVALUATION_PROMPT_RLM = """\
YOU ARE AN EXPERT UNBIASED EVALUATOR GRADING EXTRACTED CLIMATE POLICIES.

SCORING GUIDE
  +1 (great)  — policy closely matches a ground-truth entry in scope, commitment, and specificity
   0 (okay)   — policy is loosely related to a ground-truth entry but lacks detail or precision
  -1 (bad)    — policy has no ground-truth match, is vague, or contradicts the document intent

CRITERIA: {criteria}

ORIGINAL CLIMATE DOCUMENT:
{climate_document}

GROUND TRUTH POLICIES:
{ground_truth_policies}

INPUT POLICIES TO GRADE:
{policies}
"""

_IMPROVE_PROMPT = """\
YOU ARE A PROMPT OPTIMIZER improving a climate-policy extraction system prompt.

Given per-policy grades and reasoning from an evaluation run, rewrite the old prompt
to fix patterns of failure (grade -1) while preserving what works (grade +1).
Return ONLY the improved prompt — no preamble, no explanation.

EVALUATION OUTPUT:
{evaluation_output}

OLD PROMPT:
{old_prompt}
"""


# ---------------------------------------------------------------------------
# Optimizer class
# ---------------------------------------------------------------------------


class RLMOptimizer:
    """
    Evaluates RLM-extracted policies against ground truth and improves the
    system prompt based on the evaluation feedback.

    RLM extraction must be run externally; pass the resulting policy list
    to evaluate() or step().

    Args:
        model: OpenAI model to use for evaluation and prompt improvement
    """

    def __init__(self, model: str = "gpt-5.2") -> None:
        self.model = model
        self._client: OpenAI | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def _parse_raw(self, raw: _RawEvaluation) -> EvaluationResult:
        """Convert the LLM's list-based output into an EvaluationResult."""
        per_policy = {pg.policy_statement: pg for pg in raw.per_policy_eval}
        grades = [pg.grade for pg in per_policy.values()]
        aggregate = sum(grades) / len(grades) if grades else 0.0
        return EvaluationResult(aggregate_grade=aggregate, per_policy_eval=per_policy)

    def _call_eval(self, prompt: str) -> EvaluationResult:
        """Call the LLM with JSON schema mode and parse the structured response."""
        response = self._get_client().beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=_RawEvaluation,
        )
        return self._parse_raw(response.choices[0].message.parsed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        policies: list[dict[str, Any]],
        gold_policies: list[dict[str, Any]],
        evaluation_criteria: str,
        climate_document: str | None = None,
    ) -> EvaluationResult:
        """
        Grade extracted policies against ground-truth policies.

        Args:
            policies: extracted policies to grade (output of RLM)
            gold_policies: ground-truth policies to compare against
            evaluation_criteria: description of what makes a good extraction
            climate_document: when provided, includes the source document for
                richer context

        Returns:
            EvaluationResult with aggregate_grade and per_policy_eval
        """
        if climate_document is not None:
            prompt = _EVALUATION_PROMPT_RLM.format(
                criteria=evaluation_criteria,
                climate_document=climate_document,
                ground_truth_policies=json.dumps(gold_policies, indent=2),
                policies=json.dumps(policies, indent=2),
            )
        else:
            prompt = _EVALUATION_PROMPT.format(
                criteria=evaluation_criteria,
                ground_truth_policies=json.dumps(gold_policies, indent=2),
                policies=json.dumps(policies, indent=2),
            )

        return self._call_eval(prompt)

    def improve_prompt(
        self,
        evaluation: EvaluationResult,
        current_prompt: str,
    ) -> str:
        """
        Rewrite current_prompt to address deficiencies found in evaluation.

        Args:
            evaluation: output from evaluate()
            current_prompt: the system prompt used during the last RLM run

        Returns:
            Improved prompt string
        """
        prompt = _IMPROVE_PROMPT.format(
            evaluation_output=evaluation.model_dump_json(indent=2),
            old_prompt=current_prompt,
        )
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def step(
        self,
        rlm_policies: list[dict[str, Any]],
        gold_policies: list[dict[str, Any]],
        evaluation_criteria: str,
        current_prompt: str,
        climate_document: str | None = None,
    ) -> tuple[str, EvaluationResult]:
        """
        Run one full evaluate-then-improve cycle.

        Evaluates the provided RLM policies and, if aggregate_grade > 0,
        returns an improved prompt. Otherwise returns current_prompt unchanged.

        Args:
            rlm_policies: policies extracted by RLM in the caller
            gold_policies: ground-truth policies
            evaluation_criteria: description of good vs. poor extraction
            current_prompt: system prompt used to produce rlm_policies
            climate_document: optional source document for richer evaluation

        Returns:
            (next_prompt, evaluation) — next_prompt is improved when score > 0,
            unchanged otherwise
        """
        evaluation = self.evaluate(
            policies=rlm_policies,
            gold_policies=gold_policies,
            evaluation_criteria=evaluation_criteria,
            climate_document=climate_document,
        )

        if evaluation.aggregate_grade > 0:
            next_prompt = self.improve_prompt(evaluation, current_prompt)
        else:
            next_prompt = current_prompt

        return next_prompt, evaluation
