"""
prompt_optimizer.py — LEAP Prompt Optimizer (Algorithms 2 & 3)

Per-location, per-category prompt update with negative signal guard.

The RLM system prompt is decomposed into five independent prongs:

    rho_t = ( rho_t^gen, rho_t^mit, rho_t^ada, rho_t^eff, rho_t^nbs )

where:
    rho^gen  — general extraction instructions (policy definition, hierarchy
               rules, output format, what not to extract)
    rho^mit  — Mitigation-specific extraction and classification tips
    rho^ada  — Adaptation-specific tips
    rho^eff  — Resource Efficiency-specific tips
    rho^nbs  — Nature-Based Solutions-specific tips

C = { Mitigation, Adaptation, Resource Efficiency, Nature-Based Solutions }

Per-category update rule (Algorithm 2) for location l at iteration t:

    Delta_{l,c} = S_t[l, c] - S_{t-1}[l, c]

    if Delta_{l,c} > 0:  keep rho_t^c,    resample with F_{l,c}
    else:                 revert rho_{t-1}^c, resample with F_{l,c}

General prong update:

    Delta_l = mean_c( Delta_{l,c} )

    if Delta_l > 0:  keep rho_t^gen,    resample with F_l
    else:            revert rho_{t-1}^gen, resample with F_l

The revert-on-negative-signal guard prevents a degraded prompt from being used
as the base for further rewrites. The LLM feedback is always passed regardless
of direction so each iteration incorporates the latest diagnostic signal.

Optimization loop (Algorithm 3):
    Repeat evaluate -> update until max(|Delta_{l,c}|) < epsilon or t >= T.

Usage:
    from prompt_optimizer import LEAPPromptOptimizer, StructuredPrompt
    from base_rlm_pipeline_v3 import CLIMATE_RLM_SYSTEM_PROMPT

    initial = StructuredPrompt.from_flat(CLIMATE_RLM_SYSTEM_PROMPT)
    optimizer = LEAPPromptOptimizer()

    optimized = optimizer.run_loop(
        location="Seattle_US",
        extracted_policies_fn=lambda prompt: run_rlm(prompt, doc_md),
        ground_truth_policies=genius_policies,
        rubric="Grade on specificity, commitment, and mechanism...",
        initial_prompt=initial,
        source_document=doc_md,
    )

    print(optimized.compose())   # full prompt string ready for RLM
"""

import os
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from openai import OpenAI

from evaluator import CATEGORIES, EvaluationOutput, LEAPEvaluator
from rlm_pipeline import CLIMATE_RLM_SYSTEM_PROMPT, run_rlm_for_optimizer

load_dotenv()

# Maps primary_category string to StructuredPrompt field name
_CATEGORY_TO_KEY: dict[str, str] = {
    "Mitigation":             "mit",
    "Adaptation":             "ada",
    "Resource Efficiency":    "eff",
    "Nature-Based Solutions": "nbs",
}

# Section headers used to compose/decompose the structured prompt string
_SECTION_HEADERS: dict[str, str] = {
    "gen": "## General",
    "mit": "## Mitigation",
    "ada": "## Adaptation",
    "eff": "## Resource Efficiency",
    "nbs": "## Nature-Based Solutions",
}


# ---------------------------------------------------------------------------
# StructuredPrompt
# ---------------------------------------------------------------------------


@dataclass
class StructuredPrompt:
    """
    Five-prong decomposition of the RLM system prompt.

    Each prong is a plain string containing the instructions relevant to that
    scope. Prongs are updated independently by the optimizer.

    Fields:
        gen  — general extraction tips applicable across all categories
        mit  — Mitigation-specific classification and extraction guidance
        ada  — Adaptation-specific guidance
        eff  — Resource Efficiency-specific guidance
        nbs  — Nature-Based Solutions-specific guidance
    """

    gen: str = ""
    mit: str = ""
    ada: str = ""
    eff: str = ""
    nbs: str = ""

    def compose(self) -> str:
        """
        Assemble all five prongs into a single prompt string using section headers.
        Empty prongs are omitted.
        """
        parts: list[str] = []
        for key, header in _SECTION_HEADERS.items():
            content = getattr(self, key).strip()
            if content:
                parts.append(f"{header}\n{content}")
        return "\n\n".join(parts)

    @classmethod
    def decompose(cls, prompt: str) -> "StructuredPrompt":
        """
        Parse a composed prompt string back into five prongs by locating
        section headers. Falls back to placing the full text in gen if no
        headers are found (e.g. legacy flat prompts).
        """
        positions: list[tuple[int, str]] = []
        for key, header in _SECTION_HEADERS.items():
            idx = prompt.find(header)
            if idx != -1:
                positions.append((idx, key))
        positions.sort()

        if not positions:
            return cls(gen=prompt.strip())

        sections: dict[str, str] = {k: "" for k in _SECTION_HEADERS}
        for i, (pos, key) in enumerate(positions):
            start = pos + len(_SECTION_HEADERS[key])
            end = positions[i + 1][0] if i + 1 < len(positions) else len(prompt)
            sections[key] = prompt[start:end].strip()

        return cls(**sections)

    @classmethod
    def from_flat(cls, prompt: str) -> "StructuredPrompt":
        """
        Bootstrap from a flat (non-decomposed) prompt string.
        Places the full content in gen and leaves category prongs empty.
        Use this when migrating from a legacy single-string system prompt.
        """
        return cls(gen=prompt.strip())


# ---------------------------------------------------------------------------
# Resample prompts
# ---------------------------------------------------------------------------

_RESAMPLE_SYSTEM = """\
You are a prompt optimizer rewriting one section of a climate policy extraction
system prompt. The section you receive is either the general extraction prong or
a category-specific prong (Mitigation, Adaptation, Resource Efficiency, or
Nature-Based Solutions).

You will receive:
    SECTION   — the current text of the prong to rewrite
    FEEDBACK  — per-policy grade reasoning from an evaluation run on this location

Your task: rewrite SECTION to fix patterns of failure (grade -1) while preserving
what already works (grade +1). Be specific. Do not add content unrelated to the
failures identified in FEEDBACK.

Return ONLY the rewritten prong text. Do not include the section header.
No preamble, no explanation.
"""

_RESAMPLE_USER = """\
SECTION:
{section}

FEEDBACK:
{feedback}
"""


# ---------------------------------------------------------------------------
# LEAPPromptOptimizer
# ---------------------------------------------------------------------------


class LEAPPromptOptimizer:
    """
    Implements Algorithm 2: LEAP Prompt Optimizer (per-location, per-category).

    Decomposes the RLM system prompt into five prongs and updates each
    independently based on per-location performance signals. Reverts a prong
    to the previous version when a negative signal is detected for that
    category or for the location overall.
    """

    def __init__(self, model: str = "gpt-5") -> None:
        self.model = model
        self._client: Optional[OpenAI] = None

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    # ------------------------------------------------------------------
    # Resample
    # ------------------------------------------------------------------

    def _resample(self, section: str, feedback: str) -> str:
        """
        Rewrite one prompt prong conditioned on grade reasoning feedback.
        Returns the rewritten prong text (no section header).
        """
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _RESAMPLE_SYSTEM},
                {"role": "user", "content": _RESAMPLE_USER.format(
                    section=section or "(empty — write from scratch based on feedback)",
                    feedback=feedback or "No specific feedback available.",
                )},
            ],
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Feedback formatting
    # ------------------------------------------------------------------

    def _format_feedback(
        self,
        evaluation: EvaluationOutput,
        category: Optional[str] = None,
    ) -> str:
        """
        Collect grade reasoning from the evaluation output.
        If category is provided, filter to grades from that category only.
        """
        lines: list[str] = []
        for key, grade in evaluation.grades.items():
            cat, policy_id = key.split("::", 1) if "::" in key else ("", key)
            if category and cat != category:
                continue
            lines.append(
                f"[{cat}] {policy_id[:70]}\n"
                f"  grade={grade.grade:+d}  {grade.reasoning}"
            )
        return "\n".join(lines) if lines else "No grade reasoning available for this scope."

    # ------------------------------------------------------------------
    # Algorithm 2 — single location update step
    # ------------------------------------------------------------------

    def update(
        self,
        location: str,
        current_prompt: StructuredPrompt,
        previous_prompt: StructuredPrompt,
        current_eval: EvaluationOutput,
        previous_scores: dict[str, float],
    ) -> StructuredPrompt:
        """
        Algorithm 2: update each prompt prong independently for one location.

        For each category c:
            Delta_{l,c} = S_t[l,c] - S_{t-1}[l,c]
            if Delta_{l,c} > 0  -> keep current prong,   resample with F_{l,c}
            else                -> revert previous prong, resample with F_{l,c}

        For the general prong:
            Delta_l = mean_c( Delta_{l,c} )
            if Delta_l > 0  -> keep current gen prong,   resample with full F_l
            else            -> revert previous gen prong, resample with full F_l

        Args:
            location:         location key (must match current_eval.location)
            current_prompt:   five-prong prompt used to produce current_eval
            previous_prompt:  five-prong prompt from the prior iteration
            current_eval:     EvaluationOutput from LEAPEvaluator for this location
            previous_scores:  S_{t-1}[l, :] — dict[category, float]

        Returns:
            Updated StructuredPrompt rho_{t+1}.
        """
        next_prompt = StructuredPrompt()
        deltas: dict[str, float] = {}

        # Per-category prong update
        for category in CATEGORIES:
            key = _CATEGORY_TO_KEY[category]
            s_t = current_eval.scores.get(category, 0.0)
            s_prev = previous_scores.get(category, 0.0)
            delta = s_t - s_prev
            deltas[category] = delta

            # Negative signal: revert to previous prong as the rewrite base
            base = getattr(current_prompt, key) if delta > 0 else getattr(previous_prompt, key)
            feedback = self._format_feedback(current_eval, category=category)
            setattr(next_prompt, key, self._resample(base, feedback))

            direction = "keep" if delta > 0 else "revert"
            print(f"  [{location}] {category:<30s}  delta={delta:+.3f}  ({direction})")

        # General prong update — use mean delta across categories for this location
        delta_l = sum(deltas.values()) / len(deltas) if deltas else 0.0
        gen_base = current_prompt.gen if delta_l > 0 else previous_prompt.gen
        full_feedback = self._format_feedback(current_eval)
        next_prompt.gen = self._resample(gen_base, full_feedback)

        gen_direction = "keep" if delta_l > 0 else "revert"
        print(f"  [{location}] {'General':<30s}  delta={delta_l:+.3f}  ({gen_direction})")

        return next_prompt

    # ------------------------------------------------------------------
    # Algorithm 3 — optimization loop
    # ------------------------------------------------------------------

    def run_loop(
        self,
        location: str,
        extracted_policies_fn: Callable[[str], list[dict[str, Any]]],
        ground_truth_policies: list[dict[str, Any]],
        rubric: str,
        initial_prompt: StructuredPrompt,
        source_document_path: Optional[pathlib.Path | str] = None,
        max_iterations: int = 10,
        epsilon: float = 0.01,
    ) -> StructuredPrompt:
        """
        Algorithm 3: LEAP Prompt Optimization Loop for a single location.

        Repeat:
            S_t, R_t, F_t, G_t  <- Evaluator(rho_t, D_l)
            rho_{t+1}            <- PromptOptimizer(rho_t, rho_{t-1}, S_t, S_{t-1}, ...)
        Until max(|S_t - S_{t-1}|) < epsilon or t >= T.

        Args:
            location:               location key (e.g. "Seattle_US")
            extracted_policies_fn:  callable that accepts a prompt string and returns
                                    extracted policy dicts (wraps the RLM run)
            ground_truth_policies:  GENIUS ground-truth policy dicts for this location
            rubric:                 freeform grading guidelines string
            initial_prompt:         starting StructuredPrompt (rho_0)
            source_document_path:   path to the markdown source document. When
                                    provided, grading is done via RLM so the full
                                    document can be traversed.
            max_iterations:         max optimization iterations T in [5, 10]
            epsilon:                convergence threshold on max per-category delta

        Returns:
            Optimized StructuredPrompt rho*.
        """
        evaluator = LEAPEvaluator(model=self.model)

        current_prompt = initial_prompt
        previous_prompt = initial_prompt
        previous_scores: dict[str, float] = {c: 0.0 for c in CATEGORIES}

        for t in range(max_iterations):
            print(f"\n{'=' * 60}")
            print(f"Iteration {t + 1}/{max_iterations}  |  Location: {location}")
            print(f"{'=' * 60}")

            # Run RLM extraction with the current composed prompt
            extracted = extracted_policies_fn(current_prompt.compose())
            print(f"  Extracted {len(extracted)} policies")

            # Evaluate against ground truth
            eval_result = evaluator.evaluate(
                location=location,
                extracted_policies=extracted,
                ground_truth_policies=ground_truth_policies,
                rubric=rubric,
                source_document_path=source_document_path,
            )

            # Report scores
            print(f"\n  Scores (S_t[l, c]):")
            for cat, score in eval_result.scores.items():
                delta = score - previous_scores.get(cat, 0.0)
                recall = eval_result.recall.get(cat, 0.0)
                fpr = eval_result.fpr.get(cat, 0.0)
                print(
                    f"    {cat:<30s}  score={score:+.3f}  "
                    f"delta={delta:+.3f}  recall={recall:.2f}  fpr={fpr:.2f}"
                )

            # Convergence check (skip on first iteration — no previous scores yet)
            if t > 0:
                max_delta = max(
                    abs(eval_result.scores.get(c, 0.0) - previous_scores.get(c, 0.0))
                    for c in CATEGORIES
                )
                if max_delta < epsilon:
                    print(f"\n  Converged: max_delta={max_delta:.4f} < epsilon={epsilon}")
                    return current_prompt

            # Update prompt prongs
            print(f"\n  Updating prompt prongs:")
            next_prompt = self.update(
                location=location,
                current_prompt=current_prompt,
                previous_prompt=previous_prompt,
                current_eval=eval_result,
                previous_scores=previous_scores,
            )

            previous_scores = dict(eval_result.scores)
            previous_prompt = current_prompt
            current_prompt = next_prompt

        print(f"\n  Reached max iterations ({max_iterations}). Returning current prompt.")
        return current_prompt


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import csv
    import pathlib

    _HERE = pathlib.Path(__file__).resolve().parent       # LEAP/optimization/
    _LEAP_ROOT = _HERE.parent                              # LEAP/
    _GENIUS_DOCS = _LEAP_ROOT.parent / "GENIUS" / "docs" / "cities"

    OUTPUTS_DIR = _HERE / "organized_outputs"
    SEATTLE_DOC = str(_GENIUS_DOCS / "seattle_markdown.md")

    def load_policies(path: pathlib.Path) -> list[dict]:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [row for row in reader if row.get("policy_statement", "").strip()]

    ground_truth_policies = load_policies(OUTPUTS_DIR / "structured_policies.csv")
    print(f"Loaded {len(ground_truth_policies)} ground-truth policies.")

    # Live RLM extraction: the optimizer calls this with the current composed
    # prompt at each iteration, re-runs the RLM, and scores the new output.
    # This is what closes the loop — each iteration uses a fresh RLM run under
    # the candidate prompt rather than a stale cached result.
    def extracted_policies_fn(prompt: str) -> list[dict]:
        return run_rlm_for_optimizer(
            prompt_string=prompt,
            document_path=SEATTLE_DOC,
        )

    DEFAULT_RUBRIC = (
        "Grade on specificity (quantified targets, deadlines, mechanisms), "
        "commitment strength (binding vs aspirational language), "
        "and accuracy relative to the source document."
    )

    # Seed from the full current system prompt so optimization starts from a
    # working baseline rather than a blank slate.
    initial_prompt = StructuredPrompt.from_flat(CLIMATE_RLM_SYSTEM_PROMPT)

    optimizer = LEAPPromptOptimizer()
    optimized = optimizer.run_loop(
        location="Seattle_US",
        extracted_policies_fn=extracted_policies_fn,
        ground_truth_policies=ground_truth_policies,
        rubric=DEFAULT_RUBRIC,
        initial_prompt=initial_prompt,
        source_document_path=SEATTLE_DOC,
        max_iterations=5,
    )

    print("\n=== Optimized Prompt ===")
    print(optimized.compose())
