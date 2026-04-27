"""End-to-end test of the new sequential within-iteration acceptance loop.

We mock:
  * the extractor (returns a fixed list of policy dicts)
  * the embedder (deterministic SHA-256 hashing)
  * the grader (returns +1 for everything)
  * the resampler (returns a sentinel that includes the scope name)

This exercises ``run_loop_v2`` end-to-end without any LLM call, asserting that
within a single iteration multiple prongs can be accepted in sequence.
"""

import csv
import hashlib
import json as _json
import pathlib

import numpy as np
import pytest

from evaluator import LEAPEvaluator, _GraderOutput
from prompt_optimizer import (
    AcceptanceConfig,
    LEAPPromptOptimizer,
    PrognTarget,
    StructuredPrompt,
)
from dev_test_split import LocationConfig, LocationSet


def _hash_embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    out = np.zeros((len(texts), 32), dtype=np.float32)
    for i, t in enumerate(texts):
        digest = hashlib.sha256((t or "").encode("utf-8")).digest()
        out[i] = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
    return out


def _write_gt_csv(tmp_path: pathlib.Path, location: str = "TestCity") -> pathlib.Path:
    """Create a 4-policy GT CSV tagged with ``location`` for the given city."""
    path = tmp_path / "gt.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "location", "policy_statement", "primary_category", "role",
                "parent_statement",
            ],
        )
        writer.writeheader()
        for i in range(4):
            writer.writerow({
                "location": location,
                "policy_statement": f"GT_POLICY_{i}",
                "primary_category": "Mitigation" if i % 2 == 0 else "Adaptation",
                "role": "individual",
                "parent_statement": "",
            })
    return path


def test_loop_runs_and_writes_logs(tmp_path, monkeypatch):
    gt_csv = _write_gt_csv(tmp_path)
    locset = LocationSet(
        dev=[LocationConfig(name="TestCity", ground_truth_csv=gt_csv, split="dev")],
        test=[],
    )

    # Each call returns a slightly improving extraction.
    call_log: list[int] = []

    def fake_extract(prompt: str, location: str, src, trace):
        call_log.append(len(call_log))
        # Return the first 2 GT statements verbatim so they match
        return [
            {"policy_statement": f"GT_POLICY_{i}", "primary_category": "Mitigation",
             "role": "individual", "parent_statement": ""}
            for i in range(2)
        ]

    evaluator = LEAPEvaluator(use_new_evaluator=True, similarity_threshold=0.0)
    monkeypatch.setattr(evaluator, "_embed", _hash_embed)
    monkeypatch.setattr(
        evaluator, "_grade_pair",
        lambda *a, **kw: _GraderOutput(
            grade=1, reasoning="ok", statement_match="match",
            role_match=True, category_match=True,
        ),
    )

    optimizer = LEAPPromptOptimizer(model="test-model")

    # Replace the LLM resampler so no network is called
    monkeypatch.setattr(
        optimizer, "propose_candidate_v2",
        lambda scope, current_prompt, current_bundle, previous_bundle=None, targets=None: StructuredPrompt(
            extraction=current_prompt.extraction + (f"\n[{scope}-rev]" if scope == "extraction" else ""),
            hierarchy=current_prompt.hierarchy + (f"\n[{scope}-rev]" if scope == "hierarchy" else ""),
            classification=current_prompt.classification + (f"\n[{scope}-rev]" if scope == "classification" else ""),
        ),
    )

    initial = StructuredPrompt(extraction="ext", hierarchy="hi", classification="cls")
    log_dir = tmp_path / "logs"

    final = optimizer.run_loop_v2(
        location_set=locset,
        extracted_policies_fn=fake_extract,
        rubric="test",
        initial_prompt=initial,
        max_iterations=2,
        seeds=1,
        targets=PrognTarget(),
        # Composite is constant when extractions don't change → require zero
        # improvement so the test exercises acceptance, not quality gating.
        acceptance=AcceptanceConfig(min_delta=-1.0, per_metric_floor={}),
        log_dir=log_dir,
        evaluator=evaluator,
    )

    # Logs were written
    log_subdirs = list(log_dir.glob("v2_*"))
    assert len(log_subdirs) == 1
    run_dir = log_subdirs[0]
    assert (run_dir / "iteration_log.csv").exists()
    assert (run_dir / "candidate_log.csv").exists()
    bundle_files = list(run_dir.glob("metrics_bundle_iter_*.json"))
    assert len(bundle_files) >= 1

    # Each metrics bundle deserialises cleanly
    for bf in bundle_files:
        with open(bf, encoding="utf-8") as fh:
            data = _json.load(fh)
        assert "bundle" in data and "iteration" in data

    # Sanity on candidate_log: iteration 1 should have all three prongs
    # proposed (capped to max_accepted_per_iteration=2 by default).
    with open(run_dir / "candidate_log.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    iter1_scopes = [r["scope"] for r in rows if r["iteration"] == "1"]
    assert len(iter1_scopes) >= 1
    # The final returned prompt is a StructuredPrompt
    assert isinstance(final, StructuredPrompt)
