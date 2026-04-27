"""Unit tests for :class:`llm_evaluator.LLMEvaluator`.

Mocks embedder and OpenAI client so tests stay offline and fast.
"""

import csv
import hashlib
import pathlib
from unittest.mock import patch

import numpy as np
import pytest

from llm_evaluator import LLMEvaluator
from evaluator import _GraderOutput
from metrics import EvaluationBundle


_HERE = pathlib.Path(__file__).resolve().parent
_OUTPUTS_DIR = _HERE.parent / "organized_outputs"


def _load(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [
            row for row in csv.DictReader(fh)
            if row.get("policy_statement", "").strip()
        ]


def _hash_embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    out = np.zeros((len(texts), 32), dtype=np.float32)
    for i, t in enumerate(texts):
        digest = hashlib.sha256((t or "").encode("utf-8")).digest()
        out[i] = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
    return out


def _canned_output() -> _GraderOutput:
    return _GraderOutput(
        grade=1,
        reasoning="ok",
        statement_match="match",
        role_match=True,
        category_match=True,
    )


class _Msg:
    def __init__(self, parsed: _GraderOutput) -> None:
        self.parsed = parsed


class _Choice:
    def __init__(self, parsed: _GraderOutput) -> None:
        self.message = _Msg(parsed)


class _ParseResult:
    def __init__(self, parsed: _GraderOutput) -> None:
        self.choices = [_Choice(parsed)]


class _FakeCompletions:
    def __init__(self, cap: list | None) -> None:
        self._captured: list | None = cap
        self._out = _canned_output()

    def parse(self, **kwargs):
        if self._captured is not None and kwargs.get("messages"):
            self._captured.append(kwargs["messages"][-1]["content"])
        return _ParseResult(self._out)


class _FakeChat:
    def __init__(self, cap: list | None) -> None:
        self.completions = _FakeCompletions(cap)


class _FakeClient:
    def __init__(self, cap: list | None = None) -> None:
        self.beta = type("Beta", (), {})()
        self.beta.chat = _FakeChat(cap)


@pytest.mark.skipif(
    not (_OUTPUTS_DIR / "rlm_seattle_policies.csv").exists()
    or not (_OUTPUTS_DIR / "structured_policies.csv").exists(),
    reason="Seattle organized_outputs CSVs not present",
)
def test_llm_evaluator_smoke(monkeypatch, tmp_path):
    rlm_policies = _load(_OUTPUTS_DIR / "rlm_seattle_policies.csv")
    gt_policies = _load(_OUTPUTS_DIR / "structured_policies.csv")

    cap: list = []
    ev = LLMEvaluator(similarity_threshold=0.0)
    monkeypatch.setattr(ev, "_embed", _hash_embed)
    monkeypatch.setattr(ev, "_get_client", lambda: _FakeClient(cap))

    p = tmp_path / "src.md"
    p.write_text("# Doc\n", encoding="utf-8")
    bundle = ev.evaluate(
        location="Seattle_US",
        extracted_policies=rlm_policies,
        ground_truth_policies=gt_policies,
        rubric="test rubric",
        source_document_path=p,
    )

    assert isinstance(bundle, EvaluationBundle)
    assert bundle.location == "Seattle_US"
    assert 0.0 <= bundle.extraction.f1 <= 1.0
    assert 0.0 <= bundle.composite_score <= 1.0
    assert len(bundle.matching.matched) > 0
    assert cap, "grader was invoked at least once"
    assert "Not provided." in cap[0], "default should ignore source document in grader user message"


def test_llm_evaluator_rejects_legacy_flag():
    with pytest.raises(ValueError, match="always uses the new bundle"):
        LLMEvaluator(use_new_evaluator=False)


def test_llm_evaluator_grade_pair_rlm_raises():
    ev = LLMEvaluator()
    with pytest.raises(NotImplementedError, match="does not use the RLM grader"):
        ev._grade_pair_rlm(
            {"policy_statement": "a"},
            {"policy_statement": "b"},
            "r",
            "doc",
        )


def test_llm_evaluator_never_calls_rlm_path(monkeypatch, tmp_path):
    ev = LLMEvaluator(similarity_threshold=0.0, include_source_document=True)
    monkeypatch.setattr(ev, "_embed", _hash_embed)
    monkeypatch.setattr(ev, "_get_client", lambda: _FakeClient())

    ext = [
        {
            "policy_statement": "Cut emissions 50% by 2030",
            "role": "individual",
            "primary_category": "Mitigation",
        }
    ]
    gt = [dict(ext[0])]

    p = tmp_path / "doc.md"
    p.write_text("long doc " * 100, encoding="utf-8")

    def rlm_boom(*_a, **_k):
        raise RuntimeError("RLM should not be called")

    with patch.object(LLMEvaluator, "_grade_pair_rlm", side_effect=rlm_boom):
        bundle = ev.evaluate(
            location="T",
            extracted_policies=ext,
            ground_truth_policies=gt,
            rubric="r",
            source_document_path=p,
        )

    assert isinstance(bundle, EvaluationBundle)
    assert len(bundle.matching.matched) == 1


def test_llm_evaluator_document_slot_when_enabled(monkeypatch, tmp_path):
    cap: list = []
    ev = LLMEvaluator(similarity_threshold=0.0, include_source_document=True)
    monkeypatch.setattr(ev, "_embed", _hash_embed)
    monkeypatch.setattr(ev, "_get_client", lambda: _FakeClient(cap))

    ext = [
        {
            "policy_statement": "x",
            "role": "individual",
            "primary_category": "Mitigation",
        }
    ]
    p = tmp_path / "d.md"
    p.write_text("UNIQUE_SOURCE_MARKER", encoding="utf-8")
    ev.evaluate(
        location="L",
        extracted_policies=ext,
        ground_truth_policies=list(ext),
        rubric="rub",
        source_document_path=p,
    )
    assert cap
    assert "UNIQUE_SOURCE_MARKER" in cap[0]

    cap2: list = []
    ev2 = LLMEvaluator(similarity_threshold=0.0, include_source_document=False)
    monkeypatch.setattr(ev2, "_embed", _hash_embed)
    monkeypatch.setattr(ev2, "_get_client", lambda: _FakeClient(cap2))
    ev2.evaluate(
        location="L",
        extracted_policies=ext,
        ground_truth_policies=list(ext),
        rubric="rub",
        source_document_path=p,
    )
    assert cap2
    assert "Not provided." in cap2[0]
    assert "UNIQUE_SOURCE_MARKER" not in cap2[0]
