"""Robustness tests for the RLM grader's JSON parser.

These exercise ``_parse_grader_json`` directly against three adversarial inputs:
  1. Clean JSON (the ideal case).
  2. JSON with a chatty preamble (model couldn't help itself).
  3. JSON containing braces inside the reasoning string.
Plus a couple of plain failure modes to confirm we get None.
"""

import pytest

from evaluator import _parse_grader_json


def test_clean_json():
    raw = '{"grade": 1, "reasoning": "fits"}'
    out = _parse_grader_json(raw)
    assert out["grade"] == 1
    assert out["reasoning"] == "fits"


def test_json_with_preamble():
    raw = (
        "Sure! Here's the grade you asked for:\n"
        '```json\n{"grade": -1, "reasoning": "no match"}\n```\n'
    )
    out = _parse_grader_json(raw)
    assert out is not None
    assert out["grade"] == -1
    assert out["reasoning"] == "no match"


def test_json_with_braces_in_reasoning():
    raw = (
        "Some intro\n"
        '{"grade": 0, "reasoning": "the policy mentions {targets} but no deadline"}'
    )
    out = _parse_grader_json(raw)
    assert out is not None
    assert out["grade"] == 0
    assert "targets" in out["reasoning"]


def test_unparseable_returns_none():
    assert _parse_grader_json("totally not json, no braces here") is None


def test_empty_returns_none():
    assert _parse_grader_json("") is None


def test_extended_schema_preserved():
    raw = (
        '{"grade": 1, "reasoning": "ok", "statement_match": "match", '
        '"role_match": true, "category_match": false}'
    )
    out = _parse_grader_json(raw)
    assert out["statement_match"] == "match"
    assert out["role_match"] is True
    assert out["category_match"] is False
