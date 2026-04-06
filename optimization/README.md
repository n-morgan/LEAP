# optimization/

Prompt optimization loop for the climate-policy RLM pipeline.

## Overview

`rlm_optimizer.py` evaluates RLM-extracted policies against ground truth and iteratively improves the RLM system prompt. RLM extraction is kept **separate** — run it in your own script and pass the results in.

## Usage

```python
from optimization.rlm_optimizer import RLMOptimizer

optimizer = RLMOptimizer(model="gpt-5.2")

current_prompt = CLIMATE_RLM_SYSTEM_PROMPT
for run in range(max_runs):
    policies = run_rlm(current_prompt, doc_markdown)   # your RLM call
    current_prompt, evaluation = optimizer.step(
        rlm_policies=policies,
        gold_policies=gt_policies,
        evaluation_criteria=criteria,
        current_prompt=current_prompt,
        climate_document=doc_markdown,  # optional — enables richer evaluation
    )
    print(f"Run {run + 1} score: {evaluation.aggregate_grade:.3f}")
```

## API

### `RLMOptimizer(model)`
| Method | Description |
|---|---|
| `evaluate(policies, gold_policies, criteria, climate_document=None)` | Grade each extracted policy against ground truth. Returns `EvaluationResult`. |
| `improve_prompt(evaluation, current_prompt)` | Rewrite `current_prompt` to address grade-(-1) failures. Returns improved prompt string. |
| `step(rlm_policies, gold_policies, criteria, current_prompt, ...)` | One full cycle: evaluate → improve if score > 0. Returns `(next_prompt, EvaluationResult)`. |

### Output types
- **`EvaluationResult`** — `aggregate_grade: float`, `per_policy_eval: dict[str, PolicyGrade]`
- **`PolicyGrade`** — `grade: Literal[-1, 0, 1]`, `reasoning: str`

Evaluation uses OpenAI structured output (`beta.chat.completions.parse`) with the `_RawEvaluation` JSON schema enforced at the API level.

## Environment

Requires `OPENAI_API_KEY` in `.env` or the environment.
