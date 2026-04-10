# optimization/

Prompt optimization loop for the climate-policy RLM pipeline.

## Files

| File | Description |
|---|---|
| `evaluator.py` | LEAP Evaluator (Algorithm 1) — grades RLM output against GENIUS ground truth per location |
| `prompt_optimizer.py` | LEAP Prompt Optimizer (Algorithms 2 & 3) — per-category prompt update with negative signal guard |

---

## evaluator.py

Implements **Algorithm 1: LEAP Evaluator**. Groups extracted and ground-truth policies by category and role, runs optimal 1:1 Hungarian matching on embedding similarity, grades each matched pair via a grader LLM (+1 / 0 / -1), and penalizes unmatched policies on both sides.

```python
from optimization.evaluator import LEAPEvaluator

evaluator = LEAPEvaluator(model="gpt-5", embedding_model="text-embedding-3-small")

result = evaluator.evaluate(
    location="Seattle_US",
    extracted_policies=rlm_output,       # list of policy dicts from RLM
    ground_truth_policies=genius_output, # list of policy dicts from GENIUS
    rubric="Grade on specificity, commitment, and mechanism...",
    source_document=doc_markdown,        # optional, improves grading context
)

print(result.scores)   # {"Mitigation": 0.4, "Adaptation": -0.2, ...}
print(result.recall)   # {"Mitigation": 0.8, ...}
print(result.fpr)      # {"Mitigation": 0.1, ...}
```

### Output types

**`EvaluationOutput`**
| Field | Type | Description |
|---|---|---|
| `location` | `str` | Location key passed in |
| `scores` | `dict[str, float]` | Mean grade per category (includes -1 penalties) |
| `recall` | `dict[str, float]` | Matched GT count / total GT count per category |
| `fpr` | `dict[str, float]` | Unmatched extracted count / total extracted count per category |
| `grades` | `dict[str, PolicyGrade]` | Per-policy grades keyed by `'{category}::{policy_id}'` |

**`PolicyGrade`**
| Field | Type | Description |
|---|---|---|
| `policy_id` | `str` | Truncated policy statement used as identifier |
| `grade` | `Literal[-1, 0, 1]` | +1 match / 0 vague / -1 no match |
| `reasoning` | `str` | Step-by-step justification |

---

## prompt_optimizer.py

Implements **Algorithm 2: LEAP Prompt Optimizer** and **Algorithm 3: LEAP Prompt Optimization Loop**.

The RLM system prompt is decomposed into five independent prongs:

```
rho_t = ( rho_t^gen, rho_t^mit, rho_t^ada, rho_t^eff, rho_t^nbs )
```

Each prong is updated independently based on the per-location, per-category score delta. A prong reverts to its previous version when a negative signal is detected for that category, preventing a degraded prompt from being used as the base for further rewrites.

```python
from optimization.prompt_optimizer import LEAPPromptOptimizer, StructuredPrompt
from base_rlm_pipeline_v3 import CLIMATE_RLM_SYSTEM_PROMPT

# Bootstrap from existing flat prompt
initial = StructuredPrompt.from_flat(CLIMATE_RLM_SYSTEM_PROMPT)

optimizer = LEAPPromptOptimizer(model="gpt-5")

optimized = optimizer.run_loop(
    location="Seattle_US",
    extracted_policies_fn=lambda prompt: run_rlm(prompt, doc_md),  # wraps your RLM call
    ground_truth_policies=genius_policies,
    rubric="Grade on specificity, commitment, and mechanism...",
    initial_prompt=initial,
    source_document=doc_md,   # optional
    max_iterations=10,
    epsilon=0.01,
)

print(optimized.compose())   # full prompt string ready for the next RLM run
```

### StructuredPrompt

| Field | Prong | Description |
|---|---|---|
| `gen` | General | Policy definition, hierarchy rules, output format, what not to extract |
| `mit` | Mitigation | Mitigation-specific extraction and classification tips |
| `ada` | Adaptation | Adaptation-specific tips |
| `eff` | Resource Efficiency | Resource Efficiency-specific tips |
| `nbs` | Nature-Based Solutions | NBS-specific tips |

| Method | Description |
|---|---|
| `compose()` | Assemble all prongs into a single prompt string |
| `decompose(prompt)` | Parse a composed prompt string back into prongs |
| `from_flat(prompt)` | Bootstrap from a legacy flat prompt string (places content in `gen`) |

### Update rule (Algorithm 2)

For each category `c` at location `l`, iteration `t`:

```
Delta_{l,c} = S_t[l, c] - S_{t-1}[l, c]

if Delta_{l,c} > 0:  keep rho_t^c,     resample with F_{l,c}
else:                 revert rho_{t-1}^c, resample with F_{l,c}

Delta_l = mean_c( Delta_{l,c} )

if Delta_l > 0:  keep rho_t^gen,     resample with F_l
else:            revert rho_{t-1}^gen, resample with F_l
```

### Convergence (Algorithm 3)

The loop terminates when `max_c( |Delta_{l,c}| ) < epsilon` or `t >= T`.

---

## Environment

Requires `OPENAI_API_KEY` in `.env` or the environment.

```bash
pip install openai pydantic scipy numpy python-dotenv
```
