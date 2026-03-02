# Climate Policy Extraction Pipeline

Extracts, validates, and classifies climate policies from policy documents (PDFs/markdown) using RLM and DSPy.

## Pipeline

```
Document (PDF/MD) → RLM extraction → DSPy validation → DSPy classification → ClimatePolicy JSON
```

1. **RLM extraction** — recursively processes long documents to pull candidate policy statements as structured JSON
2. **DSPy validation** — filters out vague/performative policies, keeping only actionable ones
3. **DSPy classification** — categorizes each policy into: Mitigation, Adaptation, Resource Efficiency, or Nature-Based Solutions

## Files

| File | Description |
|------|-------------|
| `dspy_rlm_pipeline.py` | Main pipeline using `dspy.RLM` for extraction |
| `base_rlm_pipeline.py` | Pipeline using the standalone `rlm` library |
| `control_rlm_pipeline.py` | Baseline pipeline without DSPy validation/classification |
| `RLM_proc_instr.pdf` | Expert extraction criteria injected as domain knowledge |

## Output

Each run produces two JSON files:
- `<output>.json` — validated and classified `ClimatePolicy` objects
- `<output>_rejected.json` — policies that failed validation with reasons

## Setup

```bash
pip install dspy docling pydantic python-dotenv
```

Set `OPENAI_API_KEY` in a `.env` file.

## Usage

Edit the `__main__` block in `dspy_rlm_pipeline.py` and run:

```bash
python dspy_rlm_pipeline.py
```
