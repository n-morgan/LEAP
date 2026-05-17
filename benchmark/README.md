# LEAP Benchmark

Evaluates climate policy extraction systems against a 42-city ground-truth corpus.

## Quick Start

```bash
# Run one model on one city
python benchmark.py --configs configs/openai_gpt-5.5.yaml --cities Hiroshima

# Run one model on all cities
python benchmark.py --configs configs/openai_gpt-5.5.yaml

# Run all configs on all cities
python benchmark.py

# Skip RLM (faster)
python benchmark.py --skip-rlm
```

Or use `eval.py` for a single config/city with more verbose output:

```bash
python eval.py --config configs/openai_gpt-5.5.yaml --city Hiroshima
python eval.py --config configs/openai_gpt-5.5.yaml --all-cities
```

## Configs

| File | System | Model |
|------|--------|-------|
| `openai_gpt-5.5.yaml` | Direct completion | GPT-5.5 |
| `openai_gpt-5.4.yaml` | Direct completion | GPT-5.4 |
| `openai_gpt-5.4-mini.yaml` | Direct completion | GPT-5.4-mini |
| `openai_gpt-5.2.yaml` | Direct completion | GPT-5.2 |
| `rlm_gpt-5.5.yaml` | RLM recursive | GPT-5.5 |
| `rlm_gpt-5.2.yaml` | RLM recursive | GPT-5.2 |

## Output

Results are written to `results/{config_name}/{timestamp}/`:

| File | Contents |
|------|----------|
| `extracted_policies.csv` | Raw extracted policies, one row per policy |
| `scores.csv` | Aggregate metrics per city |
| `scores.jsonl` | Full evaluation output per city (one JSON per line) |
| `grades.csv` | Per-pair comparison: GT policy vs extracted policy with grade, similarity, and LLM reasoning |
| `traces/{location}/` | RLM reasoning traces (RLM runs only) |

## Key Metrics

- **composite** — weighted combination of F1, role agreement, category agreement, and +1 coverage
- **f1** — harmonic mean of precision and recall over matched policy pairs
- **+1_cov** — fraction of GT policies that received a +1 grade from the LLM grader
- **role_agr** — fraction of matched pairs where actor role matches GT
- **cat_agr** — fraction of matched pairs where primary category matches GT

## Data

All ground-truth data lives in `data/`:

- `all_cities_kept_classified_policies_final.csv` — annotated ground-truth policies
- `hex_location_to_final_run_csv.csv` — authoritative mapping from document hex ID to city key
- `pdf_markdown_cache/{hex}.md` — pre-converted markdown for each source document

The corpus covers 42 cities across North America, Europe, Asia, Africa, and Oceania. See `data/dataset.py` for the full list.

## Adding a City

1. Convert the PDF to markdown and drop it in `data/pdf_markdown_cache/`.
2. Add a row to `data/hex_location_to_final_run_csv.csv`.
3. Add ground-truth rows to `data/all_cities_kept_classified_policies_final.csv` with a matching `city` value.
4. Add an entry to `CORPUS` in `data/dataset.py`.
