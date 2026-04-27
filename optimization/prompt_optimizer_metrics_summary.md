# Prompt Optimizer Metrics Summary

This summary converts the two Cursor canvases into a shareable Markdown note:

- `optimization-logs.canvas.tsx`: overview of optimization notebook runs.
- `prompt-optimizer-metrics-summary.canvas.tsx`: focused summary of run `v2_2026-04-23T15-44-06`.

## Executive Summary

The prompt optimizer is currently good at recall, but weak at precision, hierarchy, and classification.

For the focused run `optimization/logs/notebook/v2_2026-04-23T15-44-06`, the accepted metrics did not improve from iteration 1 to iteration 2. Every candidate prompt rewrite was rejected by the acceptance guardrails.

Current accepted score:

| Metric | Value | What It Shows |
| --- | ---: | --- |
| Composite score | 0.3803 | Overall weighted optimizer score. Low-to-moderate. |
| Extraction F1 | 0.7083 | Extraction coverage is decent because recall is perfect, but precision is weak. |
| Extraction precision | 0.5484 | 17 of 31 extracted policies matched the golden set. |
| Extraction recall | 1.0000 | All 17 golden-set policies were matched. |
| Hierarchy role agreement | 0.4706 | Less than half of matched policies have the correct role. |
| Parent attribution | 0.0000 | Sub-policy parent links are not resolving to the matched golden parents. |
| Primary category agreement | 0.3529 | Classification is the biggest weak spot. |
| Financial instrument agreement | 0.0000 | Likely affected by a field-name mismatch: GT uses `is_financial_instrument`. |
| Secondary category agreement | 0.5882 | Moderate, but not a headline acceptance metric. |
| +1 coverage | 0.0588 | Only 1 of 17 golden policies received a full +1 grade. |

## How Similarity To The Golden Set Is Computed

The optimizer compares extracted policies to the golden set in two stages.

First, it performs embedding-based matching:

1. Take every extracted `policy_statement`.
2. Take every golden-set `policy_statement`.
3. Embed both sets with `text-embedding-3-small`.
4. Build a cosine-similarity matrix between extracted and golden policies.
5. Run Hungarian matching to find the best 1-to-1 assignment.
6. Drop assigned pairs below the similarity threshold, currently `0.55`.

This matching is category-agnostic. That means an extracted policy can still match its golden-set counterpart even if the extracted `primary_category` is wrong. This is useful because classification errors do not prevent statement-level matching.

Second, matched pairs are LLM-graded:

- `+1`: extracted policy matches the golden policy in scope, commitment, and specificity.
- `0`: directionally correct, but vague, imprecise, missing details, or has field mismatches.
- `-1`: no meaningful match, hallucination, contradiction, or wrong policy.

The LLM grader also emits:

- `statement_match`: whether the policy statement itself matches.
- `role_match`: whether `parent`, `sub`, or `individual` matches.
- `category_match`: whether `primary_category` matches.

## Focused Run: `v2_2026-04-23T15-44-06`

### Matching Snapshot

| Signal | Value | Interpretation |
| --- | ---: | --- |
| Matched golden policies | 17 / 17 | Recall is perfect against this dev aggregate. |
| Unmatched golden policies | 0 | No golden policies were missed. |
| Unmatched extracted policies | 14 | The extractor over-produces spurious or borderline policies. |
| Similarity threshold | 0.55 | Embedding matches below this cosine score are dropped. |
| +1 coverage | 1 / 17 | Only one matched policy is judged fully correct. |

The model is finding all golden policies, but it is also extracting too many extra policies. Many matched policies are only partial matches because the statement is vague, the hierarchy role is wrong, or the primary category differs from the golden set.

### Iteration Results

| Iteration | Composite | Extraction F1 | Precision | Recall | Role Agreement | Parent Attribution | Primary Category | +1 Coverage | Triggered Prongs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.3803 | 0.7083 | 0.5484 | 1.0000 | 0.4706 | 0.0000 | 0.3529 | 0.0588 | classification, extraction, hierarchy |
| 2 | 0.3803 | 0.7083 | 0.5484 | 1.0000 | 0.4706 | 0.0000 | 0.3529 | 0.0588 | classification, hierarchy |

Iteration 2 is unchanged because no candidate from iteration 1 was accepted. Extraction was no longer triggered in iteration 2 because extraction recall/F1 did not worsen, but hierarchy and classification remained below target.

### Candidate Outcomes

| Candidate | Composite | Delta | Decision |
| --- | ---: | ---: | --- |
| iter1 extraction | 0.4536 | +0.0733 | Rejected: extraction F1 dropped by 0.148, exceeding the floor. |
| iter1 hierarchy | 0.3646 | -0.0157 | Rejected: insufficient improvement. |
| iter1 classification | 0.3527 | -0.0276 | Rejected: insufficient improvement. |
| iter2 hierarchy | 0.4200 | +0.0398 | Rejected: hierarchy role agreement dropped by 0.412, exceeding the floor. |
| iter2 classification | 0.3494 | -0.0308 | Rejected: insufficient improvement. |

The important pattern is that some candidates improved the composite score, but caused a major regression in a protected metric. The guardrails correctly rejected those candidates.

## All Runs Overview

The optimization-log canvas covered 24 notebook runs from `optimization/logs/notebook/`. Only 6 runs had usable summarized iteration/candidate data in the canvas.

| Run | Iterations | Final Composite | Candidates | Status |
| --- | ---: | ---: | ---: | --- |
| `v2_2026-04-23T00-58-38` | 1 | 0.4853 | 1 accepted / 3 total | Ran; one accepted extraction candidate |
| `v2_2026-04-23T13-19-22` | 1 | 0.3634 | 0 accepted / 1 total | Candidate rejected by extraction F1 cliff |
| `v2_2026-04-23T13-43-57` | 1 | 0.4645 | 0 accepted / 3 total | Candidates rejected by insufficient improvement |
| `v2_2026-04-23T15-44-06` | 2 | 0.3803 | 0 accepted / 5 total | Focused run; no accepted candidates |
| `v2_2026-04-23T20-25-02` | 2 | 0.5393 | 0 accepted / 2 total | Best final composite in canvas, but high variance |

Across the summarized runs:

- Total runs listed: 24.
- Runs with usable data in the canvas: 6.
- Accepted candidates: 1.
- Rejected candidates: 13.
- Best final composite shown: `0.5393` for run `v2_2026-04-23T20-25-02`.
- The only accepted candidate shown was an extraction candidate in run `v2_2026-04-23T00-58-38`.

## Metric Definitions

### Composite Score

The composite score is the weighted headline score used to compare prompts:

```text
0.40 * extraction_f1
+ 0.25 * hierarchy_headline
+ 0.25 * primary_category_agreement
+ 0.10 * plus_one_coverage
```

Where:

```text
hierarchy_headline = (role_agreement + parent_attribution_accuracy) / 2
```

Higher is better. The score is intended to balance extraction quality, hierarchy correctness, classification correctness, and fully correct policy coverage.

### Extraction Precision

Precision measures how many extracted policies were valid matches:

```text
precision = matched_extracted / total_extracted
```

In the focused run:

```text
17 matched / 31 extracted = 0.5484
```

Low precision means the prompt is extracting too many policies that do not map cleanly to the golden set.

### Extraction Recall

Recall measures how many golden policies were found:

```text
recall = matched_golden / total_golden
```

In the focused run:

```text
17 matched / 17 golden = 1.0000
```

High recall means the prompt is not missing golden policies.

### Extraction F1

F1 balances precision and recall:

```text
F1 = 2 * precision * recall / (precision + recall)
```

In the focused run, recall is perfect but precision is weak, so F1 lands at `0.7083`.

### Category Distribution JSD

JSD means Jensen-Shannon distance between the extracted category distribution and the golden category distribution.

Lower is better. A high value means the extracted set has a different category mix than the golden set. In the focused run, `0.6308` suggests the extracted categories are substantially misaligned with the golden set.

### Role Agreement

Role agreement is the fraction of matched pairs where the extracted hierarchy role equals the golden role:

```text
role_agreement = matched_pairs_with_same_role / matched_pairs
```

Roles are:

- `parent`: umbrella policy with sub-actions.
- `sub`: specific action under a parent.
- `individual`: standalone policy.

In the focused run, role agreement is `0.4706`, so hierarchy role assignment is poor.

### Parent Attribution Accuracy

Parent attribution accuracy applies to matched sub-policies. It checks whether the extracted `parent_statement` points to the same matched parent as the golden sub-policy.

In the focused run, parent attribution is `0.0000`. This means the model is not linking sub-policies back to the correct parent policies, even when it extracts the sub-policies themselves.

### Primary Category Agreement

Primary category agreement is the fraction of matched pairs where extracted `primary_category` equals golden `primary_category`.

In the focused run, this is `0.3529`. The biggest pattern is golden `Mitigation` policies being extracted as `Resource Efficiency`, with some also mapped to `Nature-Based Solutions` or `Adaptation`.

### Financial Instrument Agreement

Financial instrument agreement checks whether extracted financial-instrument labeling matches the golden set.

There is an important caveat: the current metric code appears to compare `ground_truth["financial_instrument"]` to `extracted["financial_instrument"]`, while the golden data shown in the bundle uses `is_financial_instrument`. That likely explains why this metric is `0.0000` and may mean the metric is undercounting true agreement.

### Secondary Category Agreement

Secondary category agreement is the fraction of matched pairs where the secondary category matches the golden secondary category.

In the focused run, this is `0.5882`. It is better than primary category agreement, but it is not one of the main acceptance headline metrics.

### +1 Coverage

`+1 coverage` measures how many golden policies are represented by a fully correct matched extraction:

```text
+1 coverage = matched_pairs_with_grade_+1 / total_golden
```

In the focused run:

```text
1 / 17 = 0.0588
```

This is low. It means most policies are being found, but only partially or incorrectly represented.

## What The Current Results Mean

The current prompt is too broad and too loose. It captures the golden policies, but it also extracts many extra policies and often assigns the wrong hierarchy role or primary category.

The strongest immediate improvement areas are:

1. Tighten extraction criteria to reduce spurious/borderline policies without losing recall.
2. Improve hierarchy instructions so named grouped initiatives become parents and listed actions become subs.
3. Improve classification rules, especially around when waste, energy efficiency, and transportation policies should remain `Mitigation` instead of being classified as `Resource Efficiency`.
4. Fix or normalize the financial-instrument field-name mismatch before relying on that metric.

The guardrails are working as intended: candidates that improve composite but damage extraction F1 or hierarchy role agreement are rejected.
