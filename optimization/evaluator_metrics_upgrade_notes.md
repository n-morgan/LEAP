# Evaluator Metrics Upgrade Notes

This note summarizes the recent changes to `optimization/evaluator.py` and what the new metrics mean.

## What Changed

The evaluator used to match policies inside separate `primary_category` and `role` buckets. That meant a policy with the right statement but the wrong category or role could fail to match at all.

Now the evaluator matches policies globally within one location:

1. Embed every extracted `policy_statement`.
2. Embed every ground-truth `policy_statement`.
3. Run one Hungarian matching pass across all extracted vs. GT policies.
4. Keep only matches with cosine similarity `>= similarity_threshold` (`0.55` by default).
5. Grade each accepted match with the LLM grader.
6. Compute extraction, hierarchy, classification, and coverage metrics from those accepted matches.

This means category and role errors are now measured as errors after matching, instead of preventing a match.

## What Did Not Change

`prompt_optimizer.py` can still read the legacy fields:

- `scores`
- `recall`
- `fpr`
- `grades`
- `hierarchy_accuracy`

These are still returned for compatibility, but they are now derived from the global matching results.

## Metric Definitions

### `matched_count`

Number of extracted-to-GT pairs that passed both:

- Hungarian assignment
- similarity threshold

This is the number of policy pairs the evaluator considers valid statement-level matches.

### `unmatched_extracted_count`

Number of extracted policies that did not get an accepted GT match.

High value means the extractor may be over-producing spurious or borderline policies.

### `unmatched_ground_truth_count`

Number of GT policies that did not get an accepted extracted match.

High value means the extractor is missing policies from the golden set.

### `extraction_precision`

```text
matched_count / total_extracted
```

Answers: “Of everything the RLM extracted, how much matched the GT set?”

Low precision means too many extracted policies do not map cleanly to GT.

### `extraction_recall`

```text
matched_count / total_ground_truth
```

Answers: “Of everything in the GT set, how much did the RLM find?”

Low recall means the RLM is missing GT policies.

### `extraction_f1`

```text
2 * precision * recall / (precision + recall)
```

Balances precision and recall.

Useful because recall can look good even when the model extracts too many extra policies.

### `role_agreement`

```text
matched pairs with same role / matched_count
```

Checks whether matched policies agree on:

- `parent`
- `sub`
- `individual`

This is now meaningful because matching is no longer forced inside role buckets.

### `parent_attribution_accuracy`

```text
matched sub/sub pairs with same normalized parent_statement / matched sub/sub pairs
```

Only applies when both the extracted and GT policy are `sub` policies.

Answers: “Did the sub-policy point to the correct parent?”

Defaults to `1.0` when there are no matched sub/sub pairs, so irrelevant cases are not penalized.

### `primary_category_agreement`

```text
matched pairs with same primary_category / matched_count
```

Checks classification quality after statement matching.

This captures cases where the RLM found the right policy but labeled it as the wrong climate category.

### `financial_instrument_agreement`

Checks whether extracted and GT policies agree on financial-instrument labeling.

The evaluator normalizes both field names:

- extracted often uses `financial_instrument`
- GT may use `is_financial_instrument`

Values are normalized to `yes` / `no` before comparison.

### `secondary_category_agreement`

Checks whether `secondary_category` matches on accepted pairs where at least one side has a non-empty secondary category.

This is a secondary classification metric, not the main headline category score.

### `plus_one_coverage`

```text
accepted matches with LLM grade +1 / total_ground_truth
```

Answers: “How much of the GT set was covered by a fully correct extraction?”

This is stricter than recall. A policy can be matched for recall but still receive grade `0` if it is vague or incomplete.

### `composite_score`

Weighted headline score:

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

This gives one number for comparing prompt candidates while still balancing extraction, hierarchy, classification, and full-credit coverage.

## Legacy Metrics

### `scores`

Per-category mean grade.

Includes:

- LLM grades for matched GT policies in that category
- `-1` for unmatched GT policies
- `-1` for unmatched extracted policies in that category

### `recall`

Per-category matched GT count divided by total GT count in that category.

### `fpr`

Per-category unmatched extracted count divided by total extracted count in that category.

### `hierarchy_accuracy`

Legacy alias for `role_agreement`.

Kept so existing optimizer code keeps working.

## Practical Meaning

The evaluator now separates three questions:

1. Did the RLM find the right policy statements?
2. Did it assign the right hierarchy and categories?
3. Did the grader consider the match fully correct?

This should make prompt optimization easier to diagnose because wrong categories or roles no longer hide statement-level matches.
