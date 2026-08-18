# ForgeMM clean stratified val250 four-model evaluation (2026-08-06)

## 1. Question

Re-evaluate the frozen Qwen2.5-VL-3B-Instruct baseline and the three completed QLoRA variants on a larger, clean, stratified validation subset. The purpose is to determine whether the ordering observed on the old first-50 subset is reliable and whether B4 should be trained immediately.

This stage does not train a new model.

## 2. Controlled setup

The four runs use exactly the same:

- local Qwen2.5-VL-3B-Instruct base model;
- NF4 4-bit weights with FP16 compute;
- raw inference model stack, with an optional PEFT adapter loaded afterward;
- prompt, image pixel budget, greedy decoding, and `max_new_tokens=32`;
- fixed 250-row manifest and row order;
- normalized exact-match and ChartQA 5% numeric-relaxed scoring code.

Only the adapter changes:

| Run | Adapter scope | Training learning rate |
|---|---|---:|
| frozen_raw | none | — |
| B1 | language attention | `2e-4` |
| B2 | language attention + FFN | `2e-4` |
| B3 | language attention | `5e-5` |

Manifest: `artifacts/forgemm/chartqa_data_audit_v3_20260806/proposed_stratified_val250_manifest.csv`

Manifest SHA-256: `3a389596bcf9aa5fb161ca6aeb0e031c9f86dd229d92a1bce31ff4c090b93dad`

The manifest excludes the cross-split duplicate image hashes found by the data audit. Its task and answer-type labels are deterministic project heuristics, not official ChartQA annotations.

## 3. Engineering checks

Before the full evaluation, every model completed the same four-row smoke subset. All four smoke runs used the same manifest hash and inference configuration and produced 4/4 records without loading, CUDA, or scoring errors.

The full evaluator appends and flushes one JSONL record per sample. A strict resume check verifies that an existing prediction file is an exact prefix of the manifest, preventing a run from silently resuming on different data.

All four full runs completed 250/250 rows.

## 4. Main results

| Model | Exact correct | Exact accuracy | Delta vs raw | Relaxed correct | Relaxed accuracy | Delta vs raw |
|---|---:|---:|---:|---:|---:|---:|
| frozen_raw | 164/250 | 65.6% | — | 193/250 | 77.2% | — |
| B1 | 164/250 | 65.6% | 0.0 pp | 195/250 | 78.0% | +0.8 pp |
| B2 | 165/250 | 66.0% | +0.4 pp | 197/250 | 78.8% | +1.6 pp |
| B3 | 167/250 | 66.8% | **+1.2 pp** | 195/250 | 78.0% | +0.8 pp |

The old val50 ordering did not remain stable. On val50, every adapter was below the raw model in exact accuracy; on the clean val250 subset, B3 is numerically best on exact match and B2 is numerically best on relaxed accuracy. This confirms that the old 50-row result was too sample-sensitive for model selection.

## 5. Paired change analysis

### Normalized exact match

| Model | Fixed raw errors | Regressed raw correct | Net fixes | Exact two-sided paired sign-test p |
|---|---:|---:|---:|---:|
| B1 | 5 | 5 | 0 | 1.000 |
| B2 | 6 | 5 | +1 | 1.000 |
| B3 | 4 | 1 | **+3** | 0.375 |

### ChartQA relaxed accuracy

| Model | Fixed raw errors | Regressed raw correct | Net fixes | Exact two-sided paired sign-test p |
|---|---:|---:|---:|---:|
| B1 | 7 | 5 | +2 | 0.774 |
| B2 | 9 | 5 | **+4** | 0.424 |
| B3 | 4 | 2 | +2 | 0.688 |

B3 meets the predeclared exploration gate: exact accuracy is at least one percentage point above raw and the net fix count is positive. However, only five exact-match pairs are discordant, and `p=0.375`; this is not evidence of a stable improvement.

## 6. Category interpretation

B3's exact-match net gain is concentrated in a small number of categories:

- average: 6/21 to 8/21, +2 correct;
- sum: 12/23 to 14/23, +2 correct;
- direct lookup: 52/64 to 51/64, -1 correct;
- all other heuristic task categories have no net exact-count change.

By answer type, B3 gains three numeric answers and one percentage-numeric answer but loses one text answer. The direction is compatible with the project's chart-reasoning target, but the number of changed examples is still too small for a capability claim.

B2's wider adapter scope gives the highest relaxed score, but not the highest exact score. Its full evaluation took 480.6 seconds versus 288.7 seconds for B1 and 331.6 seconds for B3, with similar peak allocated CUDA memory (about 3.35–3.39 GiB). Average answer character counts are also similar (5.29 for B1, 5.34 for B2, and 5.46 for B3), so the slower B2 run must not be described as longer or more verbose output; the wider set of adapted modules is the more plausible engineering cost.

## 7. Decision

Do **not** train B4 yet.

The predeclared decision rule says that an adapter reaching at least +1 percentage point exact accuracy with positive net fixes must first enter a larger validation or multi-seed replication stage. B3 satisfies that gate, so immediately changing the training data would mix two questions:

1. whether B3's low-learning-rate attention QLoRA advantage is reproducible;
2. whether the clean joint-stratified train200 manifest improves training.

The next experiment should therefore replicate B3's recipe across additional deterministic training seeds while keeping the training data policy fixed. Evaluation must remain on this same val250 manifest. Only if the direction is reproducible should B4 isolate the data-composition variable using `proposed_stratified_train200_manifest.csv`.

No resume or portfolio claim should currently say that QLoRA definitively improves ChartQA by 1.2 percentage points. The supported statement is: on one fixed, clean, stratified 250-example subset, B3 produced a +1.2 pp exploratory exact-match difference with positive net fixes, but the paired difference was not statistically stable.

## 8. Artifacts

- evaluator: `scripts/forgemm_chartqa_manifest_eval.py`
- comparator: `scripts/forgemm_compare_manifest_evals.py`
- evaluation helpers: `src/forgellm/multimodal/evaluation.py`
- frozen predictions: `artifacts/forgemm/val250_frozen_raw_20260806/`
- B1 predictions: `artifacts/forgemm/val250_b1_attention_lr2e4_20260806/`
- B2 predictions: `artifacts/forgemm/val250_b2_attention_ffn_lr2e4_20260806/`
- B3 predictions: `artifacts/forgemm/val250_b3_attention_lr5e5_20260806/`
- final comparison report: `artifacts/forgemm/val250_comparison_v2_20260806/comparison_report.json`
- row-level comparison: `artifacts/forgemm/val250_comparison_v2_20260806/model_comparison.csv`

## 9. Claim boundary

These results cover one deterministic 250-example subset, one completed adapter per recipe, and one training seed per adapter. They are suitable for choosing the next experiment, not for a final benchmark or a stable improvement claim.
