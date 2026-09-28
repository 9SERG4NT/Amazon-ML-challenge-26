# Amazon ML Challenge 2026: all approaches and their results

Team: Satvik Barhanpure, Sumukh Chourasia, Jatin Patin, Parambrata Sanyal

All full-data numbers come from the Kaggle run reports. **Validation** means 220,140 training S1 entities held out from every model (10% of train). Every full-data variant is scored on this same set, so the numbers are directly comparable. Metric: macro F0.5 per S1 entity, singletons included.

## Summary

| Area | Approaches tried | Kept |
|---|---|---|
| Overall design | TF-IDF blocking + LightGBM; learned blocking + XGBoost | learned blocking + XGBoost |
| Candidate generation | percentile rule; absolute rule; + graph filter; + hard-negative encoder | percentile rule + graph filter + hard-negative encoder |
| Matcher features | strings + IDF; + cross-encoder; + margins; + name ambiguity | all of them |
| Match selection | global thresholds; per-entity expected F0.5 | whichever wins on validation (expected F0.5 in the final run) |
| France (unseen country) | none; per-country IDF/percentiles; orphan-rate calibration; empty-share calibration | per-country features + empty-share calibration |
| **Final** | | **validation macro F0.5 0.9875, 5.95 test candidates per S1** |

## 1. Overall design

| # | Design | Data | Result | Decision |
|---|---|---|---|---|
| A | TF-IDF token blocking (name, address, joined-name keys) + LightGBM matcher | 5% sample (earlier AWS session) | blocking recall 96.8%, validation F0.5 0.971 | replaced: larger candidate sets (about 20 per source per S1) and lower recall |
| B | Learned blocking (fine-tuned bi-encoder + FAISS IVF, record-to-entity search) + XGBoost | full data | validation F0.5 0.9875 | **final** |

## 2. Candidate generation (blocking)

The bi-encoder is `intfloat/multilingual-e5-small` (MIT, 118M parameters). It is fine-tuned on 2M (S1, matching record) pairs with an in-batch contrastive loss; half of the batches come from one city, which makes the negatives hard. A FAISS IVF index, searched separately per country label, gives each S2/S3 record its 3 closest S1 records and each S1 record its 10 closest records. A rule, and optionally a filter, then picks the candidates. Each rule is tuned for the smallest candidate set whose F0.5 ceiling (a perfect matcher on those candidates) stays within 0.002 of the best ceiling on the grid.

| Variant | Validation recall | Candidates per S1 (validation) | F0.5 ceiling | Test candidates per S1 |
|---|---:|---:|---:|---:|
| Round 1 encoder: all neighbour-graph pairs | 99.06% | 18.28 | 0.9973 | – |
| Round 1 encoder: rule, absolute similarity cut-off | 98.33% | 5.28 | 0.9953 | 6.51 |
| Round 1 encoder: rule, per-country percentile cut-off | 98.36% | 5.39 | 0.9952 | 6.81 |
| Round 2 encoder (hard negatives): all neighbour-graph pairs | 99.03% | 18.11 | 0.9973 | – |
| Round 2 encoder (hard negatives): rule, absolute similarity cut-off | 98.33% | 5.04 | 0.9952 | 6.11 |
| Round 2 encoder (hard negatives): rule, per-country percentile cut-off | 98.47% | 5.31 | 0.9955 | 6.59 |
| Round 1 encoder: percentile rule + graph filter (final candidate set) | 98.07% | 4.86 | 0.9943 | 6.13 |
| Round 2 encoder (hard negatives): percentile rule + graph filter (final candidate set) | 98.16% | 4.78 | 0.9946 | 5.95 |

- **Absolute vs percentile cut-off.** The two are about equal on US/India validation data. A fixed similarity threshold, however, depends on the score scale, which differs by country: French similarities sit higher, so the absolute cut-off keeps more French candidates (round 1: 7.67 vs 7.15 per S1). A country whose scores sat lower would lose true matches. The percentile cut-off adapts to each country, so it was kept.
- **Graph filter.** A logistic regression that uses only neighbour-graph features (no string features). It removes about 10% of the candidates for at most a 0.001 drop in the F0.5 ceiling.
- **Hard-negative encoder (round 2).** The encoder is refined on 1,000,000 mined (S1, record, closest wrong S1) triplets. The result: more true matches found at fewer candidates.
- The full grid search (every rule setting tried) is in `results_csv/06_...` and `07_...`.

## 3. Matcher: full-data runs (one change at a time)

| Run | Change | Val F0.5 | Singletons | Non-singletons | US | India | Ceiling |
|---|---|---:|---:|---:|---:|---:|---:|
| R1 | XGBoost on string similarities + per-country IDF overlap + embedding-graph and candidate-context features | **0.9812** | 0.9788 | 0.9814 | 0.9816 | 0.9807 | 0.9943 |
| R2 | + cross-encoder score (trained on encoder-training entities only) | **0.9856** | 0.9884 | 0.9854 | 0.9853 | 0.9860 | 0.9943 |
| R3 | + competition margins (similarity minus best competing pair) | **0.9867** | 0.9894 | 0.9865 | 0.9866 | 0.9869 | 0.9943 |
| R4 | + name-ambiguity counts (S1 records sharing the exact name) | **0.9871** | 0.9911 | 0.9869 | 0.9870 | 0.9873 | 0.9943 |
| R5 | + round-2 (hard-negative) candidate generation | **0.9875** | 0.9850 | 0.9876 | 0.9870 | 0.9881 | 0.9946 |

Final model, share of XGBoost gain: cross-encoder 75%, competition margins 11%, the rest string, graph, context, IDF and ambiguity features (see `results_csv/04_final_feature_importance.csv`).

## 4. Match selection

Every S2/S3 record is first assigned to at most one S1 entity (its highest-scoring claim). Then two selection methods are tuned on validation, and the better one is used:

| Run | Global thresholds (t, t_single, r) | Per-entity expected-F0.5 | Used |
|---|---:|---:|---|
| R1 | 0.9812 | 0.9811 | thresholds |
| R2 | 0.9856 | 0.9854 | thresholds |
| R3 | 0.9867 | 0.9868 | thresholds |
| R4 | 0.9871 | 0.9871 | thresholds |
| R5 | 0.9873 | 0.9875 | expected-f |

## 5. France (country label unseen in training)

- **Diagnosis** (1% sample stress test, where almost every true partner is missing): the first model accepted about 8x more wrong French matches than US/Indian ones. The cause: French addresses share "rue", "de", "la" and multi-word region names.
- **Feature fix:** IDF weights computed per country, and similarity percentiles per country. On the sample, wrong French matches per S1 fell from 0.174 to 0.114.
- **Calibration** (a logit shift applied to French probabilities only):

| Run | Calibration | France logit shift | France matches/S1 (uncalibrated → used) | France no-match share (uncalibrated → used) | US / India no-match share |
|---|---|---:|---:|---:|---:|
| R1 | equalise orphan false-accept rate | 4.03 | 3.41 → 2.98 | 5.04% → 6.91% | 5.59% / 5.57% |
| R2 | equalise orphan false-accept rate | 5.61 | 3.44 → 2.81 | 4.95% → 7.72% | 5.78% / 5.81% |
| R3 | match no-match share | 2.46 | 3.53 → 3.23 | 4.61% → 5.83% | 5.82% / 5.83% |
| R4 | match no-match share | 2.21 | 3.49 → 3.24 | 4.91% → 5.83% | 5.81% / 5.84% |
| R5 | match no-match share | 2.37 | 3.40 → 3.17 | 4.75% → 5.72% | 5.71% / 5.73% |

- **Orphan method (R1–R2): over-corrected.** It equalises France's false-accept rate on simulated distractors with the US/India rate. French businesses predicted as having no match rose to 7–8%, against about 5.7% for the US and India, so genuine French matches were lost. Without any calibration France had slightly too few no-match predictions (about 5.0%), i.e. a few extra false matches.
- **Empty-share method (R3–R5): kept.** It shifts until France's no-match share equals the US/India share on the same test data. Result: France gets 3.17 matches per S1. That fits its 4% smaller pool of S2/S3 records per S1 (the US/India level scaled down predicts about 3.24).
- **Diagnostic check.** The label-free orphan false-accept estimate reproduced the true validation rate for the US and India (R5: estimated 0.07% / 0.17% on test vs true 0.09% / 0.20%); France: 4.82%.

## 6. Smaller experiments (1% sample, for code checks and quick ablations)

These ran on a 1% sample. The local ones used stand-in embeddings, so only the direction of each change matters, not the absolute numbers.

| Experiment | Before | After |
|---|---:|---:|
| Per-country IDF features + similarity percentile (local) | F0.5 0.9701 | 0.9718 |
| Wrong French matches per S1 in the stress test (local) | 0.174 | 0.114 (0.049 with calibration) |
| Graph filter (local) | 14.81 candidates/S1 | 11.54 (ceiling 0.9792 → 0.9785) |
| Expected-F0.5 selection vs thresholds (local) | 0.97181 | 0.97218 |
| Name-ambiguity features (local) | 0.9713 | 0.9720 |
| Competition margins (local; siblings are rare in a 1% sample) | 0.9712 | 0.9713 (full data: +0.0011) |
| Kaggle GPU smoke test: first stage 2 → cross-encoder + filter + calibration | F0.5 0.9889, 5.76 cand/S1 | 0.9903, 4.38 cand/S1 |

## 7. Engineering issues met and fixed

- **FAISS GPU scratch-memory overflow.** A 16 GB request crashed the first full stage-1 run after 2h49m. Fixed with 32k-query search batches, releasing GPU memory between searches, and a CPU fallback.
- **Kaggle re-attaches only a notebook's last *successful* output.** Stage 1 was made resumable (cached artifacts), and the data-folder search now ignores sample copies.
- **GPU out-of-memory in the hard-negative round** (three texts per example). Fixed by halving that step's batch to 256.
- **Text normalisation made 2.2x faster**, with byte-identical output (checked on 318,507 records).

## 8. Not tried (next steps)

- An address-only candidate search, to catch records whose name was replaced (acronyms such as "kf" for "krishna finance", or trade names). These are part of the ~1.8% of true matches blocking misses.
- A larger encoder or cross-encoder (e.g. e5-base), and a second-stage stacking model.

## Where the results are

- **CSV tables:** `results_csv/` (index in `00_index.csv`).
- **Methodology:** `Documentation_template.md`.
- **Kaggle notebooks:** `amazon-ml-er-01-blocking` (v6 round 1, v8 round 2) and `amazon-ml-er-02-match-submit-ce` (v5 final: `matching_results.tsv`, `candidate_pairs.tsv`, `team_submission.zip`).
