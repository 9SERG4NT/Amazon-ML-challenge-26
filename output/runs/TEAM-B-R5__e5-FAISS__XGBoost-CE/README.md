# TEAM-B-R5: the team's bi-encoder + cross-encoder + XGBoost run

The teammates' second pipeline, final run R5 (design "B" in their write-up), from the Kaggle notebook
`amazon-ml-er-02-match-submit-ce` (v5). Downloaded 2026-09-26 at 21:05 IST.

| File | What it is | md5 |
|---|---|---|
| `matching_results.tsv` | **The file to upload to the leaderboard.** The predictions converted to the required tab-separated format (`source1_entity_id` TAB comma-joined `matched_entity_ids`); nothing else changed | `67bdd7f85c52d143128e40dff42418eb` |
| `matching_results_csv.zip` | The predictions as downloaded (comma-separated, quoted lists) | `07dbde492d8e89e277c6585168cc4924` |
| `all_approaches.md` | Their write-up: every approach and result | `9c1d836e8d227b17890bb356f81b7a5c` |

Official validator: **PASS** (1,732,544 S1 rows, 99,146 empty). Public leaderboard: **0.98** (uploaded 2026-09-26
evening; M-v11 scored 0.970), the best score so far.

Not here, and needed if this run becomes the final package: its `candidate_pairs.tsv` (5.95 candidates per S1, in
the notebook's `team_submission.zip`) and its code. Our `candidate_pairs.tsv` cannot be used with these matches:
106k of them are not among our candidates.

## Method (from `all_approaches.md`)

- **Blocking:** `intfloat/multilingual-e5-small` bi-encoder (MIT, 118M parameters), fine-tuned with an in-batch
  contrastive loss on 2M (S1, matching record) pairs, half the batches from one city (hard negatives), then refined on
  1M mined (S1, record, closest wrong S1) triplets. FAISS IVF search per country: each S2/S3 record gets its 3
  closest S1 records, each S1 its 10 closest records. A per-country percentile cut-off and a logistic-regression
  graph filter keep the candidates. Validation recall 98.16% at 4.78 candidates per S1 (test 5.95); a perfect matcher
  on them would score 0.9946.
- **Matcher:** XGBoost on string similarities, per-country IDF overlap, embedding-graph and candidate-context features,
  a **cross-encoder score** (75% of the XGBoost gain), competition margins (11%) and name-ambiguity counts.
- **Selection:** each record goes to its highest-scoring S1, then per-entity expected F0.5 (chosen on validation).
- **France:** per-country IDF and similarity percentiles, and a logit shift of 2.37 on French probabilities, so
  France's no-match share equals the US/India share (5.72%).
- **Validation:** 220,140 train S1 entities (10%) held out from every model. F0.5 **0.9875** (singletons 0.9850,
  others 0.9876; US 0.9870, India 0.9881). Run by run: R1 0.9812, + cross-encoder 0.9856, + margins 0.9867,
  + name ambiguity 0.9871, + hard-negative encoder 0.9875.

## Against M-v11 (our submission, leaderboard 0.970), on the test set

| | France | India | US |
|---|---:|---:|---:|
| S1 linked, team / M-v11 | 94.3% / 93.9% | 94.3% / 94.0% | 94.3% / 94.1% |
| Links per S1, team / M-v11 | 3.17 / 3.13 | 3.36 / 3.28 | 3.37 / 3.31 |
| Estimated true links per S1 (label-free, ±0.1) | 3.31 | 3.47 | 3.41 |
| S1 with identical link sets | **64.3%** | 83.6% | 85.6% |
| Links in both | 759,549 | 2,610,087 | 2,162,073 |
| Links only in team / only in M-v11 | 62,223 / 53,457 | 110,713 / 46,531 | 73,752 / 32,445 |
| S1 empty in one file but not the other | 1.65% | 0.66% | 0.47% |

The team's run links more, closer to the estimated truth, so its recall is probably higher. The two agree least on
France. The validation scores are not comparable: 0.9875 is on their own held-out entities, apparently with every
train entity still in the search (the setting where our FULL-v1 scored 0.9813), while M-v11's 0.9854 is on our
test-like universe. Only the leaderboard ranks the two.
