# Method & Results Log — Business Entity Resolution

A record of every experiment we run: what method we used, how we measured it, and what score it got. Add one entry per run, and keep the summary table in sync. The final `Documentation_template.md` is written from this file.

---

## 1. Task context

- **Goal:** for every Source 1 (S1) entity, list all Source 2 / Source 3 (S2/S3) records that refer to the same real-world business.
- **Fields:** `entity_id`, `business_name`, `business_address`, `country`. There are no shared IDs across sources.
- **Metric:** F_0.5, averaged per S1 entity (macro). Precision counts twice as much as recall.
  - If an S1 entity truly has no matches, predicting an empty list scores **1.0**, and predicting anything scores **0.0**.
  - If an entity has matches and we predict an empty list, it scores **0.0**.
  - Formula: `F_0.5 = 1.25·P·R / (0.25·P + R)`
- **Deliverables:** `output/matching_results.tsv` (this is scored) and `output/candidate_pairs.tsv` (the exact candidate set the model scores). Every ID in the matches file must also appear in the candidates file.
- **Constraints:**
  - The final model must be MIT or Apache 2.0 licensed, with at most 8B parameters.
  - No external data, APIs, or geocoding.
  - `country` is an open set. The test data includes **France**, which never appears in training.

---

## 2. Dataset facts (measured 2026-09-25)

| Split | Source | Rows | US | India | France | Empty address |
|---|---|---:|---:|---:|---:|---:|
| train | S1 | 2,206,821 | 1,323,633 | 883,188 | — | 0 |
| train | S2 | 5,034,616 | 3,016,817 | 2,017,799 | — | 168,967 |
| train | S3 | 5,285,603 | 3,170,056 | 2,115,547 | — | 175,916 |
| test | S1 | 1,732,544 | 663,106 | 809,986 | 259,452 | 0 |
| test | S2 | 4,887,273 | 1,871,330 | 2,312,565 | 703,378 | 129,408 |
| test | S3 | 5,082,316 | 1,945,701 | 2,405,000 | 731,615 | 136,098 |

No record in any file has an empty `business_name`.

**Training ground truth:**

- 2,206,821 S1 rows, one per S1 entity.
- **123,247 singletons (5.58%)**: S1 entities with no match.
- Matches per S1 entity: mean 3.46, max 11.
  - Distribution: 0→123,247 · 1→119,157 · 2→375,212 · 3→530,841 · 4→484,115 · 5→321,957 · 6→164,868 · 7→63,968 · 8→18,680 · 9→4,205 · 10→534 · 11→37
- Links: 3,693,619 go to S2 and 3,944,746 go to S3, for 7,638,365 in total.
- **Every S2/S3 record is linked to at most one S1 entity.** No ID appears under two S1 rows.
- Unmatched records, which act as distractors: **1,340,997 in S2 (26.6%)** and **1,340,857 in S3 (25.4%)**.

### What this means for method design

1. **One-to-many assignment.** Each S2/S3 record belongs to at most one S1 entity. When several S1 entities claim the same record, we can keep only the highest-scoring one. This is a free precision boost.
2. **Singletons are rare but costly.** They make up only about 5.6% of entities. Each one scores 0 if we predict even one false match, so we should consider a separate "has any match?" gate.
3. **A quarter of S2/S3 records match nothing**, so blocking has to cope with many distractors.
4. **Most entities have 2 to 5 matches.** A top-k cutoff that is too tight will cost recall, while one that is too loose will cost precision. The cutoff should be tuned against F_0.5, not set by hand.
5. **France appears only in test.** Features must not depend on language- or country-specific rules learned only from US and India data. Text normalisation should also handle French (accents, "rue", "avenue", "SARL", "SAS").
6. **About 3% of S2/S3 records have no address.** Blocking and features need a name-only path for these.

---

## 3. Evaluation protocol

_Fix this protocol before the first run, then keep it unchanged so that runs can be compared._

- **Development sample (DEV-10):** 10% of train S1 entities (seed 42), all of their true S2/S3 matches, and 10% of the distractor (unmatched) S2/S3 records, which keeps the full data's matched-to-distractor ratio. Size: 220,682 S1 records, 1,031,969 S2/S3 records and 763,784 true links. Every blocking version below is measured on DEV-10 on the laptop (12 threads).
- **Validation split (full data, `run_pipeline.py`):** train S1 entities are split once with seed 42 into fit 30% / early stopping 5% / **eval 20%** (441,521 entities) / rest 45%. Every entity stays in the candidate search, so targets are contested as densely as at test time. Aliases and region merges are learned without the eval entities' links. Stage 1 and stage 2 are both cross-fitted over the fit entities (3 folds); eval, rest and test rows get the mean of the fold models, so eval rows are scored exactly like test rows. The decision rule and its parameter are chosen on the eval slice.
- **Blocking metrics:**
  - Pair recall: the share of true links found among the candidates.
  - Mean candidates per S1 entity.
  - Reduction ratio.
- **Matching metrics:** macro F_0.5, macro precision, and macro recall on the validation set, reported separately for singletons and non-singletons, and for each country.
- **Threshold:** tuned to maximise validation F_0.5, with the chosen value recorded.
- **Format check:** run `utils/validate_submission.py` on every run that produces a submission.

---

## 4. Results summary

### 4.1 Blocking versions (candidate generation), side by side

All rows are measured on DEV-10. **Pair recall** is the share of true links that appear among the candidates, and is the upper bound on final recall. **All found** is the share of S1 entities whose true matches are all among the candidates. Top-K is per target source (S2 and S3 separately), plus a name-only pass over targets with no address from BLK-v2 onwards.

| Version | Date | What changed | Pair recall | All found | Cand. / S1 | Time | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| BLK-v1 | 2026-09-25 | IDF-weighted sparse token search. Features: name tokens, address tokens, 4-char prefixes, consonant skeletons, first/last 3 digits of house numbers. df cap 1%, top-30. | 97.66% | 92.76% | 60.0 | 26 s | Baseline |
| BLK-v2 | 2026-09-25 | + joined-name token, name character 3-grams, `shree/shri/sri` merged, name-only pass (top-5) for targets with no address | 99.31% | 97.71% | 69.1 | 135 s | Recall up, but cost grows about 100× on full data |
| BLK-v3 | 2026-09-25 | v2 with a fixed df cap (equal to 3,000 / 6,000 on full data) in place of the 1% cap | 97.74% / 98.49% | 93.24% / 95.30% | 69 | 27 / 30 s | **Rejected**: too much recall lost |
| BLK-v4 | 2026-09-25 | v2 + search split by region (tail tokens of S1 addresses, region merges learned from training links) | 99.48% | 98.20% | 66.9 | 48 s | Noisy region keys (`rd`, `nagar`) |
| **BLK-v4b** | 2026-09-25 | v4 + region keys limited to tokens that sit mostly at the end of an address: 16 Indian states, 37 US states; one learned merge (Telangana ↔ Andhra Pradesh) | **99.52%** | **98.35%** | 65.9 | **43 s** | **Current** |
| BLK-v4b @10 | 2026-09-25 | same, top-10 per source | 99.10% | — | 25.9 | — | Cheaper option for the matcher |
| **BLK-v4b@20, FULL train** | 2026-09-25 | BLK-v4b top-20 on the **full** data (2.2M S1 x 10.3M targets); eval slice shown | **98.32%** | 94.53% | 47.5 | 703 s (EC2, 8 cores) | Denser competition than DEV-10: -1.1 points. Targets without address: 88.7%. Recall still rising at k=20 |

### 4.2 End-to-end versions (validation F_0.5)

Unless noted, the scores are on the DEV-10 **evaluation slice**: 20% of the DEV-10 S1 entities (seed 42), never used for fitting or early stopping. The model is fitted on 70% and early-stopped on 10%. The competition features use all DEV-10 S1 records, as they would at test time.

| ID | Date | Method (short) | Blocking recall | Candidates / S1 | Val F_0.5 | Val P | Val R | Singletons / non-singletons | LB F_0.5 | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| M-v1 | 2026-09-25 | NORM-v2 + BLK-v4b (top-20 per source) + 44 features + LightGBM (log loss) + exclusive assignment + threshold 0.725 | 99.45% | 46.9 | 0.9882 | 0.9954 | 0.9745 | 0.9805 / 0.9887 | — | Upper bound with a perfect matcher on these candidates: 0.9984. Laptop, DEV-10 |
| M-v2 | 2026-09-25 | M-v1 + 8 soft word-alignment features (52 in total); exclusive, threshold 0.725 | 99.45% | 46.9 | 0.9894 | 0.9959 | 0.9772 | 0.9826 / 0.9899 | — | Scored on EC2 from the saved M-v2 predictions |
| M-v3 | 2026-09-25 | 2 stages: stage 1 cross-fitted (3 folds) + stage 2 on stage-1 probability context; exclusive, expected-F | 99.45% | 46.9 | **0.9906** | 0.9963 | 0.9794 | 0.9814 / 0.9911 | — | Threshold 0.725 instead: 0.9904 (singletons 0.9935). EC2, DEV-10 |

---

## 5. Experiment log

Copy the template below for each run, newest entry first. Record every run, including failed ones, and say why it failed.

<!--
### EXP-XXX — <short title>

- **Date:**
- **Git commit:**
- **Hypothesis:** what change is expected to help, and why

**Methodology**
- Preprocessing / normalisation:
- Blocking / candidate generation: (keys, index, top-k, filters)
- Features:
- Model: (type, params, license, size)
- Post-processing: (threshold, one-to-one assignment, singleton gate)

**Results**

| Metric | Overall | US | India | Singletons | Non-singletons |
|---|---:|---:|---:|---:|---:|
| Blocking recall | | | | — | |
| Candidates / S1 | | | | | |
| F_0.5 | | | | | |
| Precision | | | | | |
| Recall | | | | | |

- Chosen threshold:
- Leaderboard F_0.5 (if submitted):
- Runtime / memory:

**Analysis**
- Typical false positives:
- Typical false negatives:
- Conclusion: keep / drop / iterate
- Next step:
-->

### FULL-v1 blocking — BLK-v4b@20 on the full data

- **Date:** 2026-09-25
- **Setup:** `run_pipeline.py`, preset M-v3 (NORM-v2 + BLK-v4b@20), EC2 r7a.2xlarge (8 cores, 64 GB). Aliases and region merges learned from non-eval links: 1,105 name and 3,107 address aliases; region merges telangana↔andhrapradesh and **delhi↔haryana** (the second only shows up on full data).

**Results**

| Split | Candidates | Per S1 | Time |
|---|---:|---:|---:|
| Train (2.2M S1) | 104,748,300 | 47.5 | 703 s (India 222 s, US 481 s) |
| Test (1.73M S1) | 83,319,185 | 48.1 | 488 s (France 60 s, India 238 s, US 190 s) |

Peak memory 44.4 GB. Eval slice (441,521 S1, 1,528,407 links): **pair recall 98.32%**, all links found for 94.53% of S1.

| Eval links | Links | Recall | Missed |
|---|---:|---:|---:|
| Target without address | 66,939 | 88.73% | 7,544 |
| Target with address | 1,461,468 | 98.76% | 18,132 |
| India S2 / S3 | 295,785 / 315,346 | 98.34% / 96.58% | |
| US S2 / S3 | 443,077 / 474,199 | 98.85% / 98.97% | |

Recall by depth of the main pass (the name-only pass always counted): top-5 96.00%, top-10 97.42%, top-15 97.98%, top-20 98.32%.

**Analysis**
- DEV-10 overstated blocking recall by 1.1 points (99.45%): each region holds about 10 times more rival records on full data, so true matches fall out of the top 20 more often.
- The name-only pass for targets without an address keeps only 5 per S1; with many more same-name records it finds just 88.7% of those links, 29% of all misses.
- Recall is still rising at k=20, so a deeper search should pay. **Next (BLK-v5):** measure recall curves with top-40 main pass and top-20 name-only pass, then pick the cutoffs from the curves.

### Error analysis — what M-v3 still gets wrong

- **Date:** 2026-09-25
- **Setup:** M-v3 stage 2, exclusive assignment, threshold 0.725, DEV-10 evaluation slice (`experiments/dev_errors.py`).

**Results:** 152,187 true links, 149,431 predicted, 3,187 missed, 431 false.

| Missed links | Count | Share | Targets without address |
|---|---:|---:|---:|
| Never a candidate (blocking miss) | 839 | 26% | — |
| Lost to another S1 entity under exclusive assignment | 925 | 29% | 95% |
| Scored below the threshold | 1,423 | 45% | 46% |

**Analysis**
- Targets without an address are about 3% of all targets but about half of the recoverable misses. Their only evidence is the name, and many S1 entities share a name, so the wrong one often wins the target.
- 82% of false positives do have an address. The typical one is the same name at a **nearby house number** (831 vs 835 Winsor Pl, 3-4-114/12 vs /13, G/70/15 vs /16): the data plants such distractors. True matches also carry number noise (189 vs 889, 20 vs 19, 730 vs 30), so exact number overlap alone cannot separate them.
- Some true matches carry an invented trade name (Asahi Charitable Trust ↔ Korevo, East Saint Louis Baptist Center ↔ Ectozetalum). Only the address, or the entity's other records, can link these.
- **Next:** number-gap features (digit edit distance and numeric gap of the closest unmatched numbers) and support features (how much a candidate resembles the S1's confident candidates), tested in `experiments/dev_v3.py`.

### M-v3 — two-stage stacking and the decision rule

- **Date:** 2026-09-25
- **Git commit:** not yet committed
- **Hypothesis:** In M-v1 the margin of the *blocking score* over the best other S1 carried 79% of the gain. The same competition measured in stage-1 *probabilities* should resolve contested targets better. The set of links should also be chosen per entity rather than by one global threshold.

**Methodology**
- Stage 1: M-v2 features, cross-fitted over 3 folds of the fit entities. Fit rows get the prediction of the fold model that did not see their entity; all other rows get the mean of the 3 models.
- Stage 2: stage-1 features plus `p1` and 11 context features from `probability_context`: rank of `p1` within its S1 (overall and per source) and for its target; best other `p1` for the target, the S1 and the S1's source; margins over those; sum of `p1` over the S1; count above 0.5; in-degree of the target.
- Decision rules, each after exclusive assignment: a single threshold; **expected F** (per S1, predict the top k that maximise expected F0.5 under independent probabilities, or nothing when "no match" is the better bet); **gated expected F** (nothing unless the best candidate reaches a gate).

**Results (DEV-10 evaluation slice, 44,137 S1)**

| Model / rule | F0.5 | P | R | Singletons | Others |
|---|---:|---:|---:|---:|---:|
| Stage 1 cross-fitted, threshold 0.725 | 0.9894 | 0.9959 | 0.9770 | 0.9822 | 0.9898 |
| Stage 1, expected F | 0.9887 | 0.9948 | 0.9770 | 0.9603 | 0.9904 |
| Stage 2, threshold 0.725 | 0.9904 | 0.9970 | 0.9789 | **0.9935** | 0.9902 |
| **Stage 2, expected F** | **0.9906** | 0.9963 | 0.9794 | 0.9814 | **0.9911** |
| Stage 2, gated expected F (gate 0.55) | 0.9906 | 0.9966 | 0.9792 | 0.9858 | 0.9909 |

- **Top stage-2 features by gain:** `p1` 69%, `p1_margin_t` (margin over the best other S1 for the target) 26%, `margin_vs_other_s1` 3%.

**Analysis**
- Stage 2 adds +0.0010 to +0.0012, mostly on singletons: at the same threshold they go from 0.9822 to 0.9935, because `p1_margin_t` removes targets that a stronger S1 claims.
- A threshold protects singletons; expected F picks better set sizes for entities with matches. Gating did not combine the two on DEV-10: every gate from 0.30 to 0.55 gives 0.9906. The full pipeline picks the rule on its own evaluation slice.
- Caveat: `dev_stack.py` trains stage 2 once on the fit rows, so on DEV-10 those rows compete with in-sample probabilities during exclusive assignment. `run_pipeline.py` cross-fits stage 2 as well.
- **Conclusion:** keep both stages; choose the rule on full data.

### M-v2 — soft word alignment of names

- **Date:** 2026-09-25
- **Hypothesis:** Token-level alignment should tell noise (typos, added generic words) apart from a different business (a rare word replaced by another).

**Methodology**
- `align_features`: each core-name token is aligned to its best Jaro-Winkler match on the other side, in both directions. Per side: mean similarity (Monge-Elkan), worst token, number of unaligned tokens (similarity below 0.85) and the highest S1-side IDF among them. Tokens never seen in S1 get the country's maximum IDF. 8 features, 52 in total.

**Results (DEV-10 evaluation slice):** exclusive, threshold 0.725: **F0.5 0.9894** (P 0.9959, R 0.9772, singletons 0.9826, others 0.9899). Non-exclusive best: 0.9893 at 0.825. Expected F: 0.9890 (singletons 0.9635).

- **Conclusion:** keep (+0.0012 over M-v1).

### M-v1 — first end-to-end matcher (LightGBM + exclusive assignment)

- **Date:** 2026-09-25
- **Git commit:** not yet committed
- **Hypothesis:** A gradient-boosted classifier on string-similarity and competition features, plus the constraint that each S2/S3 record belongs to at most one S1 entity, should reach high precision.

**Methodology**
- Preprocessing and blocking: NORM-v2, learned aliases, BLK-v4b with top-20 per source plus the no-address pass. That gives 10.36M candidate pairs for 220,682 S1 records, of which 7.33% are positive.
- **Features (44):**
  - Blocking: score, name cosine and address cosine (IDF-weighted, from the blocking matrices).
  - Names (rapidfuzz), on the full name, the core name (legal words removed) and the joined-up name: ratio, token-set, token-sort, partial ratio, Jaro-Winkler.
  - Aliases: best match against the `fka` / `dba` parts.
  - Token containment in both directions, for names and addresses.
  - Addresses: ratio, token-set, token-sort, partial ratio.
  - House numbers: Jaccard, share of B's numbers in A, whether A's first number is in B, truncated-number match.
  - Lengths, and flags: address missing, website name, alias present, non-Latin script.
  - Context: rank of the candidate within its S1 record (per source and overall), gap to the best, number of candidates.
  - Competition: how many S1 records retrieved this target, this S1 record's rank among them, and the score margin over the best *other* S1 record.
  - Ambiguity: how many S1 records share this name.
- **Model:** LightGBM, `objective=binary` (**log-loss / binary cross-entropy**). Settings: 255 leaves, learning rate 0.05, min 200 rows per leaf, 0.8 feature and bagging fractions, L2 = 1. Early stopping after 100 rounds on the 10% slice; best iteration 385.
- **Why log loss:** F_0.5 is decided per S1 entity over a *set* of links, so the decision step needs calibrated probabilities. A ranking loss (LambdaRank) orders candidates well, but its scores are not probabilities and it has no natural cut-off for "no match".
- **Post-processing:** exclusive assignment (each target kept only for the S1 entity that scores it highest), then a single probability threshold tuned for macro F_0.5 (grid 0.20–0.95).

**Results (DEV-10 evaluation slice, 44,137 S1 entities)**

| Metric | Overall | Singletons | Non-singletons |
|---|---:|---:|---:|
| F_0.5 (non-exclusive, threshold 0.75) | 0.9880 | 0.9814 | 0.9884 |
| **F_0.5 (exclusive, threshold 0.725)** | **0.9882** | 0.9805 | 0.9887 |
| Precision / Recall (exclusive) | 0.9954 / 0.9745 | | |
| Upper bound (perfect matcher on these candidates) | 0.9984 | | |

- **Top features by gain:** margin over the best other S1 record 79.2%, rank among the target's S1 records 10.2%, blocking score 3.1%, then address containment, number Jaccard and address token-set.
- **Runtime (laptop, 12 threads):** blocking 67 s, features 121 s, training 175 s.

**Analysis**
- Precision is already 0.995. Most of the remaining loss is recall (0.9745 against a 0.9945 blocking ceiling).
- DEV-10 has 10× fewer competing records per region than the full data, so this score is optimistic. The model must be trained and validated on the full data.
- **Next:**
  - A second stage that uses first-stage probabilities as competition features (probability margin over the other S1 records, and over the other candidates of the same S1 record).
  - A decision step that maximises expected F_0.5 per entity.
  - A full-data run on SageMaker.

### BLK-v4b — region-split search with positional region keys (current)

- **Date:** 2026-09-25
- **Hypothesis:** BLK-v4 picked up common words that happen to end some addresses (`rd`, `nagar`) as region keys. Their groups swallowed most records and produced bogus merges. States sit at the end of an address almost every time they appear, so a positional test should keep only real regions.

**Methodology**
- Region keys: the last token of at least 0.2% of S1 addresses in that country, **and** the last token in at least 60% of the addresses that contain it.
- Merges: two keys are merged when at least 10 true training links cross them.
- A country is split only if it has training links. An unseen country (France) gets a full search, because its keys can't be validated.
- Everything else is as in BLK-v4.

**Results (DEV-10):** pair recall **99.52%**, all found 98.35%, 65.9 candidates per S1, 43 s. The keys are now clean (16 Indian states, 37 US states), with a single merge (`telangana` ↔ `andhrapradesh`, because some Telangana records are still filed under the pre-2014 state). With top-10 per source: 99.10% recall at 25.9 candidates per S1.

- **Conclusion:** keep. It is the default blocking for the first matcher.

### BLK-v4 — search split by region

- **Date:** 2026-09-25
- **Hypothesis:** Sparse search cost grows with (S1 size × target size), so BLK-v2 would take about an hour per run on the full data. A true match almost always shares its state with the S1 record. Comparing each S1 record only with targets from its own region should therefore cut the cost roughly by the number of regions, with little recall loss.
- **Measured before building:** splitting by region would lose only 0.12% of true links (both records have a region, but different ones). These were mostly Telangana ↔ Andhra Pradesh and `dl` ↔ `delhi`.

**Methodology**
- Keys: the frequent last tokens of S1 addresses.
- Region pairs that true links cross are merged with union-find.
- An S1 record is compared with targets in its own region group(s) plus all targets that have no region token. An S1 record with no region token is compared with everything.
- IDF stays per (country, source), so scores are comparable across groups.

**Results:** pair recall 99.48%, all found 98.20%, 66.9 candidates per S1, 48 s (v2: 135 s). Recall went **up** versus v2, because lookalikes from other regions no longer take places in the top 30.

**Analysis:** the learned merges included `rd`, `nagar` and `hyderabad`, because those words end some Indian addresses. Fixed in v4b.

### BLK-v3 — fixed df cap (rejected)

- **Date:** 2026-09-25
- **Hypothesis:** Character 3-grams account for 60% of the search work (US/S2 profile). A fixed df cap would bound the work per feature, so full-data cost would grow about 10× instead of 100×.
- **Results:** a cap equal to 3,000 on full data gave 97.74% recall; a cap equal to 6,000 gave 98.49%. Without 3-grams the recall was 97.36% and 98.12%.
- **Why rejected:** the recall loss (0.8–1.6 points) is too large. A fixed cap drops the medium-frequency words (city names, common name words) that rank the true match, so region splitting (v4) was used instead.

### BLK-v2 — joined names, 3-grams, name-only pass for missing addresses

- **Date:** 2026-09-25
- **Hypothesis:** The v1 misses were mainly:
  - website or joined names with no word in common with the S1 name (`eyeassociates`, `unitedit`)
  - heavy typos (`csmpaneis`)
  - targets with no address, which lose on score to same-name records that have one
  - `shree` vs `sri`

**Methodology**
- Add the whole joined-up core name as one feature (it matches about 5 records on average, so it costs almost nothing).
- Add character 3-grams of the joined-up name.
- Merge the honorific variants.
- Add a second, name-only search (top-5) restricted to targets with no address.

**Results:** pair recall 99.31% (+1.65 points), all found 97.71%, 69.1 candidates per S1. The no-address pass alone recovered 3,046 links. Time 135 s: the 3-grams dominate, and cost would grow about 100× on the full data.

### BLK-v1 — IDF-weighted sparse token search

- **Date:** 2026-09-25

**Methodology**
- **Normalisation (NORM-v2):**
  - transliterate to ASCII with anyascii
  - canonicalise legal forms, street types, US states and ordinals
  - join multi-word Indian state names into one token
  - add French street types
  - fix digit-for-letter typos (`5ummit` → `summit`)
  - split names at `fka` / `dba`
  - turn website names back into words
  - strip `(ID: …)` tags
- **Learned aliases:** per country, from training links. A target token that is unaligned in at least 50% of its pairs, is always accompanied by the same unaligned S1 token, and is rare on the S1 side becomes an alias of that S1 token. Ties are broken by spelling similarity. This yields 452 name and 135 address aliases, e.g. `praivet` → `pvt`, `eksports` → `exports`, `mh` / `mharastr` → `maharashtra`, `tn` → `tamilnadu`, `lnc` → `inc`.
- **Features:** name tokens, address tokens, 4-char prefixes, consonant skeletons, and the first/last 3 digits of house numbers (for `7530` ~ `530`). All features are hashed into 2^24 slots.
- **Scoring:** IDF weights on the target side; name and address blocks L2-normalised separately; score = cos(name) + cos(address). Top-30 per (country, source), via sparse_dot_topn (Apache-2.0).

**Results:** pair recall 97.66%, all found 92.76%, 60 candidates per S1, 26 s. Recall at top-5: 95.2%; top-10: 96.5%; top-20: 97.4%.

### Data facts that shaped the design (EDA, 2026-09-25)

- Country never differs between an S1 record and its matches (0 of 7.64M links), so blocking runs within each country, with no country hard-coded.
- 39.6% of S1 records share their exact normalised name with another S1 record ("Primary Care Group" appears 253 times), so the address and competition features must break ties.
- Only 5–6% of distractors share an exact name with an S1 record. The hard negatives are same-name businesses at a different address.
- At most 5 matches per S1 come from S2, and at most 6 from S3.
- Indian S2/S3 names are in Devanagari 7–13% of the time, and addresses about 13% of the time. S1 is always clean Latin text with a full address.
