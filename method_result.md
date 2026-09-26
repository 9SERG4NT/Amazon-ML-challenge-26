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
| BLK-v4b@40n20, FULL train | 2026-09-25 | top-40 main pass + top-20 name-only pass on the full data; eval slice shown | **98.93%** | 96.43% | 114.4 | 834 s | Main-pass depth 5 / 10 / 20 / 30 / 40 (name-only 20): 95.99 / 97.40 / 98.28 / 98.70 / 98.93% at 44 / 54 / 74 / 94 / 114 candidates per S1. Name-only depth 5 / 10 / 15 / 20 (main 40): 98.82 / 98.88 / 98.91 / 98.93% at 90 / 100 / 108 / 114. Name-only beyond 5 is not worth it; top-30 or top-40 with name-only 5 is the frontier (≈ +0.3 / +0.5 points at +47% / +89% candidates). Test: 200.5M candidates. Peak 53.2 GB |

### 4.2 End-to-end versions (validation F_0.5)

Unless noted, the scores are on the DEV-10 **evaluation slice**: 20% of the DEV-10 S1 entities (seed 42), never used for fitting or early stopping. The model is fitted on 70% and early-stopped on 10%. The competition features use all DEV-10 S1 records, as they would at test time.

| ID | Date | Method (short) | Blocking recall | Candidates / S1 | Val F_0.5 | Val P | Val R | Singletons / non-singletons | LB F_0.5 | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| M-v1 | 2026-09-25 | NORM-v2 + BLK-v4b (top-20 per source) + 44 features + LightGBM (log loss) + exclusive assignment + threshold 0.725 | 99.45% | 46.9 | 0.9882 | 0.9954 | 0.9745 | 0.9805 / 0.9887 | — | Upper bound with a perfect matcher on these candidates: 0.9984. Laptop, DEV-10 |
| M-v2 | 2026-09-25 | M-v1 + 8 soft word-alignment features (52 in total); exclusive, threshold 0.725 | 99.45% | 46.9 | 0.9894 | 0.9959 | 0.9772 | 0.9826 / 0.9899 | — | Scored on EC2 from the saved M-v2 predictions |
| M-v3 | 2026-09-25 | 2 stages: stage 1 cross-fitted (3 folds) + stage 2 on stage-1 probability context; exclusive, expected-F | 99.45% | 46.9 | **0.9906** | 0.9963 | 0.9794 | 0.9814 / 0.9911 | — | Threshold 0.725 instead: 0.9904 (singletons 0.9935). EC2, DEV-10 |
| **FULL-v1** (M-v3) | 2026-09-25 | M-v3 on the **full** data: fit 30% of train S1, scored on the full-density eval slice (441,521 S1); exclusive, gated expected-F (gate 0.5) | 98.32% | 47.5 | **0.9813** | 0.9952 | 0.9544 | 0.9774 / 0.9815 | **0.96** (public) | Perfect matcher on these candidates: 0.9947. India 0.9778, US 0.9837. Top of the public leaderboard: 0.99. The 0.02 gap to eval is a train/test shift (see "Leaderboard gap" below) |
| **M-v5** | 2026-09-26 | Test-like universe (BLK-v4b@20-tlu40) + FEAT-v3 (number gap) + MATCH-v6 (75% fitted, lr 0.1); gated expected-F (gate 0.55). **Test-like eval slice**, not comparable with the full-universe rows | 98.69% | 47.8 | **0.9852** | 0.9957 | 0.9644 | 0.9808 / 0.9854 | **0.971** (public, rank ~400) | Perfect matcher on these candidates: 0.9959. India 0.9818, US 0.9875. `num_x_edit` carries 10.9% of the stage-1 gain |
| **M-v6** | 2026-09-26 | M-v5 with FEAT-v4 (distinctive-part features); gated expected-F (gate 0.5). Test-like eval slice | 98.69% | 47.8 | **0.9856** | 0.9960 | 0.9647 | 0.9812 / 0.9859 | pending | India 0.9822, US 0.9878. Changes 6.9% of French S1's link sets (US/India 3.4–3.6%). Kaggle TPU run |
| M-v7 | 2026-09-26 | Full universe with every distractor copied (BLK-v4b@20-dup2) + FEAT-v4 + MATCH-v6 | 98.17% | 47.5 | 0.9856 | 0.9978 | 0.9599 | 0.9924 / 0.9852 | not submitted | **Rejected:** the model learned to spot the copies and over-links the test (98.5% of S1, 4.20 links each) |

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

### BLK-v5 — a learned candidate filter: from 48 to a few candidates per S1 (M-v11, M-v12)

- **Date:** 2026-09-26. Commit `0d4079d`. Why: the organisers announced on 2026-09-26 that `candidate_pairs.tsv` counts in
  the final ranking, and a smaller candidate set per S1 ranks higher. It must be the exact set the matcher scores (the
  last of several blocking/filtering stages is allowed). Our search kept 47.8 per S1 in the test-like universe (48.1 on
  the test), almost all of them low-ranked rivals: ~3.4 true links per S1.
- **Uniform top-k is a poor lever.** From the M-v6 search (eval slice): top-5 / 10 / 20 per source find 96.49 / 97.90 /
  98.69% of true links at 17.8 / 27.8 / 47.8 candidates per S1 (the name-only pass alone adds 7.8). Halving the list
  costs 0.8 points of recall.
- **Method: a second blocking stage.** A small LightGBM (63 leaves, lr 0.1, 3 folds) sees only what the search produced:
  the source, the blocking score and its name and address cosines, the name-only flag, and the competition context of
  those scores (rank within the S1's list per source and overall, gap to the S1's best, list length, how many S1
  records retrieved the target and this S1's rank among them, margin over the best other S1, S1 name frequencies). No
  string similarity: it is cheap and runs on the search output only. It is trained like the matchers (fit and rest
  entities of the universe, out-of-fold on fit rows, fold mean elsewhere). The threshold keeps **99.5% of the true links
  the search found for fit entities** (out-of-fold, so the eval slice is never used to choose it). Features, training and
  the decision rule then run only on the kept candidates, so the file is exactly what the matcher scores. The search
  output of BLK-v4b@20-tlu40 is reused when cached.
- **1% smoke run (Kaggle CPU):** 45.6 → **3.7 candidates per S1** on train; eval pair recall 0.9995 → 0.9933; eval F0.5
  0.9952 (unfiltered smoke runs: 0.9954–0.9956); a perfect matcher on the kept candidates would score 0.9983. The curve
  (share of found fit links kept: eval pair recall, candidates per S1): 0.98: 0.9787, 3.4; 0.99: 0.9889, 3.5; 0.995:
  0.9933, 3.7; 0.997: 0.9960, 4.0; 0.999: 0.9981, 5.6. The sample has ~10× less competition than the full data, so the
  full run decides.
- **Full run:** running on the EC2 runner (chain5 from 16:00 IST): M-v11 = BLK-v5-tlu40 + FEAT-v4 + MATCH-v6, then
  M-v12 = the same with MATCH-v10, then M-v10 and M-v6 on the unfiltered candidates for the comparison.

### M-v6 and M-v7 on full data — copied distractors teach the model to spot copies (M-v10 instead)

- **Date:** 2026-09-26. M-v6 ran on a Kaggle TPU VM (queued 3.5 h, run 1 h 50 min); M-v7 on the new EC2 runner
  (09:45–13:18 IST: prep 6 min, block 20 min, features 50 min, train 2 h 17 min, of which stage-1 scoring of 188M pairs
  took 54 min; peak 58.8 GB).

| Run | Eval slice | F0.5 | P | R | Singletons | India | US | Test: S1 linked | Test links / S1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M-v5 (LB 0.971) | test-like (tlu40) | 0.9852 | 0.9957 | 0.9644 | 0.9808 | 0.9818 | 0.9875 | 94.1% | 3.27 |
| **M-v6** (+ FEAT-v4) | test-like (tlu40) | **0.9856** | 0.9960 | 0.9647 | 0.9812 | 0.9822 | 0.9878 | 94.1% | 3.27 |
| M-v7 (dup2 universe) | doubled distractors (copies) | 0.9856 | 0.9978 | 0.9599 | 0.9924 | 0.9819 | 0.9880 | **98.5%** | **4.20** |

- **M-v6:** +0.0004 over M-v5 on the same slice; the test link count is unchanged (5,672,242 vs 5,672,319), but it changes
  the link set of 6.9% of French S1 against 3.4–3.6% elsewhere, which is where FEAT-v4 aims. The validator passes.
  Candidate for the next leaderboard submission.
- **M-v7 fails on the test.** Its eval looks sound (precision 0.9978), but it links 98.5% of test S1 with 4.20 links each
  (7.27M links against M-v5's 5.67M; truth ~94% and ~3.4). Every copied distractor sits next to an exact twin in its S1's
  list: the same record, the same scores, a doubled name count. The model learned "an exact twin means a distractor", which
  holds on the dup2 eval slice (it has copies too) and never on the test, where it then reads real lookalikes as matches.
  The smoke test had warned of it (34% / 16% / 10% of test S1 linked in the 1% sample, 2.5× the tlu40 model), and so had
  the `p1_margin_q` tie. **Lesson: a train-time augmentation must not leave a fingerprint the test lacks; check the test
  profile (share linked, links per S1) before trusting an eval.** M-v8 (XGBoost) and M-v9 (blend) used the same features
  and were stopped.
- **M-v10 (`MATCH-v10`, preset M-v10): the same density without copies.** On M-v6's universe and features, distractor rows
  get training weight 2 (the loss sees twice the lookalikes per S1), and the rule is chosen on a **doubled-distractor
  eval**: every link to a distractor counts twice among an S1's predictions, which is what twice the lookalikes does to
  precision when the model takes one of them. Every run now reports this second eval next to the plain one. It cannot
  count a second lookalike taken where the first was not, so it errs on the lenient side.

### Test distractors are lookalikes of present S1 records — M-v7 doubles every distractor

- **Date:** 2026-09-26 (08:45–09:10 IST)
- **Question:** M-v5 scores 0.9852 on the test-like eval slice but 0.971 on the leaderboard. Is the rest of the gap France (M-v5's
  analysis implied France ≈ 0.89), or is the test-like universe still easier than the test?
- **Script:** `experiments/distractor_twins.py <dataset_dir>` (raw TSVs, a 3% hash sample of targets, runs on the laptop).
  A *twin* of a target is an S1 record of the same country with the same core-name key whose address shares ≥ 30% of its
  words (≥ 50%, ≥ 70% as checks): a same-name, same-street lookalike.

**1. Train distractors are lookalikes of specific S1 records.** Examples: S1 "Global Foundation VI, 14655 Summit View Lane,
Loudoun County, VA" has the distractor "Global Foundation VI Corp, 14664 Summit View Ln, Loudoun County, Virginia"; S1 "Turner,
German and Feist, 12010 Jantzen Drive, Portland" has "Co Turner, German and Feist, 12031 JANTZEN DRIVE". Distractors carry the
same formatting noise as true matches (upper case 24% vs 26%, double spaces 9% vs 11%, brackets 9.5% vs 9.8%), so
formatting does not separate them.

**2. Twin rates (share of targets with a twin, overlap ≥ 0.3):**

| Targets | India | US |
|---|---:|---:|
| Train matched records | 0.367 | 0.467 |
| Train distractors, all S1 present | 0.097 | 0.200 |
| Train distractors, 40% of S1 kept (test-like universe) | 0.041 | 0.081 |
| Test, all targets | 0.256 | 0.358 |
| **Test distractors** (test rate minus 60% matched at the train rate, over the 40% distractor share) | **0.090** | **0.201** |

Test distractors are twins as often as train distractors with every S1 present, not as rarely as in the test-like universe.
The generator seems to write each distractor from a present S1, so the test, with twice the distractors per S1, has twice the
hard lookalikes per S1. The test-like universe doubled the count with orphans: dropping an S1 leaves its lookalikes looking
like no present record, which are easy negatives. Its eval slice therefore faces about half the test's hard lookalikes and
overstates the test score in every country. The "France ≈ 0.89" estimate assumed it did not, and is withdrawn.
France behaves like the other countries on unlabelled checks: a house-number conflict in 0.9% of M-v5's French links (India
0.9%, US 2.7%), and the models FULL-v1 and M-v5 disagree on 8.2% of French S1 (US 8.5%, India 5.4%).

**3. Fix: `BLK-v4b@20-dup2` (preset M-v7 = NORM-v2 + BLK-v4b@20-dup2 + FEAT-v4 + MATCH-v6).** The full train universe with
every distractor twice: 2.3 (US) and 2.6 (India) distractors per S1 (test 2.3), with train's mix of easy and hard ones, so
twice the lookalikes per S1. Blocking searches the originals; each copy then gets its original's candidate rows and every
list is cut back to its top k, which gives the lists a search over the doubled targets would (a copy ties with its
original and takes the next slot). Features count the copies (name frequencies, frequent tokens). The eval slice is the same
441,521 S1 entities, now facing test-like lookalike density; the US keeps the train's S1 density (twice the test's).
- **1% smoke test (Kaggle CPU, 275 s):** runs end to end. 26,818 copies; candidates 1,007,953 → 1,009,944 (the lists were
  full, so copies replace the weakest candidates); 2.43 distractors per S1. On the smoke test sample it linked 34% of French,
  16% of Indian and 10% of US S1 (almost all false there), against 12.7 / 6.4 / 4.6% for the tlu40 + FEAT-v4 smoke model.
  At 1% every rule scores within 0.0003 (threshold 0.675 chosen), so the rate is mostly the rule; the full-data eval decides.
- **Possible artifact:** a copied distractor ties with its copy, so its stage-2 `p1_margin_q` is never positive, which the
  test's distractors can be. Its stage-2 gain share is 0.2% in the smoke model; check it in the full run.

### M-v5 — the test-like universe, number-gap features, 75% of entities fitted

- **Date:** 2026-09-26 (00:01 IST)
- **Versions:** NORM-v2 + **BLK-v4b@20-tlu40** + **FEAT-v3** + **MATCH-v6** (preset M-v5). Run by chain8 on the runner (`pipe6/`).
- **Hypothesis:** training in a universe with the test's distractor density (~2.3 per S1) teaches the model test-like priors. The number-gap features (+0.0014 on DEV-10) and three times the fitted entities help on top.

**Methodology**
- **Universe:** every eval S1, plus 40% of the other train S1 entities with their true targets. That leaves 1,147,088 S1 (India 458,739, US 688,349) and 6.65M targets. Both countries have **5.80 targets and 2.34 distractors per S1**, against 5.76–5.82 and 2.34–2.35 in the test set.
- **Blocking (top-20):** 54.8M train candidates (47.8 per S1). Eval pair recall **98.69%**, against 98.32% in the full universe, since fewer rival records sit in each region. Test: 83.3M candidates, as before.
- **Features:** FEAT-v3 (55), computed in 1,862 s.
- **Model:** MATCH-v6, learning rate 0.1, fitted on all non-eval S1 of the universe: 33.7M training rows, 7.15% positive. Stage-1 folds took 236–289 s (388 / 547 / 463 trees); stage-2 folds 134–173 s (260 / 195 / 152 trees). Train stage 3,081 s, peak 23.2 GB. The whole run took 53 minutes.

**Results (test-like eval slice, 441,521 S1)**

| Score / rule | F0.5 | P | R | Singletons | Others |
|---|---:|---:|---:|---:|---:|
| Perfect matcher on these candidates | 0.9959 | | | | |
| Stage 1, gated expected F (gate 0.7) | 0.9834 | 0.9952 | 0.9604 | 0.9764 | 0.9839 |
| **Stage 2, gated expected F (gate 0.55)** | **0.9852** | 0.9957 | 0.9644 | 0.9808 | 0.9854 |
| Stage 2, gated expected F (gate 0.6) | 0.9852 | | | 0.9841 | 0.9852 |
| Stage 2, expected F (floor 0.4) | 0.9851 | | | 0.9764 | 0.9856 |

India 0.9818, US 0.9875.

- **Top stage-1 features by gain:** `margin_vs_other_s1` 56.7%, `rank_for_t` 20.0%, **`num_x_edit` 10.9%**, `al_b_worst` 1.3%, `al_b_n_unaligned` 1.2%, `a_b_in_a` 1.1%, `num_x_loggap` 1.0%. The number-gap features carry 12% of the gain. **Stage 2:** `p1_margin_t` 54.6%, `p1` 43.7%.
- **Test set:** 5,672,319 links for 1,630,014 of 1,732,544 S1 (94.1% linked). France 0.939 linked with 3.14 links per S1; India 3.29, US 3.31. The validator passes. md5 of `matching_results.tsv`: `9c66b92887f0eced0291bc42f78d8b2f`.

**Analysis**
- The test-like score (0.9852) is not comparable with FULL-v1's full-universe 0.9813. The test-like universe is easier in one respect (half the rival S1 records, so blocking and exclusivity lose less) and harder in another (twice the distractors per S1). chain10 scores the frozen FULL-v1 model in this universe, to put the two on one scale.
- **Public leaderboard: 0.971** (FULL-v1: 0.96; top of the board 0.99; the team is ranked ~400). Test-like training closed about half of the gap.
- **(Withdrawn 2026-09-26: the test-like universe has only half the test's hard lookalikes per S1, so its eval overstates
  every country; see "Test distractors are lookalikes of present S1 records" above.)** The rest of the gap is most likely France. The test-like eval (0.9852) covers only the US and India. If those countries score about 0.985 on the test too, 0.971 overall puts France (15% of test S1) at about **0.89**. That fits its 3× false-link rate in the 1% smoke check. Next: M-v6 (FEAT-v4, France-robust features).

### France — generic names draw about 3× the false links (FEAT-v4)

- **Date:** 2026-09-25
- **How we saw it without labels:** the 1% smoke sample draws test S1 records and test targets independently, so almost no true pair survives: nearly every predicted link is wrong. The M-v3 model linked **16.0% of French S1, against 6.8% of Indian and 4.6% of US S1**, so France gets about 3× the false links of the trained countries.
- **What the false links look like** (raw test records, S1 → linked target):
  - Same generic name, different street: "La Teste-de-Buch Maison SARL, 13 Square Clos des Chenes" → "La Teste-de-Buch Maison SAS, 57 RUE RAYMOND DAUGEY"; "Lille Parents SARL, 24 Cour Cacan" → "Lille Parents SARL, 1 Rue Du Chemin De Fer"; "Calais Compagnie SAS" → "Calais Compagnie S.A.S, 74 RUE DE VIC"; "Bordeaux Élémentaire SAS" → "SCI Bordeaux Élémentaire"; "Nantes Maison SARL" → "Nantes Maison SAS".
  - Invented name on the same street, different number: "TGO Comite SARL, 152 Rue du Jardin Public" → "Rizafaye, N° 35 RUE DU JARDIN PUBLIC".
- **Why:** French S1 names are often "<city> <generic word> <legal form>", and the test covers a handful of cities (Bordeaux, Lille, Nantes, Dunkerque, Saint-Nazaire…). Many unrelated businesses share such a name. French place names are long multi-token strings ("La Teste-de-Buch", "Nouvelle-Aquitaine", "Pays de la Loire"), so the city and region inflate token-set address similarity. The model, trained on US and India, reads "same name, same city" as a match.
- **No postal codes:** French addresses carry none, so house-number agreement is not inflated.
- **Fix, country-agnostic (FEAT-v4 = FEAT-v3 + 10 features):**
  - Per country, the tokens found in more than 1% of its records (S1 and targets) are frequent. These are cities, regions, street types, legal forms and generic words, learned from each country's own records, so France gets them from its test records.
  - Name and address similarity is recomputed on the distinctive parts, with those tokens removed: ratio, token-set, containment both ways, and the number of tokens left.
  - Two counts: how many targets share the S1's core name, and how many share the target's.
  - The model learns from generic US and Indian names ("Primary Care Group", "Cardiology Care") that a matching generic name is weak evidence, and applies that to France.
  - The FEAT-v3 columns are unchanged (exact-equality check).
- **How it is checked:** the TLU eval slice (US and India) must not drop, and the 1% smoke France link rate should fall toward the US and India rates.
- **1% smoke check** (share of test S1 given a link when almost no true pair survives, so almost all of these links are false):

| Model on the 1% sample | France | India | US |
|---|---:|---:|---:|
| M-v3 (FEAT-v2), full-train universe | 16.0% | 6.8% | 4.6% |
| TLU + FEAT-v2 + MATCH-v6 | 20.0% | 6.9% | 5.1% |
| **TLU + FEAT-v4 + MATCH-v6** | **12.7%** | 6.4% | 4.6% |

  FEAT-v4 removes about a third of France's false links in the same setup, and does not raise the other countries. France is still about twice as high as India and the US. The smoke models learn from 1% of train, so the absolute rates are noisy.

### dev_v3 — number-gap and support features on DEV-10

- **Date:** 2026-09-25
- **Setup:** `experiments/dev_v3.py` on the DEV-10 cache, same splits and folds as `dev_stack` (baselines: stage 1 0.9894 at threshold 0.725 and 0.9887 with expected F; stage 2 0.9904 / 0.9906).

| Variant | Threshold rule | Expected F |
|---|---:|---:|
| Stage 1 + number gap (FEAT-v3) | 0.9907 (0.8) | 0.9903 |
| Stage 2 + number gap | **0.9918** (0.725) | **0.9918** |
| Stage 2 + number gap + support (MATCH-v3) | 0.9918 (0.65) | 0.9919 |

- **Number gap:** +0.0013 at stage 1 and +0.0012 at stage 2 (singletons 0.9895 against 0.9935 before; others 0.9919 against 0.9902). `num_x_edit` ranks fifth by stage-1 gain. **Kept: FEAT-v3 and FEAT-v4 include it.**
- **Support features:** +0.0001, within noise, at an extra cost of ~20 minutes of full-data features. **Dropped.**

### Leaderboard gap — the test set has twice the distractors per S1

- **Date:** 2026-09-25
- **Trigger:** FULL-v1 scored **0.96 on the public leaderboard**, against 0.9813 on its eval slice. The top of the board is 0.99. Eval rows are scored exactly like test rows, so a 0.02 gap points to a difference between the train and test data, not to noise.
- **Scripts:** `experiments/shift_check.py` (leak, stats, preds) and `experiments/full_errors.py`, run on the EC2 work dir.

**1. No leak.** Over 7.64M train links, the Spearman correlation between the file row of an S1 record and the rows of its matches is +0.0010 (S2) and -0.0007 (S3). The numeric parts of the IDs give +0.0001 and +0.0002. The S2 and S3 rows of one entity give -0.0004. The top scores do not come from file order or ID numbers.

**2. The data differs.**

| Split | Country | S1 | Targets | Targets / S1 | S1 name shared | Target name = an S1 name | Target no address | Target no number |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | India | 883,188 | 4,133,346 | 4.68 | 0.541 | 0.609 | 0.030 | 0.092 |
| train | US | 1,323,633 | 6,186,873 | 4.67 | 0.477 | 0.526 | 0.036 | 0.091 |
| test | France | 259,452 | 1,434,993 | 5.53 | 0.477 | 0.565 | 0.030 | 0.067 |
| test | India | 809,986 | 4,717,565 | 5.82 | 0.534 | 0.543 | 0.024 | 0.075 |
| test | US | 663,106 | 3,817,031 | 5.76 | 0.399 | 0.472 | 0.029 | 0.074 |

**3. How many test targets are distractors?** In train, S2 and S3 hold almost exactly the same number of distractors (1,340,997 and 1,340,857), while the matched records split 3,693,619 : 3,944,746 (ratio 0.9363). The generator therefore seems to add the same number of distractors to each source. Assuming the same matched ratio in test, the S3 − S2 excess (195,043) gives the matched records, and the rest are distractors:

| | Matches / S1 | Distractors / S1 | Distractor share of targets |
|---|---:|---:|---:|
| Train US / India | 3.52 / 3.37 | **1.15 / 1.31** | 25% / 28% |
| Test US / India / France | 3.41 / 3.47 / 3.31 | **2.34 / 2.35 / 2.22** | 41% / 40% / 40% |

The test entities have as many matches as the train ones, but **twice as many distractors per S1**. The US test even has the same absolute number of distractors as the US train (0.78M vs 0.76M per source) with half the S1 records. The lower rates of missing addresses and numbers in test fit this: solved the same way, distractors almost always carry a full address and a number (about 1% miss one) and are never web or alias names, while about 12% of matched records lack a number.

**4. The model behaves the same on test.** Stage-2 profile per country, with eval rows on the full-train universe:

| Split | Country | S1 linked | Links / S1 | Top p2 > 0.95 | Links with p2 < 0.8 | Targets with max p2 > 0.5 |
|---|---|---:|---:|---:|---:|---:|
| eval | India | 0.942 | 3.29 | 0.935 | 0.005 | 0.713 |
| eval | US | 0.942 | 3.33 | 0.937 | 0.003 | 0.720 |
| test | France | 0.941 | 3.18 | 0.930 | 0.011 | 0.593 |
| test | India | 0.941 | 3.29 | 0.934 | 0.006 | 0.575 |
| test | US | 0.941 | 3.31 | 0.934 | 0.006 | 0.587 |

The model is just as confident and links as many records per S1 on test. The share of targets it claims falls with the matched share (~60%). So the extra test errors are *confident* ones, such as a lookalike taken instead of, or next to, the true record, and a stricter threshold cannot remove them. France is a little less certain than the other countries, but it does not collapse.

**Analysis**
- The eval slice never saw test-like conditions: the model learned its priors, and the competition features (`margin_vs_other_s1` and `rank_for_t` carry 82% of the stage-1 gain), at half the test's distractor density. In the US it also saw twice the test's S1 density.
- A linear estimate (twice the distractor wrong links) explains only ~0.003 of the 0.02. The rest must come from lookalikes that the denser, rival-rich train universe hid: in train, a lookalike is often claimed by its own S1, which is missing in test.
- **Fix, validation first:** a test-like train universe (`BLK-v4b@20-tlu40`). It keeps every eval S1 and 40% of the other train S1 entities, and removes the rest together with their true targets. That leaves ~2.3 distractors per S1, and a US S1 density close to the test's. Two runs use it:
  1. `MATCH-v2-frozen` scores the leaderboard model in it. If it lands near 0.96, the universe reproduces the leaderboard, and its eval slice becomes the validation to trust.
  2. `MATCH-v6` trains in it, so the model learns test-like priors and the rule is chosen under test-like conditions.

### Error analysis — FULL-v1 on the full-data eval slice

- **Date:** 2026-09-25
- **Setup:** `experiments/full_errors.py`, stage 2, gated expected-F 0.5, 441,521 eval S1 (1,528,407 true links, 1,463,512 predicted).

| Entity outcome | Entities | Loss (of 0.0187) |
|---|---:|---:|
| Some links missed, none wrong | 59,213 | **0.0109** |
| Has matches, predicted nothing | 1,593 | 0.0036 |
| Some links wrong, none missed | 4,030 | 0.0021 |
| Singleton given a link | 556 | 0.0013 |
| Missed and wrong links | 1,037 | 0.0007 |
| Has matches, every prediction wrong | 25 | 0.0001 |

| Missed links (70,681 = 4.62%) | Links | Share | Target without address |
|---|---:|---:|---:|
| Not chosen by the rule (own p2 mostly 0.05–0.8) | 26,773 | 37.9% | 32% |
| Never a candidate | 25,676 | 36.3% | 29% |
| Lost to another S1 under exclusivity | 18,232 | 25.8% | **91%** |

- **Recall is the main loss** (0.0145 of 0.0187). Wrong links are 5,786, 85% of them distractors; 44% have p2 above 0.9.
- **Not chosen:** true matches whose house number was replaced or perturbed (9052↔9053, 7732↔7730, 733↔831, 3785↔3559), a name word swapped for another (motors↔auto, medicine↔partners), or numbers dropped. Some clear matches still score low, e.g. identical address with a typo-heavy name (ficus vidyalaya ↔ ficus viddyalmaya, p2 0.32).
- **Lost to another S1:** no-address targets whose name several S1 records share ("all insurance", "eye care", "micki hammond"). The winner has the same core name 77% of the time, and its own p2 averages only 0.26, so the target often goes to nobody. Mostly irreducible from the pair alone.
- **Never a candidate:** initials or acronyms ("chorus vanijya" ↔ "cvprivate", "cosmos brothers" ↔ "bc"), invented names ("kundan multimedia" ↔ "nylaquo"), and plain typos that still rank below 20 lookalikes ("lakshmi international" ↔ "lksmi international", same address). Top-40 blocking recovers 0.61 points of links.
- **Wrong links:** near-copies of the S1 record that belong to no S1: the same number with a unit letter ("3027 c douglas ave"), a nearby number plus a PMB, or a name variant at the same address.

### FULL-v1 matcher — M-v3 on the full data

- **Date:** 2026-09-25
- **Git commit:** `9b5c959`. The runner bundles were packed from the working tree shortly before that commit: `pipe/` for training, and `pipe3/` for the predict re-run with the gated rule.
- **Hypothesis:** DEV-10 has about 10 times less competition per region than the full data, so its 0.9906 is optimistic. Training and scoring on the full data, with the eval slice contested as densely as the test set, gives the score to expect on the leaderboard.

**Methodology**
- Preprocessing, blocking and features: NORM-v2, BLK-v4b@20 (see the FULL-v1 blocking entry below), FEAT-v2 (52 features).
- Model: MATCH-v2. Stage 1 and stage 2 are both cross-fitted over 3 folds of the fit entities (30% of train S1), early-stopped on 5%. Training rows (fit + early stop): 36,648,850, 7.17% positive. LightGBM, learning rate 0.05, 255 leaves.
- Stage 2 has 64 features: the 52 of stage 1, plus `p1` and its 11 context columns.
- Decision: exclusive assignment, then the rule and its parameter chosen on the eval slice among threshold (0.20–0.95), expected F (floors 0.05–0.4) and gated expected F (gates 0.30–0.95).

**Results (eval slice, 441,521 S1; 20% of train S1, never fitted)**

| Score / rule | F0.5 | P | R | Singletons | Others |
|---|---:|---:|---:|---:|---:|
| Perfect matcher on these candidates | 0.9947 | 1.0000 | 0.9833 | 1.0000 | 0.9944 |
| Stage 1, threshold 0.725 | 0.9791 | 0.9947 | 0.9494 | 0.9756 | 0.9793 |
| Stage 1, gated expected F (gate 0.7) | 0.9793 | 0.9948 | 0.9492 | 0.9737 | 0.9796 |
| Stage 2, threshold 0.65 | 0.9809 | 0.9945 | 0.9564 | 0.9848 | — |
| Stage 2, expected F (floor 0.4) | 0.9813 | 0.9951 | 0.9544 | 0.9759 | 0.9816 |
| **Stage 2, gated expected F (gate 0.5): chosen** | **0.9813** | 0.9952 | 0.9544 | 0.9774 | 0.9815 |

| Country | Eval S1 | F0.5 | P | R | Singletons | Others |
|---|---:|---:|---:|---:|---:|---:|
| India | 176,610 | 0.9778 | 0.9942 | 0.9463 | 0.9743 | 0.9780 |
| US | 264,911 | 0.9837 | 0.9958 | 0.9597 | 0.9795 | 0.9839 |

- **Best iterations:** stage 1 1,392 / 1,277 / 1,469; stage 2 573 / 444 / 454.
- **Top stage-1 features by gain:** `margin_vs_other_s1` 55%, `rank_for_t` 27%, `num_jacc` 3.8%, then `al_b_worst`, `n_tsort`, `al_b_unaligned_idf`, `num_b_in_a`. **Stage 2:** `p1` 71%, `p1_margin_t` 23%, `p1_rank_t` 3.5%.
- **Test set:** 5,683,607 links for 1,630,378 of 1,732,544 S1 (5.9% empty). The share of S1 with a link is 0.941 in every country, **France included** (3.18 links per S1; India 3.29, US 3.31). The eval truth has 0.944 and 3.46. The official validator passes.
- **Runtime (EC2 r7a.2xlarge, 8 cores):** 4 h 46 min wall clock, peak 44.4 GB. Prep 386 s, block 1,202 s, features 2,344 s, train 13,137 s, predict 90 s. Training includes 2.4 h of stage-1 scoring, slowed by a second job that competed for the CPU.
- **Output:** `predictions/M-v3/` in S3 and the repo's `output/` (md5 of `matching_results.tsv`: `b4644450c78b5bf64f8124223223a005`). The first predict, with expected F at floor 0.4, is kept in `predictions/full/`; it scores the same to 6 decimals.

**Analysis**
- **Full data costs 0.009 against DEV-10** (0.9906 → 0.9813), almost all of it in recall (0.9794 → 0.9544). Precision barely moves (0.9963 → 0.9952).
- **The matcher now loses more than blocking does.** Blocking caps the score at 0.9947 (a loss of 0.0053), and the matcher loses a further 0.0134. 2.9% of the true links are among the candidates but are not predicted; on DEV-10 that figure was 1.5%. Denser competition hurts twice: a target has more rival S1 records that claim it, and each S1 record has more lookalike candidates.
- The decision rule hardly matters any more. The best threshold, expected-F and gated rules are within 0.0004 of each other. The gains have to come from better probabilities.
- France gets the same prediction profile as India and the US, so the pipeline does not collapse on the unseen country, but its score cannot be measured.
- **Next:** MATCH-v6 (fit on 75% of train S1 instead of 30%, on the cached features), the BLK-v4b@40n20 depth curves (for deeper blocking in BLK-v5), and an error analysis of the full-data eval misses.

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
