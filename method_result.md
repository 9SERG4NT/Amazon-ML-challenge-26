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

- **Validation split:** _TBD._ Proposed default: hold out 10% of train S1 entities with a fixed seed of 42, stratified by country and by number of matches. The full train S2/S3 pool stays as the search space so that blocking difficulty is realistic.
- **Blocking metrics:**
  - Pair recall: the share of true links found among the candidates.
  - Mean candidates per S1 entity.
  - Reduction ratio.
- **Matching metrics:** macro F_0.5, macro precision, and macro recall on the validation set, reported separately for singletons and non-singletons, and for each country.
- **Threshold:** tuned to maximise validation F_0.5, with the chosen value recorded.
- **Format check:** run `utils/validate_submission.py` on every run that produces a submission.

---

## 4. Results summary

| ID | Date | Method (short) | Blocking recall | Candidates / S1 | Val F_0.5 | Val P | Val R | LB F_0.5 | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| — | — | _no runs yet_ | | | | | | | |

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

_No experiments yet._
