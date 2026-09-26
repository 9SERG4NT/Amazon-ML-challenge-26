# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary

We match records in four steps. An IDF-weighted sparse search generates candidates, and a small
learned filter keeps the plausible ones (**6.5 per Source 1 record**). A two-stage LightGBM
classifier scores each (Source 1, candidate) pair, and a per-entity decision turns probabilities
into the match list that maximises macro F0.5. Four ideas carry most of the accuracy:

- **Competition features.** Every Source 2/3 record belongs to at most one Source 1 entity, so a
  candidate is judged against the rival Source 1 records that want the same record.
- **A validation set built to look like the test set.** The test set has about twice as many
  distractor records per Source 1 entity as the training set. We found this from the per-source
  record counts after our first leaderboard score (0.96) fell short of our held-out score
  (0.9813). We therefore train, tune and validate in a *test-like universe* made from the training
  data.
- **Country-agnostic features for the unseen country.** French names are often "<city> <generic
  word> <legal form>". Similarity is therefore also measured on the distinctive part of each name
  and address, after removing the tokens that are frequent in that country's own records.
- **A learned candidate filter.** The search keeps 48 candidates per record to find 98.7% of the
  true links. A small LightGBM that sees only the search scores and their competition context
  cuts that to 6.5 at a cost of 0.0002 F0.5.

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the training files (2.2M Source 1, 5.0M Source 2, 5.3M Source 3 records):

- **One-to-many, never many-to-many.** No Source 2/3 record is linked to two Source 1 entities.
  A record claimed by several Source 1 entities can therefore go to at most one of them.
- **Matches per entity:** mean 3.46, at most 5 from Source 2 and 6 from Source 3. 5.6% of entities
  have no match; each such singleton scores 1.0 only if nothing is predicted for it.
- **Distractors:** 26% of Source 2/3 records match nothing. Only 5–6% of them share an exact name
  with a Source 1 record; the hard negatives are same-name businesses at a *different address*.
- **Shared names:** 39.6% of Source 1 records share their normalised name with another Source 1
  record ("Primary Care Group" appears 253 times), so the address must break ties.
- **Noise:** abbreviations and legal-form variants, typos (including digit-for-letter, "5ummit"),
  word-order changes, website-style names, "formerly known as / doing business as" names, invented
  trade names ("Korevo" for "Asahi Charitable Trust"), Devanagari names in 7–13% and addresses in
  ~13% of Indian Source 2/3 records, house-number perturbations, and missing addresses (~3%).
- **Country** never differs between linked records. Test adds France, absent from training.
- **Train and test differ.** No leak: the file row order and the numeric part of the IDs are
  uncorrelated with the ground truth (Spearman |ρ| ≤ 0.001 over 7.6M links). In training,
  Source 2 and Source 3 hold almost exactly the same number of distractors (1,340,997 vs
  1,340,857), while matched records split 0.936 : 1. Applying that to the test files, test entities
  have as many matches as training ones (3.3–3.5), but **2.2–2.35 distractors per Source 1 entity
  against 1.15–1.31 in training**, about 40% of test records against 26%. The US test set also has
  half the training set's Source 1 density. Distractors are "clean" records: full address and
  house number, never website or alias names.
- **France:** French Source 1 names are often a city plus a generic word plus a legal form
  ("Nantes Maison SARL", "Lille Amis SCI"). The test covers only a few cities, so many unrelated
  businesses share such a name, and multi-word place names ("La Teste-de-Buch",
  "Nouvelle-Aquitaine") inflate address similarity. French addresses carry no postal codes.

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned candidate filter + two-stage classifier + per-entity set decision  
**Core Innovation:** competition-aware scoring (rank and probability margin against rival Source 1
records, then exclusive assignment), a test-like validation and training universe,
distinctive-part similarity for an unseen country, and a learned filter that keeps the candidate
set small.

Every component is a named version recorded in `src/ber/versions.py`: normalisation (NORM),
blocking (BLK), features (FEAT) and matcher (MATCH). Every stage caches its output under the
versions it depends on, so any logged result can be re-run by name. The submitted run is
preset **M-v11 = NORM-v2 + BLK-v5-tlu40 + FEAT-v4 + MATCH-v6**.

---

## 3. Candidate Generation (Blocking)

- **Normalisation (NORM-v2):** transliteration to ASCII (anyascii), canonical legal forms, street
  types, US states and ordinals, Indian state names joined into one token, French street types,
  digit-for-letter repairs, "fka / dba" names split, website names turned into words. On top,
  **per-country token aliases learned from the training links** (1,105 name and 3,107 address
  aliases, e.g. `praivet → pvt`, `mh → maharashtra`); France gets none, by design.
- **Blocking keys used:** each record becomes a sparse vector of hashed features — name and
  address tokens, 4-character prefixes, consonant skeletons, leading/trailing digits of house
  numbers, the whole joined name, and character 3-grams of the joined name. Features are
  IDF-weighted on the target side, name and address blocks are normalised separately, and the
  score is cos(name) + cos(address). The exact top-k per (Source 1 record, target source) is
  found with a multi-threaded sparse top-n product (`sparse_dot_topn`, Apache-2.0).
- **Region split:** within a country, the search runs inside region groups — frequent last
  tokens of Source 1 addresses that sit at the end of an address (states), with merges learned
  from training links that cross them (Telangana ↔ Andhra Pradesh, Delhi ↔ Haryana). Records
  without a region token are searched against everything. France, having no training links, is
  searched whole.
- **Name-only pass** for targets without an address, which would otherwise lose to same-name
  records that have one.
- **Search output:** top 20 per Source 1 record and target source, plus the top 5 targets without
  an address: 83.3M test pairs (48.1 per Source 1 record). It scales: every record is compared only
  with records of its own country and region through a sparse index, never all pairs, and the
  cost grows with records × k.
- **Learned candidate filter (second blocking stage, BLK-v5-tlu40).** Most of the 48 are
  low-ranked rivals: each Source 1 record has ~3.4 true links. Cutting the search depth uniformly
  is a poor trade (top-5 per source: 17.8 candidates, −2.2 points of recall). Instead:
  - A small LightGBM (63 leaves, 3 folds) sees only what the search produced. Its inputs are the
    source, the blocking score and its name/address cosines, the name-only flag, and the
    competition context of those scores: the candidate's rank in its record's list, the gap to
    the record's best, how many Source 1 records retrieved the target, this record's rank among
    them and its margin over the best one. It computes no string similarity, so it stays cheap.
  - It is trained like the matcher: cross-fitted over the fit entities, out-of-fold on its own
    training rows.
  - Its threshold keeps 99.5% of the true links the search found for fit entities. It is set
    out-of-fold, never on the evaluation slice.
  - Features and the matcher then see **only the kept candidates**, so `candidate_pairs.tsv` is
    exactly the set the model scores.
- **Candidate pairs generated:** **11.19M for the test set, 6.5 per Source 1 record** (median 6,
  90th percentile 10, 99th 14, maximum 40; 0.03% of records have none), against 48.1 before the
  filter.
- **How we ensured true matches were not lost:** recall is measured on held-out training entities.
  - With every training entity in the search, BLK-v4b@20 finds 98.32% of the true links: 88.7% for
    targets without an address, 98.8% for the rest.
  - In the test-like universe it finds 98.69%, because each region holds fewer rival records.
  - A depth measurement (top 40 in the main pass, top 20 in the name-only pass) gives 95.99 /
    97.40 / 98.28 / 98.70 / 98.93% at main-pass depths 5 / 10 / 20 / 30 / 40.
  - The name-only pass adds only 0.11 points between depth 5 and 20, so it stays at 5.
  - The filter keeps 98.21% of the evaluation slice's true links (search alone: 98.69%). A
    perfect matcher on the kept candidates would score 0.9945 (0.9959 on the search output). The
    trade-off curve on the full data (share of found links kept → evaluation recall, candidates per
    test record): 99% → 97.73%, 5.7; **99.5% → 98.21%, 6.5**; 99.7% → 98.40%, 7.1; 99.9% → 98.59%, 8.5.
  - The matcher's F0.5 changes from 0.9856 (search output) to 0.9854 (filtered). It would have
    missed most of the dropped links anyway, and precision rises slightly.

---

## 4. Matching Model

**Features used** (65 in stage 1):
- **Name features:** rapidfuzz ratio, token-set, token-sort and partial ratios, and Jaro-Winkler
  on the full, core (legal words removed) and joined names; best match against "fka / dba" parts;
  token containment both ways; **soft word alignment** (each core-name token aligned to its best
  Jaro-Winkler match, both directions: mean, worst, number of unaligned tokens and the highest IDF
  among them — separating typos from a different business).
- **Address features:** ratio, token-set, token-sort and partial ratios; token containment;
  house-number agreement (Jaccard, share of target numbers present, truncated-number match).
- **Number-gap features (FEAT-v3):** for the closest pair of house numbers found on one side
  only, the digit edit distance, the log numeric gap and the relative gap. Planted distractors sit
  at a nearby number (831 vs 835), while true matches carry digit typos (9052 vs 9053) and
  replaced numbers. On the development sample these three features lift stage 2 from 0.9904 to
  0.9918.
- **Distinctive-part features (FEAT-v4):**
  - The frequent tokens of a country are those in more than 1% of its records, learned from that
    country's own records: cities, regions, street types, legal forms, generic words. France
    therefore gets them without any labels.
  - Name and address similarity is also computed with those tokens removed: ratio, token-set,
    containment, and the number of tokens left.
  - Two counts are added: how many target records share the Source 1 core name, and how many share
    the target's.
  - The model learns from generic US and Indian names ("Primary Care Group") that a matching generic
    name is weak evidence. In a sample where almost no true pair survives, these features cut the
    share of French Source 1 records given a (false) link from 20.0% to 12.7%.
- **Competition and context features:** blocking score and name/address cosines; rank of the
  candidate within its Source 1 record (per source and overall); gap to the best candidate;
  number of candidates; **number of Source 1 records that retrieved the target, this record's rank
  among them, and its score margin over the best rival**; how many Source 1 records share the name.
- **Stage 2** adds the stage-1 probability and its context: rank among the record's candidates and
  among the target's Source 1 records, best rival probability and the margins over it.

**Model type:** LightGBM (MIT), binary log loss, 255 leaves, early stopping. The submitted
matcher (MATCH-v6) uses learning rate 0.1 and fits 75% of the training entities (the first run:
0.05 and 30%). Stage 1 and stage 2 are both **cross-fitted** over 3 folds of the fit entities,
so every probability used downstream is out-of-sample. After the filter it trains on 4.2M rows
(56.8% positive) in about 10 minutes on 8 cores.

**Loss function, chosen from the data:**
- The decision rule reads the scores as probabilities, so the loss must be a proper scoring rule:
  **binary log loss**.
- **Not focal loss:** after the filter the classes are balanced, and the negatives left are the
  hard lookalikes, so there is no flood of easy negatives to down-weight.
- **Not AdaBoost's exponential loss:** it lets label noise dominate (invented trade names cannot
  be learned) and gives no probabilities.
- Tried on the full data, on the same candidates and evaluation slice, and none beat plain log
  loss (0.9854):
  - distractor rows weighted ×2, for the test's doubled lookalike density: 0.9852;
  - each row weighted by what its error costs its entity's F0.5, the metric's own asymmetry:
    0.9851 (the decision rule already applies these costs, so the weights count them twice);
  - a feed-forward neural network (numpy, Adam, learning-rate decay, early stopping): 0.9825;
  - its average with LightGBM: 0.9847;
  - CatBoost (Apache-2.0): 0.9850; averaged with LightGBM: 0.9854 (a tie, so the simpler single
    model is kept); LightGBM + network + CatBoost: 0.9851.
- The model family and the loss are therefore no longer the bottleneck. The remaining loss is
  recall on links whose evidence is missing (invented names, replaced house numbers, no address).

**Threshold selection method:** each target is kept only for the Source 1 entity that scores it
highest (exclusive assignment); then a rule chosen on held-out training entities by macro F0.5:
a single threshold, expected-F (per entity, the top-k maximising expected F0.5 under the
predicted probabilities, or no match), or gated expected-F.

**Validation:** Source 1 training entities are split once: fit 30%, early stopping 5%,
evaluation 20% (441,521 entities), rest 45%. Aliases and region merges are learned without the
evaluation links, and evaluation rows are scored exactly like test rows.

- **The full-train universe was not enough.** Our first full run scored 0.9813 there but 0.96 on
  the public leaderboard: with every training entity in the search, each Source 1 entity faces
  half the test's distractor density.
- **The test-like universe (BLK-v4b@20-tlu40)** keeps every evaluation entity and 40% of the
  other training entities, and removes the rest together with their true matches, while every
  distractor stays.
- **The result matches the test profile:** 5.80 targets and 2.34 distractors per Source 1 entity
  in both countries, against 5.76–5.82 and 2.34–2.35 in the test set.
- **Everything happens in this universe:** blocking, features, training and the choice of
  decision rule. The evaluation slice remains the same 441,521 entities. The earlier model is
  also scored there without retraining, to check that the universe reproduces its leaderboard
  score.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9854** on the test-like evaluation slice (441,521 entities;
  precision 0.9962, recall 0.9639; India 0.9817, US 0.9879); **public leaderboard 0.970**.
  On the test set it links 94.0% of Source 1 records with 3.1–3.3 links each (evaluation truth:
  94.4% and 3.46). For reference:
  - The first full run (M-v3, full-train universe) scored 0.9813 on the evaluation slice
    (precision 0.9952, recall 0.9544; India 0.9778, US 0.9837) and 0.96 on the public
    leaderboard. A perfect matcher on its candidates would score 0.9947.
  - On the development sample (10%, 44,137 entities), stage 2 with expected-F scored 0.9906.
- **Where the first full run lost F0.5 (0.0187 in total):** 0.0145 is recall.
  - Of the 4.6% of true links it missed, 38% were candidates that the decision rule did not
    choose. These are mostly true matches whose house number was replaced or perturbed, or whose
    name had one word swapped ("motors" / "auto").
  - 36% were never candidates: initials or acronym names ("chorus vanijya" / "cvprivate"),
    invented names, typo-heavy names.
  - 26% were taken by another Source 1 entity. 91% of these are targets without an address whose
    name several Source 1 entities share, which is mostly irreducible.
- **The unseen country.** The evaluation slice cannot score France, which is 15% of the test entities and has no
  training links. Leave-one-country-out runs measure what an unseen country costs. A matcher fitted on the US
  alone scores India at 0.9444 (0.9817 when India is fitted), and one fitted on India scores the US at 0.9637 (0.9879),
  in precision and recall alike. No decision rule recovers it (the best rule on India's own labels: 0.9446). We tried
  three remedies, none of which moved the evaluation:
  - self-training on the unseen country's confident candidates: +0.0014 on India;
  - a hyperparameter search scored across countries (Optuna, with monotone constraints and extra-trees in the space);
  - keeping small house numbers, which France's frequency filters dropped. 19.3% of French candidates looked like
    the same address with another number, but the final model already linked only 0.46% of such pairs.

  French predictions read correctly on inspection. This is the most likely part of the gap between the evaluation
  (0.985) and the leaderboard (0.970).
- **Common false positives (wrong merges):** 85% are distractors that look like a copy of the
  Source 1 record: the same name at a nearby number, the same number with a unit letter
  ("3027 c douglas ave"), or a name variant at the same address. In France, the same generic
  name on a different street.
- **Common false negatives (missed matches):** as above. Records without an address, perturbed
  house numbers, and invented or acronym names.

| Version | Data | F0.5 | Precision | Recall |
|---|---|---:|---:|---:|
| Single model, 44 features | development sample | 0.9882 | 0.9954 | 0.9745 |
| + soft word alignment (52) | development sample | 0.9894 | 0.9959 | 0.9772 |
| + stage 2, expected-F | development sample | 0.9906 | 0.9963 | 0.9794 |
| + number-gap features (55) | development sample | 0.9918 | 0.9975 | 0.9810 |
| M-v3, full-train universe | full data, evaluation slice | 0.9813 | 0.9952 | 0.9544 |
| M-v3 | public leaderboard | 0.96 | | |
| M-v5: test-like universe, number-gap features, 75% fitted | test-like evaluation slice | 0.9852 | 0.9957 | 0.9644 |
| M-v5 | public leaderboard | 0.971 | | |
| M-v6: + distinctive-part features | test-like evaluation slice | 0.9856 | 0.9960 | 0.9647 |
| **M-v11: M-v6 + learned candidate filter (6.5 candidates per record, not 48.1)** | test-like evaluation slice | **0.9854** | 0.9962 | 0.9639 |
| M-v11 | public leaderboard | **0.970** | | |

---

## 6. Conclusion

Treating the problem as competition for records, rather than as independent pair
classification, and validating under test conditions were the decisive choices. Validation had
to be fixed twice:

- A convenient 10% sample hid a 1.1-point blocking-recall loss that only full-density validation
  exposed.
- Full-density validation still missed that the test set has twice the distractors per entity.
  The per-source record counts revealed it, and a test-like universe built from the training data
  corrected it.

The organisers' emphasis on small candidate sets added a third lesson. The search needs depth to
find hard links, but the matcher does not need to see that depth. A cheap learned filter on the
search's own scores cut the candidate set 7.4-fold at a cost of 0.0002 F0.5.

Final submission: M-v11, test-like evaluation F0.5 0.9854, public leaderboard 0.970. It scores the same as M-v5
(0.971) within rounding, with a candidate set 7.4 times smaller.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: `src/run_pipeline.py` runs every stage (prep → block →
features → train → predict) and writes `output/matching_results.tsv` and
`output/candidate_pairs.tsv`, then runs the official validator. `src/ber/` holds the components
(`normalize`, `aliases`, `blocking`, `features`, `model`, `metrics`, `partition`, `versions`).
`src/experiments/` holds the experiments behind each version: development-sample studies, the
full-data error analysis (`full_errors.py`) and the train/test shift checks (`shift_check.py`).
`README.md` gives the exact command and `requirements.txt` the pinned environment (Python 3.12).

### B. Additional Results

Full experiment log with every version, including the full-data tables, the shift analysis and
the France study: `method_result.md`.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
