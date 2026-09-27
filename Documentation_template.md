# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team_Sumukh  
**Team Members:** Satvik Barhanpure, Sumukh Chourasia, Jatin Patin, Parambrata Sanyal  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

Our submission combines the team's two complete pipelines.

- **Pipeline A** (sections 3 and 4 describe it in detail): an IDF-weighted sparse search and a small learned
  filter generate candidates (**6.5 per Source 1 record**). A two-stage LightGBM classifier scores each (Source 1,
  candidate) pair, with the match probabilities of three fine-tuned transformer **cross-encoders** among its
  features. A per-entity decision turns the probabilities into the match list that maximises macro F0.5.
  Test-like evaluation F0.5 0.9887.
- **Pipeline B:** a fine-tuned multilingual bi-encoder with FAISS search generates candidates (6.2 per record),
  and XGBoost scores them, led by a cross-encoder score. Public leaderboard 0.980.
- **The combination (COMB-v1):** a LightGBM reads both pipelines' probabilities and their competition context
  on the union of their candidates (**7.5 per record**) and decides each entity's match list. It is trained on the
  training entities that both pipelines held out, where it scores F0.5 **0.9914** (A alone 0.9885, B alone
  0.9877). **Public leaderboard 0.98462.**

The ideas that carry most of the accuracy:

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
- **Cross-encoder features.** Two small transformers (a 4-layer BERT on normalised text and
  multilingual-e5-small on the raw text, both MIT/Apache-2.0) are fine-tuned to score candidate
  pairs. They are cross-fitted, so the classifier never sees a score from a model trained on the
  same entity. They raised the evaluation F0.5 from 0.9854 to 0.9885 and the leaderboard score
  from 0.970 to 0.978; a third (multilingual-e5-base) added 0.0002 more (0.9887).
- **Combining two different pipelines.** The two pipelines search differently (sparse tokens against dense
  embeddings) and each finds true links the other misses. Learning how to weigh them, on entities neither
  had trained on, lifted recall at the same precision: 0.980 / 0.978 alone, **0.98462** combined.

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

**Approach Type:** two pipelines (blocking + learned candidate filter + classifier + per-entity set decision), combined
by a learned second stage on the union of their candidates  
**Core Innovation:** competition-aware scoring (rank and probability margin against rival Source 1
records, then exclusive assignment), a test-like validation and training universe,
distinctive-part similarity for an unseen country, a learned filter that keeps the candidate
set small, and a learned combination of two independent pipelines.

Every component is a named version recorded in `src/ber/versions.py`: normalisation (NORM),
blocking (BLK), features (FEAT) and matcher (MATCH). Every stage caches its output under the
versions it depends on, so any logged result can be re-run by name. The submitted run is
**COMB-v1**: pipeline A's preset **M-v29 = NORM-v2 + BLK-v5-tlu40 + FEAT-v10 + MATCH-v6** (M-v11
plus three cross-encoder scores and their competition context), pipeline B (`team_pipeline/`), and
the combination (`src/experiments/combine_team.py`, section 4.3).

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

**Pipeline B's candidate generation.**
- `intfloat/multilingual-e5-small` (MIT, 118M parameters) is fine-tuned as a bi-encoder on 2M (Source 1, matching
  record) pairs with an in-batch contrastive loss. Half of the batches come from one city, which makes the in-batch
  negatives hard.
- A FAISS IVF index, searched per country, gives each Source 2/3 record its 3 closest Source 1 records and each
  Source 1 record its 10 closest records. This is sub-linear search, never all pairs.
- A rule on this neighbour graph keeps candidates with a per-country percentile cut-off, so it adapts to France's
  score scale. A logistic-regression graph filter, on graph features only, removes about 10% more.
- Result: 98.09% of held-out true links found with 4.89 candidates per validation record, and 6.18 per test record.

**The submitted candidate set is the union of both pipelines' candidates**: 12.96M test pairs, **7.48 per Source 1
record** (pipeline A 11.19M, B 10.71M). On the training entities that both pipelines held out, the union holds
**99.31%** of the true links. Alone, A finds 98.21% on its evaluation slice and B 98.09% on its held-out entities.
`candidate_pairs.tsv` is exactly this set, and the combination scores every pair in it.

---

## 4. Matching Model

### 4.1 Pipeline A

**Features used** (83 in stage 1: 65 string, number and context features, plus 18 from the
cross-encoders):
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
- **Cross-encoder features (FEAT-v10).** A cross-encoder reads the two records of a pair together
  ("name | address" of each), so it compares them token by token: typos, transliterations,
  abbreviations, reordered or swapped words.
  - **ce1:** `google/bert_uncased_L-4_H-256_A-4` (Apache-2.0, 11M parameters) on our normalised
    ASCII texts, trained on 8 CPU cores.
  - **ce2:** `intfloat/multilingual-e5-small` (MIT, 118M parameters) on the raw texts, so it also
    reads Devanagari and French accents; trained on two T4 GPUs.
  - **ce3:** `intfloat/multilingual-e5-base` (MIT, 278M parameters) on the raw texts, trained like ce2.
  - Each is a one-logit classifier fine-tuned with binary cross-entropy for one epoch (600k and
    1.2M pairs).
  - **Leakage control:** the fit entities are split into two halves by a hash of the Source 1
    row. The half-A model scores the half-B pairs and vice versa, and every other pair
    (early-stopping, evaluation, test) is scored by a model that never saw its entity. So the
    LightGBM only ever reads out-of-sample scores, as for its own stage 2.
  - Alone they separate the evaluation pairs with log loss 0.061 (ce1), 0.047 (ce2) and 0.043 (ce3);
    the full LightGBM reaches about 0.040.
  - Each score comes with its competition context: rank and margin over the next candidate of the
    same Source 1 record and over the best rival Source 1 record for the same target, and the
    record's top score.
  - Both models are far below the 8B-parameter limit and carry MIT/Apache-2.0 licences.
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
- The model family and the loss were therefore no longer the bottleneck; the features were. The
  cross-encoder scores, a new kind of evidence rather than a new learner, gave the largest gain
  since the test-like universe: evaluation 0.9854 → 0.9885 (precision 0.9962 → 0.9983,
  singletons 0.9816 → 0.9934) and leaderboard 0.970 → 0.978.

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

### 4.2 Pipeline B

- **Model:** XGBoost (Apache-2.0).
- **Features:** string similarities; per-country IDF token overlap; the bi-encoder's neighbour-graph features
  (similarity, forward and reverse ranks and gaps); candidate context; competition margins (similarity minus the
  best competing pair); and name-ambiguity counts.
- **Cross-encoder score:** multilingual-e5-small, fine-tuned only on the entities the bi-encoder was trained on.
  It carries 75% of the model's gain.
- **Selection:** exclusive assignment, then global thresholds or per-entity expected F0.5, whichever wins on its
  held-out 10% of training entities.
- **France:** a label-free logit shift makes France's share of empty predictions equal to the trained countries'.
- **Result:** validation F0.5 0.9875; public leaderboard 0.980.

### 4.3 The combination (COMB-v1)

- **Inputs:** for every pair in the union of both candidate sets:
  - each pipeline's probability (missing when that pipeline's search did not produce the pair) and a flag for it;
  - each probability's competition context: rank and margin over the next candidate of the same Source 1 record,
    margin over the best rival Source 1 record for the same target, the record's top score and candidate count;
  - the two probabilities' mean and difference.

  16 features in total.
- **Training without leakage:** pipeline B holds out 10% of the training entities from all its models. Pipeline
  A's scores are out-of-fold or come from models that never saw the entity. The 87,911 training entities that
  are both in B's held-out 10% and in A's test-like universe (571,098 union pairs) are therefore out-of-sample
  for both pipelines.
- **Model and rule:** a LightGBM (binary log loss, 63 leaves, early stopping) is cross-fitted over these entities
  in 5 folds. The decision rule is chosen on its out-of-fold probabilities: exclusive assignment, then gated
  expected F0.5 with gate 0.65. Test pairs get the mean of the 5 fold models.
- **France:** the same label-free shift as pipeline B's, applied to the combined probabilities (logit +1.15).
  France then leaves 5.82% of its Source 1 records empty, as the other countries do, with 3.31 links per record.
  That matches the estimate from the per-source record counts (3.31). Without the shift: 3.39.
- **Why it helps:** on the same entities, a plain average of the two probabilities already scores 0.9905 against
  0.9885 and 0.9877 for each alone. The learned combination scores 0.9914 (precision 0.9982, recall 0.9772,
  singletons 0.9957). Most of the gain is recall at the same precision: each pipeline finds true links the other
  misses. The combination is deterministic: two runs give byte-identical files.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), final submission (COMB-v1): 0.9914** on the 87,911 training entities both pipelines held
  out (precision 0.9982, recall 0.9772), and **0.98462 on the public leaderboard**. It links 94.2% of the test
  Source 1 records in every country, with 3.31–3.39 links each.
- **Pipeline A alone (M-v28):** **0.9885** on the test-like evaluation slice (441,521 entities;
  precision 0.9983, recall 0.9682, singletons 0.9934; India 0.9864, US 0.9900; 0.9882 when
  links to distractors count twice, the test's lookalike density); **public leaderboard 0.978**.
  On the test set it links 94.0–94.2% of Source 1 records with 3.2–3.4 links each (evaluation
  truth: 94.4% and 3.46; France 3.22 against a label-free estimate of 3.31). For reference:
  - Without the cross-encoder features (M-v11): 0.9854 on the evaluation slice, 0.970 on the
    leaderboard.
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
  (0.985) and the leaderboard (0.970). The cross-encoder features narrowed it: they added 0.003 on the evaluation
  slice but 0.008 on the leaderboard. That fits a pretrained multilingual text model transferring to an unseen
  country better than hand-made string features.
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
| M-v11: M-v6 + learned candidate filter (6.5 candidates per record, not 48.1) | test-like evaluation slice | 0.9854 | 0.9962 | 0.9639 |
| M-v11 | public leaderboard | 0.970 | | |
| M-v25: + cross-encoder ce1 (4-layer BERT) | test-like evaluation slice | 0.9875 | 0.9979 | 0.9664 |
| M-v26: + cross-encoder ce2 (multilingual-e5-small) instead | test-like evaluation slice | 0.9884 | 0.9981 | 0.9680 |
| **M-v28: + both, with their competition context** | test-like evaluation slice | **0.9885** | 0.9983 | 0.9682 |
| M-v28 | public leaderboard | 0.978 | | |
| M-v29: + a third cross-encoder (multilingual-e5-base) | test-like evaluation slice | 0.9887 | 0.9984 | 0.9685 |
| Pipeline B (bi-encoder + XGBoost with a cross-encoder) | its held-out 10% of training entities | 0.9875 | | |
| Pipeline B | public leaderboard | 0.980 | | |
| M-v29 alone | entities both pipelines held out (87,911) | 0.9885 | 0.9984 | 0.9680 |
| Pipeline B alone | entities both pipelines held out | 0.9877 | 0.9973 | 0.9670 |
| Average of the two probabilities | entities both pipelines held out | 0.9905 | 0.9973 | 0.9769 |
| **COMB-v1: learned combination** | entities both pipelines held out | **0.9914** | 0.9982 | 0.9772 |
| **COMB-v1** | **public leaderboard** | **0.98462** | | |

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

The last lesson came from a second pipeline in our team, a fine-tuned bi-encoder with a
cross-encoder feature. Once string features, learners and losses had saturated, a
different kind of evidence was what moved the score. A cross-encoder that reads both records
together, trained and scored so that the classifier sees only out-of-sample values, added 0.008
on the leaderboard.

The final lesson: two good pipelines built differently are worth more together than either tuned further. Each
finds true links the other misses. A small model trained on entities that neither pipeline had seen learned how
to weigh them, and moved the leaderboard score from 0.980 (the better single pipeline) to 0.98462.

Final submission: COMB-v1, the learned combination of pipeline A (M-v29) and pipeline B. F0.5 0.9914 on the
training entities both held out, **public leaderboard 0.98462**, 7.48 candidates per Source 1 record (the union of
both pipelines' candidate sets).

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: `src/run_pipeline.py` runs every stage (prep → block →
features → train → predict) and writes `output/matching_results.tsv` and
`output/candidate_pairs.tsv`, then runs the official validator. `src/ber/` holds the components
(`normalize`, `aliases`, `blocking`, `features`, `model`, `metrics`, `partition`, `versions`).
`src/experiments/` holds the experiments behind each version: development-sample studies, the
full-data error analysis (`full_errors.py`) and the train/test shift checks (`shift_check.py`).
The cross-encoder features come from `src/experiments/cross_encoder.py`:
- `export` writes the candidate pairs with their texts and cross-fitting halves;
- `run` fine-tunes and scores ce1 on CPU or GPU;
- `kaggle/run_ce.py` runs ce2 on two GPUs from the same module.

`run_pipeline.py` then joins the scores (FEAT-v10) and trains the matcher. The combination is
`src/experiments/combine_team.py`, which reads pipeline A's probabilities from
`src/experiments/export_scores.py`. `team_pipeline/` holds pipeline B: its source, the Kaggle notebook that ran it,
its README and pinned requirements. `README.md` gives the exact commands for all three, and `requirements.txt` the
pinned environment (Python 3.12; PyTorch and transformers only for the cross-encoders).

### B. Additional Results

Full experiment log with every version, including the full-data tables, the shift analysis and
the France study: `method_result.md`.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
