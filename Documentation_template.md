# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary

We match records in three stages: an IDF-weighted sparse search generates candidates, a
two-stage LightGBM classifier scores each (Source 1, candidate) pair, and a per-entity decision
turns probabilities into the match list that maximises macro F0.5. Two ideas carry most of the
accuracy: **competition features** — every Source 2/3 record belongs to at most one Source 1
entity, so a candidate is judged against the rival Source 1 records that want the same record —
and a **validation protocol that keeps the full competition density**, which showed that a 10%
development sample overstates blocking recall by more than a point.

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

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage classifier + per-entity set decision  
**Core Innovation:** competition-aware scoring (rank and probability margin against rival Source 1
records, then exclusive assignment) and full-density validation.

Every component is a named version (normalisation NORM, blocking BLK, features FEAT, matcher
MATCH) recorded in `src/ber/versions.py`; the submitted run is preset **[pending: M-v3 or
later]**.

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
- **Candidate pairs generated:** 83.3M for the test set (48.1 per Source 1 record) with
  BLK-v4b@20 **[pending: final BLK version]**.
- **How we ensured true matches were not lost:** recall is measured on held-out training entities
  at full density: 98.32% of true links with BLK-v4b@20 (88.7% for targets without an address,
  98.8% for the rest). Recall-versus-depth curves for both passes decide the cut-offs
  **[pending: BLK-v5 measurement]**.

---

## 4. Matching Model

**Features used** (52 in stage 1):
- **Name features:** rapidfuzz ratio, token-set, token-sort and partial ratios, and Jaro-Winkler
  on the full, core (legal words removed) and joined names; best match against "fka / dba" parts;
  token containment both ways; **soft word alignment** (each core-name token aligned to its best
  Jaro-Winkler match, both directions: mean, worst, number of unaligned tokens and the highest IDF
  among them — separating typos from a different business).
- **Address features:** ratio, token-set, token-sort and partial ratios; token containment;
  house-number agreement (Jaccard, share of target numbers present, truncated-number match).
- **Competition and context features:** blocking score and name/address cosines; rank of the
  candidate within its Source 1 record (per source and overall); gap to the best candidate;
  number of candidates; **number of Source 1 records that retrieved the target, this record's rank
  among them, and its score margin over the best rival**; how many Source 1 records share the name.
- **Stage 2** adds the stage-1 probability and its context: rank among the record's candidates and
  among the target's Source 1 records, best rival probability and the margins over it.

**Model type:** LightGBM (MIT), binary log loss, 255 leaves, learning rate 0.05, early stopping.
Stage 1 and stage 2 are both **cross-fitted** over 3 folds of the fit entities, so every
probability used downstream is out-of-sample.

**Threshold selection method:** each target is kept only for the Source 1 entity that scores it
highest (exclusive assignment); then a rule chosen on held-out training entities by macro F0.5:
a single threshold, expected-F (per entity, the top-k maximising expected F0.5 under the
predicted probabilities, or no match), or gated expected-F.

**Validation:** Source 1 training entities are split once — fit 30%, early stopping 5%,
evaluation 20% (441,521 entities), rest 45%. All entities stay in the candidate search, aliases
and region merges are learned without evaluation links, and evaluation rows are scored exactly
like test rows.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **[pending: full-data evaluation slice]**. Development sample (10%,
  44,137 entities): 0.9906 (precision 0.9963, recall 0.9794) with stage 2 and expected-F,
  against 0.9882 for the first single-stage model.
- **Common false positives (wrong merges):** the same business name at a *nearby* house number
  (831 vs 835 Winsor Pl; 3-4-114/12 vs /13) — planted distractors that differ from true matches
  only in the number; names that differ in one distinctive word.
- **Common false negatives (missed matches):** records without an address (about half of the
  recoverable misses on the development sample: with only a name, a same-name Source 1 entity
  often claims the record); invented trade names that match only through the address; blocking
  misses in dense regions.

| Version (development sample) | F0.5 | Precision | Recall |
|---|---:|---:|---:|
| Single model, 44 features | 0.9882 | 0.9954 | 0.9745 |
| + soft word alignment (52) | 0.9894 | 0.9959 | 0.9772 |
| + stage 2, expected-F | 0.9906 | 0.9963 | 0.9794 |

---

## 6. Conclusion

Treating the problem as competition for records — rather than independent pair classification —
and validating at the real competition density were the decisive choices. The main lesson: a
convenient 10% sample hid a 1.1-point blocking-recall loss that only full-density validation
exposed. **[pending: final score and what the deeper blocking recovered]**

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: `src/run_pipeline.py` runs every stage (prep → block →
features → train → predict) and writes `output/matching_results.tsv` and
`output/candidate_pairs.tsv`, then runs the official validator. `src/ber/` holds the components
(`normalize`, `aliases`, `blocking`, `features`, `model`, `metrics`, `versions`); `src/experiments/`
holds the development-sample experiments behind each version. `README.md` gives the exact command
and `requirements.txt` the pinned environment (Python 3.12).

### B. Additional Results

Full experiment log with every version: `method_result.md`. **[pending: full-data tables]**

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
