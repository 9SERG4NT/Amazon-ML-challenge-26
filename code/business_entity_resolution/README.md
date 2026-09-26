# Business Entity Resolution

End-to-end pipeline for the Amazon ML Challenge 2026: for every Source 1 record, find the
Source 2 / Source 3 records of the same business. Stages: normalise → block (candidate
generation) → pairwise features → two-stage LightGBM → per-entity decision → the two output
files.

## Setup

Python 3.12.

```bash
pip install -r requirements.txt
```

Every dependency is open source; the model is LightGBM (MIT). No external data, APIs or
lookups are used — everything is learned from the training files.

## Data

Place the challenge dataset at `resources/student_resource/dataset/` (with `train/` and `test/`).
It is not in git: the files exceed GitHub's size limit.

## Run

From `src/`:

```bash
python run_pipeline.py \
  --data ../../../resources/student_resource/dataset \
  --work ../../../work \
  --out  ../../../output \
  --validator ../../../resources/student_resource/utils/validate_submission.py
```

This writes `output/matching_results.tsv` (the scored file) and `output/candidate_pairs.tsv`
(the exact candidate set the model scored), then runs the official validator on both. A full
run peaks at about 45 GB of RAM (during blocking); on 8 cores, normalisation takes about 6
minutes and blocking about 20. `--sample 0.01` runs the same pipeline end to end on 1% of the
data in about a minute.

Stages can be run separately (`--stages prep,block,features,train,predict`); each one reads
the previous stage's files from `--work`, so an interrupted run resumes where it stopped.

## Versions

Every component is a named version in `src/ber/versions.py`, using the IDs of the
experiment log (`method_result.md`). A run is one combination of the four:

| Component | Stage | Current (preset M-v6) |
|---|---|---|
| NORM — normalisation and learned aliases | prep | NORM-v2 |
| BLK — candidate generation | block | BLK-v4b@20-tlu40 (top-20, test-like train universe) |
| FEAT — pairwise features | features | FEAT-v4 (65: number gaps, distinctive parts of names and addresses) |
| MATCH — models and decision rule | train, predict | MATCH-v6 (two-stage LightGBM, 75% of entities fitted) |

End-to-end presets (`--list` prints them all; the default stays M-v3, the first full-data run):

| Preset | Components | Status |
|---|---|---|
| M-v3 | BLK-v4b@20, FEAT-v2, MATCH-v2 | FULL-v1, leaderboard 0.96 |
| M-v5 | BLK-v4b@20-tlu40, FEAT-v3, MATCH-v6 | leaderboard 0.971 |
| **M-v6** | BLK-v4b@20-tlu40, FEAT-v4, MATCH-v6 | test-like eval 0.9856; next submission |
| M-v7 | BLK-v4b@20-dup2 (every distractor copied), FEAT-v4, MATCH-v6 | rejected: learns to spot the copies |
| M-v8 / M-v9 | M-v7 with XGBoost (MATCH-v8) / a LightGBM + XGBoost blend (MATCH-v9) | built on M-v7's features, not run |
| M-v10 | M-v6 with distractor rows weighted ×2 and the rule chosen on the doubled-distractor eval (MATCH-v10) | running |
| **M-v11** | M-v6 on the filtered candidates (BLK-v5-tlu40: 6.5 per S1 instead of 48.1) | eval 0.9854; next submission |
| M-v12 to M-v18 | on M-v11's candidates: distractors ×2 (MATCH-v10), neural network (v13), CatBoost (v15), metric-aligned loss (v17), blends (v14, v16, v18) | eval 0.9825–0.9854: none beats M-v11 |
| M-v19-us / M-v19-in | leave-one-country-out diagnostic (`train_countries`): fit one country, score the other as unseen | India unseen 0.9444, US unseen 0.9637 |
| M-v20-us / M-v20 | self-training on the unseen country (`self_train`), checked on India, applied to France | +0.0014 on India |
| M-v21 | FEAT-v5: house numbers kept in the distinctive parts + `addr_twin_num_diff` | eval 0.9854, same as M-v11 |

```bash
python run_pipeline.py --list                          # every version and preset
python run_pipeline.py ... --preset M-v6               # an end-to-end version
python run_pipeline.py ... --preset M-v6 --match MATCH-v10 --stages train,predict   # swap one component, reuse the rest
```

BLK-v5-tlu40 adds a **learned candidate filter** after the search: a small LightGBM on the blocking scores and
their competition context (no string similarity) keeps the candidates above the threshold that retains 99.5% of
the true links the search found for fit entities (out-of-fold). Features and the matcher then see only those, so
`candidate_pairs.tsv` is exactly the scored set. `search_from` reuses a cached search with the same settings.

MATCH versions can also switch the model family (`algo="xgb"`, XGBoost, Apache-2.0), blend two
runs on the same features (`blend_of`), weight distractor rows (`distractor_weight`) and score an
old run's models in a new universe (`frozen_from`).

Each stage caches its output under the versions it depends on (for example
`work/block/NORM-v2__BLK-v4b@20/`), so swapping the matcher reuses the blocking, and a new
blocking version never reuses stale features. `work/runs/<versions>/metrics.json` records the
versions, the data scope and every stage's results. A registered version is never edited;
a change is a new version.

## Validation

Train S1 entities are split once (seed 42): fit 30%, early stopping 5%, **evaluation 20%**,
rest 45%. All entities stay in the candidate search, so targets are contested as densely as
at test time. Aliases and region merges are learned without the evaluation entities' links,
and the models are cross-fitted, so evaluation rows are scored exactly like test rows. The
decision rule and its parameter are chosen on the evaluation slice.

The test differs from train in its distractors (records that match nothing), so validation
rebuilds that part of it:

- **Test-like universe** (`keep_nonevals`, BLK-v4b@20-tlu40): every eval S1 stays, 40% of the
  other train S1 entities stay with their true targets, and every distractor stays. That gives
  2.34 distractors per S1, as in the test (train: 1.2).
- **Doubled-distractor eval** (reported for every run; MATCH-v10 chooses its rule on it): each
  link to a distractor counts twice. The test's distractors copy present S1 records (same name,
  same street, a nearby house number) as often as train's, so it has twice the hard lookalikes
  per S1, which the test-like universe does not recreate.
- **Test profile check:** each run logs the share of test S1 linked and the links per S1 by
  country, to compare with the eval truth (~94% linked, ~3.4 links). M-v7 passed its eval but
  failed this check (98.5%, 4.2).

## Layout

```
src/
  run_pipeline.py      the pipeline (stages above)
  sagemaker_entry.py   entry point when the pipeline runs as a SageMaker training job
  ber/                 normalize, aliases, blocking, features, model, metrics, data, versions
  experiments/         DEV-10 experiments behind the versions (see method_result.md)
```

`infra/aws/` (repository root) holds the AWS setup: SageMaker training-job launcher, EC2
runner bootstrap, quota and bucket scripts.
