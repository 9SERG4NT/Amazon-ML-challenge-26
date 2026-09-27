# Business Entity Resolution

End-to-end pipeline for the Amazon ML Challenge 2026: for every Source 1 record, find the
Source 2 / Source 3 records of the same business. The final submission, **COMB-v2**, combines the
team's two pipelines (COMB-v1, the same combination without the pre-filter, scored **0.98462** on the
public leaderboard):

- **Pipeline A (`src/`, preset M-v29):** normalise → sparse IDF search + learned candidate filter →
  pairwise features, including three fine-tuned cross-encoders' pair scores → two-stage LightGBM
  (test-like evaluation F0.5 0.9887; M-v28, its predecessor, scored 0.978 on the leaderboard).
- **Pipeline B (`team_pipeline/`):** a fine-tuned multilingual-e5-small bi-encoder with FAISS
  search and a graph filter → XGBoost with a cross-encoder score (the teammates' pipeline; 0.980 on
  the leaderboard).
- **Combination (`src/experiments/combine_team.py`):** a LightGBM over the union of both candidate
  sets reads each pipeline's probability and competition context. It is trained on the training
  entities that both pipelines held out, where it scores 0.9914 against 0.9885 (A) and 0.9877 (B).
  French probabilities get a label-free calibration. A pre-filter first drops union pairs that both
  pipelines score below 0.1: 3.68 candidates per test record instead of 7.48, at the same held-out F0.5.

## Setup

Python 3.12.

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu   # CPU build; any CUDA build for the GPU step
pip install -r requirements.txt
```

Every dependency is open source. Pipeline A's models are LightGBM (MIT) and three cross-encoders
fine-tuned from `google/bert_uncased_L-4_H-256_A-4` (Apache-2.0, 11M parameters),
`intfloat/multilingual-e5-small` (MIT, 118M) and `intfloat/multilingual-e5-base` (MIT, 278M).
Pipeline B's are a bi-encoder and a cross-encoder fine-tuned from `intfloat/multilingual-e5-small`
and XGBoost (Apache-2.0); its pinned environment is `team_pipeline/requirements.txt`. The base models
are downloaded from the Hugging Face hub on first use. Every model is far below 8B parameters. No
external data, APIs or lookups are used: everything is learned from the training files.

## Data

Place the challenge dataset at `resources/student_resource/dataset/` (with `train/` and `test/`).
It is not in git: the files exceed GitHub's size limit.

## Run

### The final submission (COMB-v2)

From `src/`, with `D=<dataset dir with train/ and test/>`, `W=<work dir>` and `R=<dir for the team run>`:

```bash
# A. pipeline A, preset M-v29: steps 1-4 below, then ce3 and the matcher on FEAT-v10
CE_CONFIG='{"model": "intfloat/multilingual-e5-base", "lr": 3e-5, "score_batch": 512, "name": "ce3"}' \
  python ../kaggle/run_ce.py                                  # ~3 h on 2x T4; copy ce3_*.parquet to $W/extra/NORM-v2__BLK-v5-tlu40/
python run_pipeline.py --data $D --work $W --out $W/out_m29 --preset M-v29 --stages features,train,predict
python -m experiments.export_scores $W NORM-v2__BLK-v5-tlu40__FEAT-v10__MATCH-v6 $W/m29   # -> m29_{train,test}.parquet

# B. pipeline B (from team_pipeline/, GPU, ~4.7 h on Kaggle's 2x T4); it saves its validation and test pair scores
cd ../team_pipeline
python -m src.stage1_blocking --data-dir $D --out-dir $R/stage1
python -m src.stage2_match --stage1-dir $R/stage1 --data-dir $D --out-dir $R/stage2 --team-name Team_Sumukh \
  --margins --cross-encoder --fit-max-pairs 4000000 --ce-pairs 2000000 --ce-minutes 45
cd ../src

# C. the combination (8 cores, ~2.5 min, 9 GB): writes matching_results.tsv, candidate_pairs.tsv, combine_validation.json
python -m experiments.combine_team --team $R/stage2 --ours-train $W/m29_train.parquet --ours-test $W/m29_test.parquet \
  --truth $D/train/train_ground_truth.tsv --s1 $D/test/test_source1.tsv --out ../../../output --threads 8 --france-shift auto \
  --prefilter 0.1        # without --prefilter: COMB-v1
```

The combination is deterministic: two runs give byte-identical files. The team notebook that ran
pipeline B on Kaggle is `team_pipeline/team_pipeline.ipynb`; it writes the same source files and
runs the two commands above.

### Pipeline A alone (M-v28, four steps)

```bash
# 1. normalise, candidate search + learned filter, base features (65)
python run_pipeline.py --data $D --work $W --out ../../../output --preset M-v11 --stages prep,block,features

# 2. the candidate pairs with their texts and cross-fitting halves, then ce1 (4-layer BERT) on the CPU
#    (about 3 h on 8 cores; bf16 is faster on CPUs that support it)
python -m experiments.cross_encoder export $W NORM-v2__BLK-v5-tlu40
python -m experiments.cross_encoder run $W NORM-v2__BLK-v5-tlu40 --name ce1 --threads 8 --half

# 3. ce2 (multilingual-e5-small, raw texts) on two GPUs (about 1.5 h on 2x T4): ../kaggle/run_ce.py reads the
#    challenge TSVs and $W/ce/NORM-v2__BLK-v5-tlu40/ids_{train,test}.parquet (set CE_INPUT / CE_TMP / CE_OUT
#    outside Kaggle) and writes ce2_{train,test}.parquet; copy them to $W/extra/NORM-v2__BLK-v5-tlu40/
python ../kaggle/run_ce.py

# 4. join both scores to the base features (FEAT-v9), train, predict, validate
python run_pipeline.py --data $D --work $W --out ../../../output --preset M-v28 --stages features,train,predict \
  --validator ../../../resources/student_resource/utils/validate_submission.py
```

This writes `output/matching_results.tsv` (the scored file) and `output/candidate_pairs.tsv`
(the exact candidate set the model scored), then runs the official validator on both. A full
run peaks at about 45 GB of RAM (during blocking); on 8 cores, normalisation takes about 6
minutes, blocking about 20, and step 4 about 7. `--sample 0.01` runs steps 1 and 4 (with
`--preset M-v11`) end to end on 1% of the data in about a minute.

Stages can be run separately (`--stages prep,block,features,train,predict`); each one reads
the previous stage's files from `--work`, so an interrupted run resumes where it stopped.

## Versions

Every component is a named version in `src/ber/versions.py`, using the IDs of the
experiment log (`method_result.md`). A run is one combination of the four:

| Component | Stage | Pipeline A in the final submission (preset M-v29) |
|---|---|---|
| NORM — normalisation and learned aliases | prep | NORM-v2 |
| BLK — candidate generation | block | BLK-v5-tlu40 (top-20 search in the test-like train universe + learned filter: 6.5 per S1) |
| FEAT — pairwise features | features | FEAT-v10 (83: FEAT-v4's 65 + three cross-encoder scores and their competition context) |
| MATCH — models and decision rule | train, predict | MATCH-v6 (two-stage LightGBM, 75% of entities fitted) |

End-to-end presets (`--list` prints them all; the default stays M-v3, the first full-data run):

| Preset | Components | Status |
|---|---|---|
| M-v3 | BLK-v4b@20, FEAT-v2, MATCH-v2 | FULL-v1, leaderboard 0.96 |
| M-v5 | BLK-v4b@20-tlu40, FEAT-v3, MATCH-v6 | leaderboard 0.971 |
| M-v6 | BLK-v4b@20-tlu40, FEAT-v4, MATCH-v6 | test-like eval 0.9856 |
| M-v7 | BLK-v4b@20-dup2 (every distractor copied), FEAT-v4, MATCH-v6 | rejected: learns to spot the copies |
| M-v8 / M-v9 | M-v7 with XGBoost (MATCH-v8) / a LightGBM + XGBoost blend (MATCH-v9) | built on M-v7's features, not run |
| M-v10 | M-v6 with distractor rows weighted ×2 and the rule chosen on the doubled-distractor eval (MATCH-v10) | running |
| M-v11 | M-v6 on the filtered candidates (BLK-v5-tlu40: 6.5 per S1 instead of 48.1) | eval 0.9854, leaderboard 0.970 |
| M-v12 to M-v18 | on M-v11's candidates: distractors ×2 (MATCH-v10), neural network (v13), CatBoost (v15), metric-aligned loss (v17), blends (v14, v16, v18) | eval 0.9825–0.9854: none beats M-v11 |
| M-v19-us / M-v19-in | leave-one-country-out diagnostic (`train_countries`): fit one country, score the other as unseen | India unseen 0.9444, US unseen 0.9637 |
| M-v20-us / M-v20 | self-training on the unseen country (`self_train`), checked on India, applied to France | +0.0014 on India |
| M-v21 | FEAT-v5: house numbers kept in the distinctive parts + `addr_twin_num_diff` | eval 0.9854, same as M-v11 |
| M-v25 | M-v11 + ce1, a 4-layer BERT cross-encoder's pair score (FEAT-v6) | eval 0.9875 |
| M-v26 / M-v27 | M-v11 + ce2, multilingual-e5-small (FEAT-v7); + its competition context (FEAT-v8) | eval 0.9884 / 0.9884 |
| M-v28 | M-v11 + ce1 + ce2 + their competition context (FEAT-v9) | eval 0.9885, leaderboard 0.978 |
| **M-v29** | M-v28 + ce3, multilingual-e5-base (FEAT-v10) | **eval 0.9887; pipeline A of the final submission** |

The final submission, **COMB-v2**, is not a preset: `experiments/combine_team.py` combines M-v29 with pipeline B
(shared held-out F0.5 0.9914; COMB-v1, without the pre-filter, **public leaderboard 0.98462**; see "The final
submission" above).

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
  experiments/         experiments behind the versions (see method_result.md); cross_encoder.py builds the
                       cross-encoder features (export, CPU/GPU training and cross-fitted scoring)
  experiments/combine_team.py   the final combination of pipelines A and B (union of candidates, cross-fitted LightGBM,
                       French calibration); experiments/export_scores.py exports a run's probabilities for it
kaggle/
  run_ce.py            ce2 / ce3 on two GPUs (one cross-fitting half per GPU), using experiments/cross_encoder.py
team_pipeline/         pipeline B (the teammates' bi-encoder + XGBoost pipeline, run on Kaggle): src/, its notebook,
                       README and pinned requirements. It ships in the submission zip; it is not in the public repository
```

`infra/aws/` (repository root) holds the AWS setup: SageMaker training-job launcher, EC2
runner bootstrap, quota and bucket scripts.
