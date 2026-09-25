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

| Component | Stage | Current |
|---|---|---|
| NORM — normalisation and learned aliases | prep | NORM-v2 |
| BLK — candidate generation | block | BLK-v4b@20 |
| FEAT — pairwise features | features | FEAT-v2 |
| MATCH — models and decision rule | train, predict | MATCH-v2 |

```bash
python run_pipeline.py --list                          # every version and preset
python run_pipeline.py ... --preset M-v4               # another end-to-end version
python run_pipeline.py ... --feat FEAT-v3 --stages features,train,predict   # swap one component
```

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
