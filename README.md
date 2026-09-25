# Amazon ML Challenge 2026 — Business Entity Resolution

For every business record in Source 1, find the records of the same business in Sources 2
and 3 — no shared IDs, noisy names and addresses, three countries (France only in test).
Scored by macro F0.5 per Source 1 entity. Challenge statement:
[`resources/student_resource/README.md`](resources/student_resource/README.md).

## Approach

```
normalise ──> block ──> features ──> LightGBM stage 1 ──> stage 2 ──> exclusive assignment + decision rule
(aliases)     (sparse     (string,     (cross-fitted)       (rival        (each S2/S3 record to at most
              top-k       numbers,                          probability   one S1; threshold or
              search)     competition)                      context)      expected F0.5 per entity)
```

- **Normalisation** canonicalises legal forms, street types and states, transliterates
  Devanagari, and learns per-country token aliases from the training links.
- **Blocking** runs an IDF-weighted sparse top-k search on name/address tokens, typo-tolerant
  variants and name 3-grams, split by region (state), with a name-only pass for records
  without an address.
- **The matcher** is a two-stage LightGBM (MIT licence). Its strongest signals are
  *competition* features: how a candidate compares with the rival Source 1 records that want
  the same record.
- **Validation** holds out 20% of the training entities at the full competition density.
  A 10% sample overstated blocking recall by more than a point.

## Results so far

| Version | Data | F0.5 | Precision | Recall | Public leaderboard |
|---|---|---:|---:|---:|---:|
| M-v1: single model, 44 features | 10% dev sample | 0.9882 | 0.9954 | 0.9745 | |
| M-v2: + soft word alignment | 10% dev sample | 0.9894 | 0.9959 | 0.9772 | |
| M-v3: + stage 2, expected-F rule | 10% dev sample | 0.9906 | 0.9963 | 0.9794 | |
| FULL-v1 (M-v3) | full data, 441,521 eval S1 | **0.9813** | 0.9952 | 0.9544 | **0.96** |

The leaderboard disagrees with the full-data eval slice because the test set has **twice as
many distractors per S1** as train (about 2.3 against 1.2, derived from the per-source record
counts). The next run trains and validates in a test-like universe built from the train data
(`BLK-v4b@20-tlu40`), with number-gap features (FEAT-v3). Every experiment, with methods and
numbers: [`method_result.md`](method_result.md).

## Repository

| Path | Contents |
|---|---|
| [`code/business_entity_resolution/`](code/business_entity_resolution/) | the pipeline — see its [README](code/business_entity_resolution/README.md) for how to run it |
| [`method_result.md`](method_result.md) | experiment log: every version, how it was measured, what it scored |
| [`Documentation_template.md`](Documentation_template.md) | methodology write-up for the submission (draft; final numbers pending) |
| [`infra/aws/`](infra/aws/) | AWS setup: SageMaker training-job launcher, EC2 runner, quotas — see its [README](infra/aws/README.md) |
| [`make_submission.py`](make_submission.py) | validates the outputs and builds `<team>_submission.zip` |
| [`CLAUDE.md`](CLAUDE.md) | context for Claude Code sessions working in this repository |
| `resources/` | challenge statement, validator; the dataset itself is not in git (too large) |
| `output/` | `matching_results.tsv` and `candidate_pairs.tsv` of the chosen run |

## Quick start

Python 3.12, with the dataset placed at `resources/student_resource/dataset/`:

```bash
pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution/src
python run_pipeline.py --list          # the registered versions of every component
python run_pipeline.py --data ../../../resources/student_resource/dataset \
  --work ../../../work --out ../../../output --sample 0.01    # 1% smoke run, about a minute
```

A full run needs about 45 GB of RAM, so it runs on AWS (SageMaker training job or the EC2
runner, see [`infra/aws/README.md`](infra/aws/README.md)). Every component is a named version
(NORM, BLK, FEAT, MATCH) in `src/ber/versions.py`, so any logged result can be re-run by name.
