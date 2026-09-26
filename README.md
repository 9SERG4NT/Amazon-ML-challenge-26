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
- **Validation** holds out 20% of the training entities (the same 441,521 in every run). It is
  scored in a *test-like universe*: train rebuilt with the test's distractor density, because the
  test has twice train's distractors per S1. Every run also reports a *doubled-distractor* eval
  that counts each link to a distractor twice, the test's density of same-name lookalikes.

## Results so far

| Version | Eval data | F0.5 | Precision | Recall | Public leaderboard |
|---|---|---:|---:|---:|---:|
| M-v1: single model, 44 features | 10% dev sample | 0.9882 | 0.9954 | 0.9745 | |
| M-v2: + soft word alignment | 10% dev sample | 0.9894 | 0.9959 | 0.9772 | |
| M-v3: + stage 2, expected-F rule | 10% dev sample | 0.9906 | 0.9963 | 0.9794 | |
| FULL-v1 (M-v3) | full data | 0.9813 | 0.9952 | 0.9544 | 0.96 |
| M-v5: test-like universe, number-gap features, 75% of entities fitted | test-like | 0.9852 | 0.9957 | 0.9644 | **0.971** |
| M-v6: M-v5 + France-robust features (FEAT-v4) | test-like | **0.9856** | 0.9960 | 0.9647 | pending |
| M-v7: every distractor copied | copies | 0.9856 | 0.9978 | 0.9599 | rejected |
| **M-v11: M-v6 behind a learned candidate filter (6.5 candidates per S1 instead of 48)** | test-like | 0.9854 | 0.9962 | 0.9639 | **0.970** (submitted) |
| M-v12 to M-v18: distractor weights ×2, a metric-aligned loss, a neural network (Adam), CatBoost, blends (all on M-v11's candidates) | test-like | 0.9825–0.9854 | | | none beats M-v11 |

Rows with different eval data are not comparable. What the leaderboard taught us:

1. **FULL-v1 (0.96):** the test has **twice as many distractors per S1** as train (about 2.3
   against 1.2, derived from the per-source record counts). M-v5 trains and validates in a
   test-like universe with that density and scored 0.971.
2. **The test's distractors are lookalikes of S1 records that are present** (same name, same
   street, nearby house number) as often as train's. So the test has twice the hard lookalikes
   per S1. The test-like universe only doubled easy, orphaned ones, so its eval (0.985) is still
   optimistic in every country.
3. **M-v7 copied every distractor to double the lookalikes.** The model learned to recognise the
   exact copies and over-linked the test (98.5% of S1 linked against ~94%), so it was not
   submitted. M-v10 gets the same density from training weights, with no copies.

Every experiment, with methods and numbers: [`method_result.md`](method_result.md). The
submitted files and every run's outputs are listed in [`output/README.md`](output/README.md).

## Repository

| Path | Contents |
|---|---|
| [`code/business_entity_resolution/`](code/business_entity_resolution/) | the pipeline — see its [README](code/business_entity_resolution/README.md) for how to run it |
| [`method_result.md`](method_result.md) | experiment log: every version, how it was measured, what it scored |
| [`Documentation_template.md`](Documentation_template.md) | methodology write-up for the submission (draft; final numbers pending) |
| [`infra/aws/`](infra/aws/) | AWS setup: SageMaker training-job launcher, EC2 runner (`ec2/launch_runner.sh` creates it in a fresh account), quotas — see its [README](infra/aws/README.md) |
| [`infra/kaggle/`](infra/kaggle/) | Kaggle notebooks that run a preset on a private copy of the dataset (TPU VM for its RAM; CPU for 1% smoke tests) |
| [`make_submission.py`](make_submission.py) | validates the outputs and builds `<team>_submission.zip` |
| [`CLAUDE.md`](CLAUDE.md) | context for Claude Code sessions working in this repository |
| `resources/` | challenge statement, validator; the dataset itself is not in git (too large) |
| `output/` | `matching_results.tsv` and `candidate_pairs.tsv` of the submitted run; `runs/<version>/` for every complete run; [`output/README.md`](output/README.md) is the manifest (eval, leaderboard, md5) |

## Quick start

Python 3.12, with the dataset placed at `resources/student_resource/dataset/`:

```bash
pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution/src
python run_pipeline.py --list          # the registered versions of every component
python run_pipeline.py --data ../../../resources/student_resource/dataset \
  --work ../../../work --out ../../../output --sample 0.01    # 1% smoke run, about a minute
```

A full run needs about 45–60 GB of RAM and 2–3.5 hours on 8 cores, so it runs on an EC2
runner (r7a.2xlarge, see [`infra/aws/README.md`](infra/aws/README.md)) or a Kaggle TPU VM
([`infra/kaggle/`](infra/kaggle/)). Every component is a named version
(NORM, BLK, FEAT, MATCH) in `src/ber/versions.py`, so any logged result can be re-run by name.
