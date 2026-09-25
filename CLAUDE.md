# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Task

Amazon ML Challenge 2026, Business Entity Resolution: for every Source 1 (S1) record, list
the Source 2/3 records of the same business. Scored by **macro F0.5 per S1 entity**
(precision counts double; an S1 with no true match scores 1.0 only if its list is empty).
Full rules: `resources/student_resource/README.md`. Hard constraints that shape every choice:

- Final model must be MIT/Apache-2.0 and at most 8B parameters (we use LightGBM, MIT).
- No external data, lookups, APIs or geocoding — everything is learned from the train files.
- `country` is an open set: test has **France**, which never appears in train. Never hard-code
  `{US, India}`; per-country learned artefacts (aliases, region merges) simply don't exist for France.
- Deliverables: `output/matching_results.tsv` (scored), `output/candidate_pairs.tsv` (the exact set
  the model scored; every match must be in it), runnable `code/business_entity_resolution/`,
  filled `Documentation_template.md`.

`method_result.md` is the experiment log and the source for the final write-up. Add every
run there (DEV-10 and full-data), keep its summary tables in sync.

## Commands

Run from `code/business_entity_resolution/src` (Python 3.12, `pip install -r ../requirements.txt`):

```bash
# full pipeline: data -> normalise -> block -> features -> 2-stage LightGBM -> decision -> TSVs
python run_pipeline.py --data <dataset_dir> --work <work_dir> --out <out_dir> \
  --validator ../../../resources/student_resource/utils/validate_submission.py   # preset M-v3
python run_pipeline.py --list                       # every registered version and preset
python run_pipeline.py ... --preset M-v4            # another end-to-end version
python run_pipeline.py ... --feat FEAT-v3 --stages features,train,predict   # swap one component
python run_pipeline.py ... --sample 0.01            # end-to-end smoke run, ~1 min, ~6 GB RAM

# DEV-10 experiments (cache dir holds dev_Q/dev_T/dev_pairs/dev_features_v2/dev_part/...)
python -m experiments.dev_stack <cache>                              # stage 1 vs stage 2
python -m experiments.dev_decide <cache> dev_scores_v3.parquet p2    # decision rules on saved scores
python -m experiments.dev_errors <cache> dev_scores_v3.parquet p2 0.725
python -m experiments.dev_v3 <cache>                                 # FEAT-v3 number gap + support features
```

Final package: copy the chosen run's TSVs from `s3://…/predictions/<version>/` into `output/`,
then `python make_submission.py --team "<name>"` (repo root) validates them and writes
`<team>_submission.zip` in the required layout.

There is no test suite or linter. The format check is the official validator
(`resources/student_resource/utils/validate_submission.py`, stdlib only); `run_pipeline.py`
runs it automatically when `--validator` is given and `--sample` is 1.

The dataset (`resources/student_resource/dataset/`, 2.4 GB) is not in git. The laptop has 16 GB
RAM with little free; full-data runs (~100M train candidate pairs) need ~40 GB and run on AWS.

## Pipeline architecture

**Versions.** Every component is a named, frozen version in `ber/versions.py` — NORM
(normalisation + aliases), BLK (blocking), FEAT (features), MATCH (models + decision rules) —
using the IDs of `method_result.md`. End-to-end versions M-v1…M-v4 are named combinations
(`PRESETS`). Never edit a registered version: add a new one and log it. `run_pipeline.py` takes
`--preset` plus optional `--norm/--block/--feat/--match` overrides; there are no ad-hoc tuning flags.

Each stage caches under the versions it depends on, so swapping the matcher reuses the
(expensive) blocking and a new blocking version can never reuse stale features:
`<work>/prep/NORM-v2/`, `<work>/block/NORM-v2__BLK-v4b@20/`, `<work>/feat/…__FEAT-v2/{train,test}/part-*.parquet`,
`<work>/runs/…__MATCH-v2/` (models, scores, `metrics.json` gathering every upstream stage's `<stage>.json`).
One work dir = one data scope (`--sample/--seed/--fit/--es/--eval`); prep refuses reuse under another.

1. **prep** — `ber/normalize.py` (transliteration via anyascii, legal-form/street/state
   canonicalisation, fka/dba split, website names) + `ber/aliases.py` (per-country token aliases
   learned from training links, e.g. Devanagari transliterations `praivet -> pvt`, `tn -> tamilnadu`).
2. **block** — `ber/blocking.py`: hashed sparse features (name/address tokens, 4-char prefixes,
   consonant skeletons, house-number prefixes/suffixes, char 3-grams of the joined name), IDF per
   (country, source), score = cos(name) + cos(address), top-k per source with `sparse_dot_topn`.
   Search is split by **region** (states = frequent last address tokens; merges learned from links
   that cross them, e.g. telangana↔andhrapradesh, delhi↔haryana). A second name-only pass (top 5)
   covers targets with no address. Run one country at a time to bound memory.
3. **features** — `ber/features.py`: rapidfuzz similarities, house-number agreement, soft word
   alignment (Monge-Elkan / Jaro-Winkler with IDF), and **competition features** (rank of the
   candidate for its S1, rank of the S1 for the target, margin over the best *other* S1).
   Context features need all candidates at once; the rest are computed in parts.
4. **train** — `ber/model.py`: LightGBM binary log loss (calibrated probabilities for the set
   decision). Stage 1 is cross-fitted; stage 2 adds `probability_context` (stage-1 probability
   of rival candidates / rival S1 records) and is cross-fitted too.
5. **predict** — **exclusive assignment** (each S2/S3 record goes only to the S1 that scores it
   highest — ground truth never links a record to two S1s), then the rule chosen on eval
   entities: threshold, expected-F (`decide_expected_f`) or gated expected-F (`decide_gated`).

Validation design (in `stage_prep`): one seed-42 split of train S1 into fit 30% / early-stop 5% /
eval 20% / rest 45%. Every entity stays in blocking so targets are contested as densely as at
test time; aliases and region merges are learned without eval links; fit rows get out-of-fold
predictions and all other rows (eval, rest, test) the mean of the fold models, so eval rows are
scored exactly like test rows.

Gotchas:
- `probability_context` returns `p1` as its first column; appending `p1` again duplicates it.
- `nums` lists come from `unique()` without order, so "first number" is not the house number.
- DEV-10 (10% of S1, their matches, 10% of distractors) has ~10× less competition per region
  than full data: its scores are optimistic. Compare variants on DEV-10, report on full data.

## Results and decisions (DEV-10 eval slice, 44,137 S1)

| Version | F0.5 | P | R | Singletons | Others |
|---|---:|---:|---:|---:|---:|
| M-v1: 44 features, threshold 0.725 | 0.9882 | 0.9954 | 0.9745 | 0.9805 | 0.9887 |
| M-v2: + alignment features (52), threshold 0.725 | 0.9894 | 0.9959 | 0.9772 | 0.9826 | 0.9899 |
| Stage 2 (probability context), threshold 0.725 | 0.9904 | 0.9970 | 0.9789 | 0.9935 | 0.9902 |
| Stage 2, expected-F | **0.9906** | 0.9963 | 0.9794 | 0.9814 | 0.9911 |
| Stage 2, gated expected-F (gate 0.55) | 0.9906 | 0.9966 | 0.9792 | 0.9858 | 0.9909 |

Blocking (BLK-v4b, top-20/source): 99.45% pair recall; a perfect matcher on these candidates
would score 0.9984. Stage 2's gain comes from `p1` and `p1_margin_t` (margin over the best other
S1 for the target). Threshold rules protect singletons, expected-F helps multi-match entities;
gating did not beat expected-F on DEV-10, so the full run picks the rule on its eval slice.

Error analysis (`dev_errors`, stage 2 at 0.725): of 3,187 missed links, 26% were never
candidates, 29% were lost to another S1 under exclusivity (95% of those are targets with **no
address**), 45% scored below threshold (46% no address). False positives are mostly the same
name at a **nearby house number** (831 vs 835 Winsor Pl) — the data plants such distractors,
while true matches also carry number typos (189 vs 889). Some true matches carry an invented
trade name ("Korevo" for "Asahi Charitable Trust") and can only link through the address or
the entity's other records. Candidate next features, registered but not yet validated:
number-gap features (FEAT-v3: digit edit distance and numeric gap of the closest unmatched
numbers) and near-duplicate "support" from the entity's confident candidates (MATCH-v3); both
in `ber/features.py`, tested by `experiments/dev_v3.py`, combined in preset M-v4.

**Full data changes the picture (FULL-v1, eval slice):** BLK-v4b@20 finds only **98.32%** of
true links (DEV-10: 99.45%) because regions hold ~10x more rival records. Targets without an
address: 88.7% (the name-only pass keeps just 5). Recall by main-pass depth: top-5 96.0%,
top-10 97.4%, top-20 98.3% and still rising. Blocking depth is therefore the biggest lever;
BLK-v4b@40n20 measures both depth curves (`recall_at_k`, `noaddr_recall_at_k` in the block stage).

## AWS

- **Account 125650147728, region us-east-1.** A teammate's notes describe a different account
  (911797456769); its bucket is not accessible from here.
- **Auth:** `aws login` (short-lived credentials, root user; root has no access keys; enabling root
  MFA is recommended). Local boto3 needs `pip install "botocore[crt]"` for the login provider. In Claude Code,
  use the aws-mcp tools (its `run_script` sandbox blocks the `base64` module).
- **Budget:** `mlc26-monthly-50usd` ($50/month, email alerts at 50% and 100%).
- **S3:** `s3://amazon-ml-challenge-26-sagemaker-125650147728`. The user wants it to hold **data and
  results only**: `dataset/` (the 7 TSVs), `experiments/` (logs, metrics), `predictions/<version>/`,
  `models/<version>/` (fold models, not the multi-GB score files). Code bundles for EC2/SageMaker go
  under `code/`, which a lifecycle rule deletes after 7 days. AES256, public access blocked,
  TLS-only policy; versioning off, so deletes are permanent.
- **SageMaker:** domain `d-tgxrzl4mojbh` (us-east-1), profile `Sumukh`, space `test`
  (JupyterLab on ml.t3.medium, 4 GB RAM / 5 GB disk — too small for this pipeline; no idle
  shutdown configured). Execution role `AmazonSageMaker-ExecutionRole-20260925T163023` can read
  and write the bucket. An older, empty domain `d-d4wvkyftnhlu` exists in eu-north-1.
- **Quotas (us-east-1):** approved — Studio user profiles 2, domains 2, running Studio apps 40.
  Pending AWS review — training on ml.m5.4xlarge / ml.m5.xlarge / ml.g5.2xlarge, processing on
  ml.g5.xlarge, Studio JupyterLab on ml.r5.4xlarge / ml.g5.xlarge. Every other training and
  processing quota is 0; only ml.t3.medium Studio/notebook compute works. EC2 standard vCPU quota: 8.
- **SageMaker training path (ready, blocked on quota):** `infra/aws/sagemaker/launch_training_job.py`
  packages `src/` + pinned requirements + validator and runs `src/sagemaker_entry.py` on
  `pytorch-training:2.7.1-cpu-py312-ubuntu22.04-sagemaker`; working files in `/tmp`, outputs in
  `/opt/ml/output/data` and `/opt/ml/model`. It checks the instance quota before launching.
- **EC2 runner (current compute):** `i-033e809bc1d8c21b5`, r7a.2xlarge (8 cores, 64 GB, ~$0.61/h),
  Amazon Linux 2023, 200 GB gp3, role/profile `mlc26-ec2-runner` (bucket read/write + SSM),
  security group `sg-02c6cc9396979636b` (no inbound). Driven by SSM Run Command, no SSH. Layout:
  `/opt/mlc26/venv` (Python 3.12, pinned deps), `/opt/mlc26/pipe*/src` (code versions),
  `/opt/mlc26/runs/full` (the full-data work dir, version-keyed), `/opt/mlc26/logs/`, data in
  `/data/dataset`. Long jobs run as `nohup` chains (`/opt/mlc26/chain*.sh`) that wait on marker
  files in `logs/`. Run memory-heavy stages (blocking peaks ~45 GB) one at a time. A systemd
  timer **stops** the instance after 60 minutes with load below 0.3; start it again with
  `StartInstances`. Bootstrap: `infra/aws/ec2/user_data.sh`. Code updates: tar `src/` plus
  `tools/validate_submission.py`, PUT to `code/<name>.tar.gz` with a presigned URL, then
  `aws s3 cp ... - | tar -xz` into a fresh `pipe*` dir (running jobs keep their code). Presigned
  URLs are long: write them to a file before calling curl (a multi-URL command got truncated).
- AWS runbook (resources, compute options, costs, credentials): `infra/aws/README.md`; quota
  requests: `infra/aws/request_quotas.py --status / --apply`.
- **Console links** (the user wants these in every report that touches S3 or SageMaker):
  - Bucket: https://us-east-1.console.aws.amazon.com/s3/buckets/amazon-ml-challenge-26-sagemaker-125650147728?region=us-east-1&tab=objects
    (append `&prefix=predictions/full/` etc. for a folder)
  - SageMaker domain: https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/studio/d-tgxrzl4mojbh
  - Training jobs: https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs (one job: `#/jobs/<name>`)
  - EC2 runner: https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#InstanceDetails:instanceId=i-033e809bc1d8c21b5
