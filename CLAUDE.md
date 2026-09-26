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
- **Organisers' update (2026-09-26): `candidate_pairs.tsv` and the code behind it count in the final
  ranking, and a smaller candidate set per S1 ranks higher** (beyond the leaderboard score). Blocking must
  scale (no all-pairs). Several blocking/filtering stages are allowed; the file is the last one, i.e. what
  the matcher scores. Our answer: the learned candidate filter BLK-v5-tlu40 (presets M-v11, M-v12).

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
python run_pipeline.py ... --match MATCH-v6 --stages train,predict   # new matcher on cached blocking/features
python run_pipeline.py ... --sample 0.01            # end-to-end smoke run, ~1 min, ~6 GB RAM
# test-like train universe (the validation that tracks the leaderboard) and the frozen leaderboard model in it
python run_pipeline.py ... --block BLK-v4b@20-tlu40 --feat FEAT-v3 --match MATCH-v6
python run_pipeline.py ... --block BLK-v4b@20-tlu40 --match MATCH-v2-frozen --stages block,features,train,predict

# DEV-10 experiments (cache dir holds dev_Q/dev_T/dev_pairs/dev_features_v2/dev_part/...)
python -m experiments.dev_stack <cache>                              # stage 1 vs stage 2
python -m experiments.dev_decide <cache> dev_scores_v3.parquet p2    # decision rules on saved scores
python -m experiments.dev_errors <cache> dev_scores_v3.parquet p2 0.725
python -m experiments.dev_v3 <cache>                                 # FEAT-v3 number gap + support features
# full-data work dir: eval-loss breakdown with examples; train/test shift checks (leak, stats, preds)
python -m experiments.full_errors <work> NORM-v2__BLK-v4b@20__FEAT-v2__MATCH-v2 p2 gated_ef 0.5 12
python -m experiments.shift_check <work> leak,stats,preds <dataset_dir>
```

Final package: copy the chosen run's TSVs from `s3://…/predictions/<version>/` into `output/`,
then `python make_submission.py --team "<name>"` (repo root) validates them and writes
`<team>_submission.zip` in the required layout. The user wants the best complete run's TSVs kept in
the local `output/` for leaderboard uploads: download them after every full run that beats the
current best (presigned GET + curl), and check the md5 against the runner. `output/` now holds
**M-v5** (test-like eval 0.9852, **public leaderboard 0.971**, rank ~400; md5 `9c66b928…`). The remaining gap is most likely France (implied ~0.89): next is M-v6 (FEAT-v4). Before it came FULL-v1
(`predictions/M-v3/`, eval F0.5 0.9813, **public leaderboard 0.96**; top of the board 0.99),
which is kept in `output/runs/`.
Output naming: `output/matching_results.tsv` and `output/candidate_pairs.tsv` are the current submission
(exact names required). Every complete run also goes to `output/runs/<version key>/`, named like
`predictions/<version key>/` in S3, and `output/README.md` is the manifest: run name, version key,
eval F0.5, leaderboard score, md5. Update it with every run. FULL-v1 is also copied to
`predictions/NORM-v2__BLK-v4b@20__FEAT-v2__MATCH-v2/`. The older prefixes `predictions/M-v3/`
(identical) and `predictions/full/` (expected-F 0.4 rule, same score) still exist: ask the user
before deleting them.
The user has 1–2 leaderboard submissions a day: use them for milestone models, never to tune on the
public subset.

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
   Context features need all candidates of a country at once; the rest are computed in parts of
   4M rows. `ber/partition.py` runs the context features and stage 2's `probability_context` one
   country at a time (blocking never pairs two countries, so no window crosses one): identical
   values, lower peak memory — needed for deeper blocking (~200M train candidates at top-40).
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
scored exactly like test rows. The split draws one uniform number per entity (fit `u < 0.3`,
early-stop `[0.3, 0.35)`, eval `u >= 0.8`), so the eval slice never depends on the fit share, and
every full-data run is scored on the same 441,521 eval entities. MATCH-v6 (`fit_rest`) turns the
rest entities into fit entities (75% of train S1 instead of 30%) without touching prep, blocking
or features.

Gotchas:
- `probability_context` returns `p1` as its first column; appending `p1` again duplicates it.
- `nums` lists come from `unique()` without order, so "first number" is not the house number.
- DEV-10 (10% of S1, their matches, 10% of distractors) has ~10× less competition per region
  than full data: its scores are optimistic. Compare variants on DEV-10, report on full data.
- `rid` is a dense row index per split (`with_row_index` in prep), so per-record lookups can use
  plain arrays indexed by `q_rid` / `t_rid`.
- A change that must not alter results (e.g. a memory refactor) needs an exact-equality check
  against the old code path, including ordinal-rank ties and nulls.

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

**FULL-v1 matcher (M-v3 on full data; current best, the file in `output/`):** eval slice (441,521 S1)
**F0.5 0.9813**, P 0.9952, R 0.9544, singletons 0.9774, others 0.9815; India 0.9778, US 0.9837.
Chosen rule: stage 2, gated expected-F, gate 0.5. Expected-F (floor 0.4) scores the same to 6
decimals, and the best threshold is only 0.0004 lower, so the rule no longer matters. A perfect
matcher on these candidates would score 0.9947. **The matcher (-0.0134) now loses more than
blocking (-0.0053).** 2.9% of true links are candidates but are not predicted (1.5% on DEV-10),
so DEV-10 overstated F0.5 by 0.009. Test: 94.1% of S1 get a link in every country, France
included (3.18 links/S1; eval truth 94.4%, 3.46). The validator passes.

**Leaderboard: FULL-v1 scored 0.96 (public), top 0.99 — the eval slice (0.9813) does not track
the test.** No leak (row order and ID numbers are uncorrelated with the truth). The test has
**twice the distractors per S1**: S2 and S3 hold equal distractor counts in train (1,340,997 vs
1,340,857), so the S3−S2 excess gives the matched records. Test: 3.3–3.5 matches but **2.2–2.35
distractors per S1** (train 1.15–1.31), i.e. ~40% of test targets (train 26%). The US test has half
the train's S1 density with the same absolute distractor count. Distractors are "clean" records
(full address and number, never web/alias names). On test the model is as confident and links as
much as on eval, so the extra errors are confident lookalike picks, not threshold effects. Fix,
validation first: **test-like universe `BLK-v4b@20-tlu40`**. It keeps every eval S1 and 40% of the
other train S1 entities, and drops the rest with their true targets, giving ~2.3 distractors per S1.
Train and choose the rule there (MATCH-v6); `MATCH-v2-frozen` scores the leaderboard model there, to
check that this universe reproduces ~0.96. `experiments/shift_check.py` and `full_errors.py` hold
the diagnostics.

Full-data error analysis (FULL-v1 eval, loss 0.0187): **recall is 0.0145 of it.** Of the 70.7k
missed links (4.6%), 38% are candidates not chosen by the rule (house number replaced or perturbed,
a name word swapped), 36% were never candidates (initials/acronyms, invented names, typo-heavy
names that rank below 20 lookalikes), and 26% were lost to another S1 (91% no-address targets
with a name several S1 records share: mostly irreducible). Wrong links (5.8k) are 85% distractors.
DEV-10 `dev_v3`: FEAT-v3 number-gap features lift stage 1 from 0.9894 to 0.9907 (threshold) and
from 0.9887 to 0.9903 (expected-F), so the TLU run uses FEAT-v3.

FULL-v1 cost on the EC2 runner (M-v3, 8 cores): prep 386 s, block 1,202 s (peak 44.4 GB during US
blocking: 1.32M S1 x 6.19M targets), features 2,344 s (104.7M train + 83.3M test candidates, 47.5
and 48.1 per S1). Stage-1 folds 638–718 s (1,277–1,469 trees at lr 0.05, 36.6M training rows,
7.2% positive); stage-2 folds 255–330 s (444–573 trees). Scoring all 188M candidates with the
stage-1 models took 2.4 h (with a second job competing for the CPU).

Matcher versions after M-v3: MATCH-v4 (LightGBM prediction early stopping) is **experimental** —
on tiny data it moved stage-1 probabilities by up to 0.31, so it is not a free speedup; MATCH-v5
= learning rate 0.1 (about half the trees, same DEV score on tiny data); MATCH-v6 = MATCH-v5 +
`fit_rest`. Next full-data decisions wait on: the BLK-v4b@40n20 depth curves (top-k for BLK-v5),
`dev_v3` (keep FEAT-v3 number-gap and MATCH-v3 support features?), and MATCH-v6 vs FULL-v1 on
the shared eval slice.

## AWS

- **The AWS account was suspended on 2026-09-26 (~01:00 IST)**: the runner and S3 are unreachable. The fallback is a Kaggle notebook (`infra/kaggle/m-v6/`, TPU VM machine shape for its RAM), run on a private Kaggle dataset `serg4nt/mlc26-ber-data` that the user uploads. The permission classifier blocks Claude from uploading the challenge data.
- **Kaggle:** the dataset `serg4nt/mlc26-ber-data` is up (7 files, sizes match). Kernels: `infra/kaggle/run_preset.py` is the
  generic runner (logs memory every 5 min, saves the fold models); each kernel folder (`m-v6/`, `m-v7/`, `smoke/`) holds a
  copy with its own CONFIG; push with `kaggle kernels push -p infra/kaggle/<folder>`. Only **one TPU batch session per user,
  queued ones included** (a second push fails with "Maximum batch TPU session count of 1 reached"); TPU queue waits run
  30+ minutes. CPU sessions (30 GB, 4 cores) start at once and fit the 1% smoke run (~5 min). `kaggle kernels logs` is
  empty while a kernel runs; `kaggle kernels output <id> -p <dir>` downloads the finished outputs.
- **New AWS account 278311879294 (2026-09-26, root via `aws login`):** bucket `amazon-ml-challenge-26-278311879294`
  (private, AES256, TLS only), role/instance profile `mlc26-ec2-runner` (SSM + that bucket), security group
  `sg-008a21c8fd1ee3aca` (no inbound), 8-vCPU quota. It is on the **free plan: RunInstances refuses r7a.2xlarge** ("not
  eligible for Free Tier") until the user upgrades to a paid plan. **Upgraded to the paid plan at ~09:40 IST ($140 credits
  carried over).** Runner **`i-04fb5a4206b8bb43e`** (r7a.2xlarge, us-east-1d, 48 GB swap) launched 09:33 IST; the user
  uploaded `dataset/` (the classifier blocks Claude's presigned upload as "Data Exfiltration"). It runs M-v7 (log
  `logs/M-v7.log`, 09:45 IST start), then `chain3.sh` (replaced chain2 at 10:29 IST before it started anything), each via
  `infra/aws/ec2/run_versions.sh`: M-v8 (XGBoost, MATCH-v8, on M-v7's cached features; installs xgboost 3.4.1 first), M-v9
  (MATCH-v9 = mean of M-v7's and M-v8's probabilities), M-v6 (reuses prep), then `MATCH-v6-frozen-tlu` on M-v7's
  features, touching `logs/CHAIN3_DONE`. chain3 was stopped after M-v7 failed its test profile. **chain4** (15:08 IST)
  ran tlu40 blocking + FEAT-v4 features (`logs/tlu40-FEAT-v4-features.log`); its bash was killed at 15:50 IST so its
  M-v10 / M-v6 steps would not run, and **chain5** (`/opt/mlc26/chain5.sh`) waits for `tlu40-FEAT-v4-features.exit`,
  then runs M-v11 (block,features,train,predict: the learned candidate filter), M-v12 (train,predict), M-v10 and M-v6
  (train,predict on the unfiltered candidates) and touches `logs/CHAIN5_DONE`; log names are the version keys. Kaggle GPU sessions have the same ~29 GB RAM as CPU ones: no full run fits. Read-only SSM commands (tail logs) work. SageMaker:
  every large-instance quota is 0; ml.m5.4xlarge training/spot requests are CASE_OPENED. Kaggle CPU sessions (30 GB)
  cannot hold a full run: M-v6 there was killed in train blocking after prep peaked at 23.8 GB.
  Then `infra/aws/ec2/launch_runner.sh <PRESET>` (or the same
  RunInstances call) starts a runner whose bootstrap waits for `dataset/`, runs the preset and syncs
  `experiments/logs/<PRESET>.log`, `predictions/<key>/`, `experiments/metrics/`, `models/<key>/` to the bucket. Creating
  resources there needs the user's explicit go-ahead (the classifier blocked the first attempt as "Modify Shared Resources").
- **Account 125650147728, region us-east-1.** A teammate's notes describe a different account
  (911797456769); its bucket is not accessible from here.
- **Auth:** `aws login` (short-lived credentials, root user; root has no access keys; enabling root
  MFA is recommended). Local boto3 needs `pip install "botocore[crt]"` for the login provider. In Claude Code,
  use the aws-mcp tools (its `run_script` sandbox blocks the `base64` module). In `call_boto3`,
  operation names are PascalCase (`SendCommand`, `GetCommandInvocation`); a script that runs
  longer than ~1 minute turns into a task polled with `get_tasks`, so keep scripts short and never
  sleep in them. Presigned URLs carry temporary credentials: never show them to the user.
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
  Pending AWS review (still `CASE_OPENED` at 15:50 UTC on 2026-09-25, filed 07:05–07:35 UTC) —
  training on ml.m5.4xlarge / ml.m5.xlarge / ml.g5.2xlarge, processing on ml.g5.xlarge, Studio
  JupyterLab on ml.r5.4xlarge / ml.g5.xlarge. Every other training and processing quota is 0; only
  ml.t3.medium Studio/notebook compute works. EC2 standard vCPU quota: 8. ml.m5.4xlarge has the
  same 64 GB as the EC2 runner, so a run that fits one fits the other.
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
  files in `logs/`. Run memory-heavy stages (blocking peaks ~45 GB) one at a time, and **never run a
  second LightGBM/OpenMP job beside a full run, even under `nice 19`**: the two thread pools fight
  over the 8 cores and full-data stage-1 scoring took 2.4 h instead of ~1 h. Small jobs are no
  exception: a 1% smoke test beside `dev_v3` pushed the load to 16 and slowed both several-fold.
  Give any side job `OMP_NUM_THREADS=2 POLARS_MAX_THREADS=2`, or queue it behind a marker. Full-data models grow
  ~1,300–1,500 trees (lr 0.05); scoring 188M pairs with 3 of them is the slowest step. A systemd
  timer **stops** the instance after 60 minutes with load below 0.3; start it again with
  `StartInstances`. Bootstrap: `infra/aws/ec2/user_data.sh`. Code updates: tar `src/` plus
  `tools/validate_submission.py`, PUT to `code/<name>.tar.gz` with a presigned URL, then
  `aws s3 cp ... - | tar -xz` into a fresh `pipe*` dir (running jobs keep their code). Presigned
  URLs are long: write them to a file before calling curl (a multi-URL command got truncated).
- AWS runbook (resources, compute options, costs, credentials): `infra/aws/README.md`; quota
  requests: `infra/aws/request_quotas.py --status / --apply`.
- **Runner job queue (2026-09-25; each chain waits for the previous marker in `logs/`):**
  chain2 = FULL-v1 (M-v3, code `pipe/`) → uploads `predictions/full/`, `models/full/`, touches
  `FULL_DONE` (**done** 15:56 UTC, 4 h 46 min). chain5 (`pipe3/`) = `migrate_v1.py` (moves FULL-v1
  into the version-keyed layout), then a predict re-run of M-v3 with the gated rule → `predictions/M-v3/`
  (**done** 15:58), then the BLK-v4b@40n20 block stage for the depth curves (running since 15:59)
  → `CHAIN5_DONE`. chain6 (`pipe2/`) = `dev_v3` on DEV-10 (`/opt/mlc26/dev10`) → `CHAIN6_DONE`.
  chain7 (`pipe4/`) = 1% smoke test of the new code (`logs/smoke4.log`), then MATCH-v6 train+predict
  on the cached FULL-v1 features (`logs/full_v6.log`) → `predictions/NORM-v2__BLK-v4b@20__FEAT-v2__MATCH-v6/` →
  `CHAIN7_DONE`. **chain7 was cancelled at 16:33 with the user's approval, before it started any
  job**: MATCH-v6 in the full-train universe no longer targets the leaderboard. **chain8** (`pipe6/`)
  waits for `CHAIN6_DONE` and a clean `smoke6.log` (1% smoke test of the TLU and frozen-model code).
  It runs `--block BLK-v4b@20-tlu40 --feat FEAT-v3` block+features, then train+predict with the
  MATCH id in `/opt/mlc26/chain8.match` (default MATCH-v6), logs to `logs/tlu.log`, uploads
  `predictions/NORM-v2__BLK-v4b@20-tlu40__FEAT-v3__<MATCH>/` and touches `CHAIN8_DONE` (started
  16:51 UTC). **chain9** (`pipe7/`) waits for `CHAIN8_DONE` and a clean `smoke7.log`, then runs
  FEAT-v4 features + MATCH-v6 train/predict on chain8's blocking (`logs/tlu_v4.log`,
  `predictions/NORM-v2__BLK-v4b@20-tlu40__FEAT-v4__MATCH-v6/`, `CHAIN9_DONE`). Both run
  MATCH-v6: support features (MATCH-v3) added only +0.0001 on DEV-10 in `dev_v3`. The runner's
  `pipe7/` also registers an unused MATCH-v7 (v6 + support) that was dropped from git.
  Launch chains detached with an absolute path and no `cd … &&` in front, e.g.
  `setsid nohup /opt/mlc26/chainN.sh > /opt/mlc26/logs/chainN.out 2>&1 < /dev/null &` (each
  chain `cd`s itself). With `cd … && …&`, bash keeps a wrapper subshell holding the SSM command's
  output pipe: the command sits InProgress until its timeout kills the subshell. `setsid` keeps the
  chain alive through that; without it the chain would die with the subshell. chain7's first launch
  got stuck this way and was relaunched with setsid at 15:58.
  Logs sync to `experiments/logs/`. After the chains, delete `models/full/scores_*.parquet` from S3
  (multi-GB; the instance role cannot delete by design).
- **Console links** (the user wants these in every report that touches S3 or SageMaker):
  - Bucket: https://us-east-1.console.aws.amazon.com/s3/buckets/amazon-ml-challenge-26-sagemaker-125650147728?region=us-east-1&tab=objects
    (append `&prefix=predictions/full/` etc. for a folder)
  - SageMaker domain: https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/studio/d-tgxrzl4mojbh
  - Training jobs: https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs (one job: `#/jobs/<name>`)
  - EC2 runner: https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#InstanceDetails:instanceId=i-033e809bc1d8c21b5

## Git

- Branch `sumukh/full-pipeline-aws` on a **public** GitHub repo. Commits are authored by the user
  (9SERG4NT) only: no AI co-author or attribution lines in commit messages or PR descriptions.
- `output/candidate_pairs.tsv` is ignored (about 1 GB, over GitHub's 100 MB limit; it lives in
  S3). `output/matching_results.tsv` is not ignored (the user's original rule), but it is a test-set
  prediction: ask before committing it to the public repo.
