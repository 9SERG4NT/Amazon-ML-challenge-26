# Output files

`matching_results.tsv` and `candidate_pairs.tsv` in this folder are the **current submission**: the
file uploaded to the leaderboard, and the two files `make_submission.py` packs. They must keep
these exact names.

**Best public leaderboard score: TEAM-B-R5, 0.98** (the teammates' pipeline, in `runs/TEAM-B-R5__e5-FAISS__XGBoost-CE/`).
It becomes the current submission once its own `candidate_pairs.tsv` is downloaded from their Kaggle notebook. Until
then the two files at the top of this folder stay M-v11's: its matches with our candidate file would break the rule
that every match is a candidate.

**Current submission: M-v11** (`NORM-v2__BLK-v5-tlu40__FEAT-v4__MATCH-v6`, public leaderboard **0.970**): M-v6 behind
the learned candidate filter, **6.5 candidates per S1 instead of 48.1**, stage-2 probabilities with the expected-F rule
(floor 0.4). It replaced M-v5 (0.971, kept in `runs/`): the same leaderboard score within rounding, with a candidate set
7.4 times smaller, which the organisers rank higher in the final review.

Every complete run keeps its own copy in `runs/<version key>/`, named like its S3 folder
`s3://amazon-ml-challenge-26-sagemaker-125650147728/predictions/<version key>/`. The TSVs are not
in git: `*.tsv` is ignored outside the top of this folder, and `candidate_pairs.tsv` is ~1.1 GB.

| Run (preset) | Version key (folder name) | Eval universe | Eval F0.5 | Public leaderboard | md5 of `matching_results.tsv` |
|---|---|---|---:|---:|---|
| FULL-v1 (M-v3) | `NORM-v2__BLK-v4b@20__FEAT-v2__MATCH-v2` | full train | 0.9813 | 0.96 | `b4644450c78b5bf64f8124223223a005` |
| M-v5 | `NORM-v2__BLK-v4b@20-tlu40__FEAT-v3__MATCH-v6` | test-like | 0.9852 | 0.971 | `9c66b92887f0eced0291bc42f78d8b2f` |
| M-v6 (Kaggle TPU run) | `NORM-v2__BLK-v4b@20-tlu40__FEAT-v4__MATCH-v6` | test-like | 0.9856 | pending | `f055edc3edaeb41e0204d35dae4be8e9` |
| **M-v11** (EC2, current) | `NORM-v2__BLK-v5-tlu40__FEAT-v4__MATCH-v6` | test-like | 0.9854 | **0.970** | `f96f52d773991cc04870a36cf4b3ed46` |
| M-v7 (rejected, not downloaded) | `NORM-v2__BLK-v4b@20-dup2__FEAT-v4__MATCH-v6` | doubled distractors (copies) | 0.9856 | not submitted | — |
| **Team pipeline B, run R5** (Kaggle, teammates; best) | `TEAM-B-R5__e5-FAISS__XGBoost-CE` | their own 10% held out | 0.9875 (not comparable) | **0.98** | `67bdd7f85c52d143128e40dff42418eb` |

**TEAM-B-R5 (the teammates' pipeline, leaderboard 0.98, the best so far).** A fine-tuned e5-small bi-encoder with FAISS search for
blocking (5.95 candidates per test S1), then XGBoost with a cross-encoder score and expected-F selection. Its folder
holds the predictions converted to TSV (the file to upload), the CSV as downloaded, the team's write-up
(`all_approaches.md`) and a README comparing it with M-v11. It links more than M-v11 (France 3.17 against 3.13 links
per S1) and agrees on only 64% of French S1 (84–86% elsewhere). Its `candidate_pairs.tsv` is not downloaded yet.

M-v6 changes the M-v5 link set for 6.9% of French S1 (India 3.6%, US 3.4%), as its France-robust features intend;
its test profile matches M-v5's (94.1% of S1 linked, 3.27 links per S1). **M-v7 is rejected:** its training copied every
distractor, and the model learned to recognise the exact twins, so on the test (no twins) it links 98.5% of S1 with
4.20 links each (truth ~94%, ~3.4). Its files stay in S3 only
(`s3://amazon-ml-challenge-26-278311879294/predictions/NORM-v2__BLK-v4b@20-dup2__FEAT-v4__MATCH-v6/`).

**M-v11 (submitted, 0.970).** It is M-v6 behind the learned candidate filter (BLK-v5-tlu40): **6.5 candidates per
S1 instead of 48.1** (the organisers rank a smaller candidate set higher), eval F0.5 0.9854 against M-v6's 0.9856, the
same test profile (94.0% of S1 linked, 3.13–3.31 links each) and a passing validator. `candidate_pairs.tsv` is 167 MB
(md5 `64f560335280ca4965118ec0d78c0d54`) instead of ~1.1 GB.

Eval scores are comparable only within one universe. The test-like universe keeps every eval S1
plus 40% of the other train S1, which gives ~2.3 distractors per S1 as in the test set. It differs
from the full-train one in both directions: it has twice the distractors per S1, but half the rival
S1 records, so blocking and exclusivity lose less. The eval slice is the same 441,521 S1 entities
in both. Neither covers France. Methods and full results: [`../method_result.md`](../method_result.md).
