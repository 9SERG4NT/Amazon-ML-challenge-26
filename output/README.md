# Output files

`matching_results.tsv` and `candidate_pairs.tsv` in this folder are the **current submission**: the
file uploaded to the leaderboard, and the two files `make_submission.py` packs. They must keep
these exact names.

**Current submission: M-v5** (`NORM-v2__BLK-v4b@20-tlu40__FEAT-v3__MATCH-v6`): trained and
validated in the test-like universe, stage-2 probabilities with the gated expected-F rule (gate
0.55). It replaced FULL-v1 (kept in `runs/`) and raised the public leaderboard from 0.96 to 0.971.

Every complete run keeps its own copy in `runs/<version key>/`, named like its S3 folder
`s3://amazon-ml-challenge-26-sagemaker-125650147728/predictions/<version key>/`. The TSVs are not
in git: `*.tsv` is ignored outside the top of this folder, and `candidate_pairs.tsv` is ~1.1 GB.

| Run (preset) | Version key (folder name) | Eval universe | Eval F0.5 | Public leaderboard | md5 of `matching_results.tsv` |
|---|---|---|---:|---:|---|
| FULL-v1 (M-v3) | `NORM-v2__BLK-v4b@20__FEAT-v2__MATCH-v2` | full train | 0.9813 | 0.96 | `b4644450c78b5bf64f8124223223a005` |
| **M-v5** | `NORM-v2__BLK-v4b@20-tlu40__FEAT-v3__MATCH-v6` | test-like | 0.9852 | **0.971** | `9c66b92887f0eced0291bc42f78d8b2f` |
| M-v6 | `NORM-v2__BLK-v4b@20-tlu40__FEAT-v4__MATCH-v6` | test-like | queued | | |

Eval scores are comparable only within one universe. The test-like universe (every eval S1 plus
40% of the other train S1, ~2.3 distractors per S1 like the test set) is harder than the full-train
one, so its scores are lower by construction. The eval slice is the same 441,521 S1 entities in
both. Methods and full results: [`../method_result.md`](../method_result.md).
