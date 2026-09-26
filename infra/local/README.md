# Local runner

`run_local.py` runs the newest presets end to end on your own machine (Windows or Linux) in one command. It checks
the machine, finds the 7 challenge TSVs under `--data` (any folder layout), installs the pinned requirements, downloads
the pipeline from GitHub, and runs **M-v11** and then **M-v12**. Both use the learned candidate filter (BLK-v5-tlu40);
M-v12 re-runs only training and prediction on M-v11's candidates and features.

```bash
python run_local.py --data <folder with the TSVs> --sample 0.01   # 1% smoke run first: ~5 min, ~6 GB RAM
python run_local.py --data <folder with the TSVs>                 # full run: ~48 GB RAM, 8+ cores, ~100 GB disk, 2-3 h
```

Results go to `mlc26_out/<preset>/`: `matching_results.tsv`, `candidate_pairs.tsv`, `metrics.json` and `run.log`.
`mlc26_out/summary.json` compares the runs: eval F0.5, doubled-distractor eval, candidates per S1, and the md5 of each
file. The stage cache (`mlc26_work/`) lets an interrupted run resume: run the same command again.

A full run stops early when the machine has less than ~48 GB of RAM (`--force` tries anyway, which needs a large
swap or page file). The pipeline runs on the CPU (LightGBM), so an NVIDIA GPU is reported but not used. Use Python
3.12, which the pinned package versions were tested on.
