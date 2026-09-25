# experiments-jatin — Business Entity Resolution

Jatin's workspace for the ML Challenge entity-resolution pipeline. Everything for this
experiment lives in this folder to avoid merge conflicts with teammates' work on the
shared skeleton (`code/`, `output/` at the repo root stay untouched).

```
experiments-jatin/
├── run_pipeline.py      # pipeline: blocking → LightGBM (GPU/CPU) → F0.5-tuned decisions
├── requirements.txt     # pinned dependencies (Python 3.12)
├── utils/
│   └── validate_submission.py   # official validator (copied unchanged)
└── output_dev/          # run outputs: matching_results.tsv + candidate_pairs.tsv
```

## Setup (one-time, from the repo root)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r experiments-jatin/requirements.txt
```

`lightgbm==4.7.0` pip wheels ship CUDA support, so `--device gpu` works out of the box
on any recent NVIDIA driver; fall back to `--device cpu` if needed.

## Run

From the **repo root**:

```bash
# Full run (train + inference), GPU:
.venv/bin/python -u experiments-jatin/run_pipeline.py \
    --data-dir resources/student_resource/dataset \
    --output-dir experiments-jatin/output_dev \
    --device gpu

# Quick dev smoke test (20k S1 records + reduced S2/S3 pool):
.venv/bin/python -u experiments-jatin/run_pipeline.py \
    --data-dir resources/student_resource/dataset \
    --output-dir experiments-jatin/output_dev \
    --device gpu --sample 20000
```

## Validate

From the **repo root**:

```bash
cd resources/student_resource
python3 ../../experiments-jatin/utils/validate_submission.py \
    --matching ../../experiments-jatin/output_dev/matching_results.tsv \
    --candidate ../../experiments-jatin/output_dev/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

(The validator only needs `dataset/test` relative to `student_resource/`; it never uses
ground truth and never computes a score. `--check-ids` is optional but recommended —
it also verifies every matched ID exists in the test set.)

## Current status

- Run: 2026-09-25, `--sample 20000` (reduced train pool; full test inference), GPU
- Validator: **PASS** (including `--check-ids`)
- Outputs: `output_dev/matching_results.tsv` (1,732,544 rows), `output_dev/candidate_pairs.tsv`
