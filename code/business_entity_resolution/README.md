# Business Entity Resolution

End-to-end pipeline: data → blocking → matching → output.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place the challenge dataset at `resources/student_resource/dataset/` (train/ and test/ TSVs). The dataset is not committed to git because the files exceed GitHub's size limit.

## Run

_TODO: add the exact commands that regenerate `output/matching_results.tsv` and `output/candidate_pairs.tsv`._

## Validate

```bash
cd resources/student_resource
python utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir dataset/test
```
