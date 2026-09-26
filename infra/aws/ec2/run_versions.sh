#!/bin/bash
# One run_pipeline.py invocation on the runner, with its log and outputs synced to S3 (the bucket in run.env):
#   run_versions.sh <name> <run_pipeline.py flags...>     e.g. run_versions.sh <version key> --preset M-v6
# Log: /opt/mlc26/logs/<name>.log, copied to s3://$BUCKET/experiments/logs/ every 5 minutes; exit code in <name>.exit.
# Uploads predictions/<name>/ (if the run wrote test files), every stage's metrics JSON and every run's fold models.
source /opt/mlc26/run.env
NAME=$1
shift
ROOT=/opt/mlc26 PY=/opt/mlc26/venv/bin/python LOG=/opt/mlc26/logs/$NAME.log S3=s3://$BUCKET
cd "$ROOT/repo/code/business_entity_resolution/src" || exit 1
(while sleep 300; do aws s3 cp "$LOG" "$S3/experiments/logs/$NAME.log" --only-show-errors; done) &
SYNC=$!
$PY -u run_pipeline.py --data /data/dataset --work "$ROOT/runs/full" --out "$ROOT/out/$NAME" \
  --validator "$ROOT/repo/resources/student_resource/utils/validate_submission.py" "$@" > "$LOG" 2>&1
rc=$?
kill $SYNC
[ -d "$ROOT/out/$NAME" ] && aws s3 cp "$ROOT/out/$NAME" "$S3/predictions/$NAME/" --recursive --only-show-errors
for f in "$ROOT"/runs/full/*/*/*.json; do
  case $(basename "$f") in stage*) continue;; esac  # XGBoost fold models, uploaded below
  aws s3 cp "$f" "$S3/experiments/metrics/$(basename "$(dirname "$f")")/$(basename "$f")" --only-show-errors
done
aws s3 cp "$ROOT/runs/full/block/" "$S3/models/" --recursive --exclude "*" --include "*/filter_fold*.txt" --only-show-errors
aws s3 cp "$ROOT/runs/full/runs/" "$S3/models/" --recursive --exclude "*" --include "*/stage*_fold*.txt" --include "*/stage*_fold*.json" --include "*/stage*_fold*.cbm" --include "*/stage*_fold*.npz" --include "*/filter_fold*.txt" --only-show-errors
echo "$rc" > "$ROOT/logs/$NAME.exit"
aws s3 cp "$LOG" "$S3/experiments/logs/$NAME.log" --only-show-errors
aws s3 cp "$ROOT/logs/$NAME.exit" "$S3/experiments/logs/$NAME.exit" --only-show-errors
exit $rc
