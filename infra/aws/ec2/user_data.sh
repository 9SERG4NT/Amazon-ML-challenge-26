#!/bin/bash
# First-boot bootstrap for the mlc26 batch runner (Amazon Linux 2023). launch_runner.sh fills in BUCKET and
# PRESET. With a PRESET, the runner then runs that preset end to end on its own and syncs its log (every
# 5 min) and results to s3://$BUCKET/, so it needs no further commands.
# Log: /var/log/mlc26-bootstrap.log. Success marker: /opt/mlc26/READY, failure: /opt/mlc26/FAILED.
set -euxo pipefail
exec > >(tee -a /var/log/mlc26-bootstrap.log) 2>&1

BUCKET=amazon-ml-challenge-26-ACCOUNT
PRESET=
BRANCH=sumukh/full-pipeline-aws
REPO=https://github.com/9SERG4NT/Amazon-ML-challenge-26
ROOT=/opt/mlc26
mkdir -p "$ROOT/logs" /data
trap 'echo "bootstrap failed at line $LINENO" > $ROOT/FAILED; aws s3 cp /var/log/mlc26-bootstrap.log "s3://$BUCKET/experiments/logs/bootstrap.log" || true' ERR

# Idle watchdog first, so a failed bootstrap can't leave a paid instance running.
# Every 5 min: if the 5-min load average is below 0.3 for 12 checks in a row
# (60 min), shut down. InstanceInitiatedShutdownBehavior=stop, so this stops
# (not terminates) the instance and the EBS volume keeps all work.
cat > /usr/local/bin/mlc26-idle-stop <<'EOF'
#!/bin/bash
f=/run/mlc26-idle   # tmpfs: the counter resets on every boot
load=$(cut -d' ' -f2 /proc/loadavg)
if awk "BEGIN{exit !($load < 0.3)}"; then n=$(( $(cat $f 2>/dev/null || echo 0) + 1 )); else n=0; fi
echo "$n" > "$f"
if [ "$n" -ge 12 ]; then logger "mlc26: idle for 60 min, stopping instance"; /sbin/shutdown -h now; fi
EOF
chmod +x /usr/local/bin/mlc26-idle-stop
cat > /etc/systemd/system/mlc26-idle-stop.service <<'EOF'
[Unit]
Description=Stop this instance when it has been idle for 60 minutes
[Service]
Type=oneshot
ExecStart=/usr/local/bin/mlc26-idle-stop
EOF
cat > /etc/systemd/system/mlc26-idle-stop.timer <<'EOF'
[Unit]
Description=Idle check every 5 minutes
[Timer]
OnBootSec=5min
OnUnitActiveSec=5min
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now mlc26-idle-stop.timer

# Swap: a memory peak above the 64 GB slows a run down instead of killing it.
fallocate -l 48G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile

# Build tools only matter if a wheel is missing and pip has to compile.
dnf install -y -q tar gzip git htop tmux gcc-c++ cmake

# Python 3.12 (matches local dev) with the pipeline's pinned dependencies.
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
export UV_PYTHON_INSTALL_DIR=/opt/uv-python
/usr/local/bin/uv venv --python 3.12 "$ROOT/venv"
/usr/local/bin/uv pip install --python "$ROOT/venv/bin/python" \
  polars==1.41.2 numpy==2.4.6 scipy==1.17.1 lightgbm==4.7.0 rapidfuzz==3.14.5 \
  sparse_dot_topn==1.2.0 anyascii==0.3.3 pyarrow==24.0.0 psutil==7.2.2

# Competition data: wait (up to 2 h) until the 7 TSVs are in S3, then same-region S3 -> EBS, about 2.4 GB.
for i in $(seq 1 240); do
  n=$(aws s3 ls "s3://$BUCKET/dataset/" --recursive | grep -c '\.tsv$' || true)
  [ "$n" -ge 7 ] && break
  sleep 30
done
sleep 60  # the last file may still be uploading when it first appears in the listing
aws s3 sync "s3://$BUCKET/dataset/" /data/dataset/ --only-show-errors
du -sh /data/dataset

"$ROOT/venv/bin/python" -c "import polars, lightgbm, rapidfuzz, sparse_dot_topn, anyascii; print('imports ok')"
git clone -q --depth 1 --branch "$BRANCH" "$REPO" "$ROOT/repo"
touch "$ROOT/READY"
[ -z "$PRESET" ] && exit 0

# The run: one preset end to end, detached from this boot script.
printf 'BUCKET=%s\nPRESET=%s\n' "$BUCKET" "$PRESET" > "$ROOT/run.env"
cat > "$ROOT/run.sh" <<'EOF'
#!/bin/bash
# Run $PRESET end to end; sync its log to S3 every 5 minutes, then upload predictions, metrics and fold models.
source /opt/mlc26/run.env
ROOT=/opt/mlc26 PY=/opt/mlc26/venv/bin/python LOG=/opt/mlc26/logs/$PRESET.log S3=s3://$BUCKET
cd "$ROOT/repo/code/business_entity_resolution/src"
KEY=$($PY -c "import ber.versions as V; print(V.resolve('$PRESET').key('predict'))")
(while sleep 300; do aws s3 cp "$LOG" "$S3/experiments/logs/$PRESET.log" --only-show-errors; done) &
SYNC=$!
$PY -u run_pipeline.py --data /data/dataset --work "$ROOT/runs/full" --out "$ROOT/out/$KEY" --preset "$PRESET" \
  --validator "$ROOT/repo/resources/student_resource/utils/validate_submission.py" > "$LOG" 2>&1
rc=$?
kill $SYNC
aws s3 cp "$ROOT/out/$KEY" "$S3/predictions/$KEY/" --recursive --only-show-errors
for f in "$ROOT"/runs/full/*/*/*.json; do
  aws s3 cp "$f" "$S3/experiments/metrics/$(basename "$(dirname "$f")")/$(basename "$f")" --only-show-errors
done
aws s3 cp "$ROOT/runs/full/runs/$KEY/" "$S3/models/$KEY/" --recursive --exclude "*" --include "stage*_fold*.txt" --only-show-errors
echo "$rc" > "$ROOT/logs/$PRESET.exit"
aws s3 cp "$LOG" "$S3/experiments/logs/$PRESET.log" --only-show-errors
aws s3 cp "$ROOT/logs/$PRESET.exit" "$S3/experiments/logs/$PRESET.exit" --only-show-errors
EOF
chmod +x "$ROOT/run.sh"
setsid nohup "$ROOT/run.sh" > "$ROOT/logs/run.out" 2>&1 < /dev/null &
