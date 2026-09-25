#!/bin/bash
# First-boot bootstrap for the mlc26 batch runner (Amazon Linux 2023).
# Log: /var/log/mlc26-bootstrap.log. Success marker: /opt/mlc26/READY, failure: /opt/mlc26/FAILED.
set -euxo pipefail
exec > >(tee -a /var/log/mlc26-bootstrap.log) 2>&1
trap 'echo "bootstrap failed at line $LINENO" > /opt/mlc26/FAILED' ERR

BUCKET=amazon-ml-challenge-26-sagemaker-125650147728
ROOT=/opt/mlc26
mkdir -p "$ROOT" /data

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

# Build tools only matter if a wheel is missing and pip has to compile.
dnf install -y -q tar gzip git htop tmux gcc-c++ cmake

# Python 3.12 (matches local dev) with the pipeline's pinned dependencies.
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
export UV_PYTHON_INSTALL_DIR=/opt/uv-python
/usr/local/bin/uv venv --python 3.12 "$ROOT/venv"
/usr/local/bin/uv pip install --python "$ROOT/venv/bin/python" \
  polars==1.41.2 numpy==2.4.6 scipy==1.17.1 lightgbm==4.7.0 rapidfuzz==3.14.5 \
  sparse_dot_topn==1.2.0 anyascii==0.3.3 pyarrow==24.0.0 psutil==7.2.2

# Competition data: same-region S3 -> EBS, about 2.4 GB.
aws s3 sync "s3://$BUCKET/dataset/" /data/dataset/ --only-show-errors
du -sh /data/dataset

"$ROOT/venv/bin/python" -c "import polars, lightgbm, rapidfuzz, sparse_dot_topn, anyascii; print('imports ok')"
touch "$ROOT/READY"
