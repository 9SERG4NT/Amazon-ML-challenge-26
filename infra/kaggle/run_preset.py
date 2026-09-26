"""Run one pipeline preset end to end on a Kaggle session, on the challenge data attached as a private dataset.

Used when the AWS runner is unavailable. Each kernel folder under infra/kaggle/ holds a copy of this file
with its own CONFIG (Kaggle pushes a single code file): a full run needs ~48 GB of RAM (blocking peaks at
~41 GB), so it runs on a TPU VM machine shape for its host memory, not for the TPU; a smoke run
(SAMPLE < 1) fits a CPU session. Outputs land in /kaggle/working/: output/matching_results.tsv,
output/candidate_pairs.tsv, run.log, the stage metrics JSON files and the fold models (models/<run key>/).
"""
import glob
import os
import shutil
import subprocess
import sys
import threading
import time

CONFIG = {"preset": "M-v7", "sample": 1.0, "min_ram_gb": 48, "args": ""}  # args: extra run_pipeline.py flags
REPO = "https://github.com/9SERG4NT/Amazon-ML-challenge-26"
BRANCH = "sumukh/full-pipeline-aws"
NEEDED = {"train": ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"],
          "test": ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]}
WORK, DATA, OUT = "/tmp/work", "/tmp/data", "/kaggle/working"
T0 = time.time()


def sh(cmd: str, check: bool = True) -> int:
    print(f"[{time.time() - T0:7.0f}s] $ {cmd}", flush=True)
    rc = subprocess.run(cmd, shell=True, executable="/bin/bash").returncode  # bash: PIPESTATUS below
    if check and rc != 0:
        raise SystemExit(f"failed ({rc}): {cmd}")
    return rc


def meminfo() -> dict:
    with open("/proc/meminfo") as f:
        return {line.split(":")[0]: int(line.split()[1]) / 2**20 for line in f}


def watch_memory(every: int = 300) -> None:
    """Log used RAM and free disk every few minutes: a run killed for memory leaves this trail in the log."""
    while True:
        m = meminfo()
        free_disk = shutil.disk_usage("/tmp").free / 2**30
        print(f"[{time.time() - T0:7.0f}s] mem used {m['MemTotal'] - m['MemAvailable']:.1f} of {m['MemTotal']:.0f} GB, "
              f"/tmp free {free_disk:.0f} GB", flush=True)
        time.sleep(every)


def find(name: str) -> str | None:
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) + glob.glob(f"/tmp/unzipped/**/{name}", recursive=True)
    return hits[0] if hits else None


# 1. machine: fail fast if the session is too small for the run
sh("nproc; free -g; df -h /tmp /kaggle/working; python --version", check=False)
ram = meminfo()["MemTotal"]
if ram < CONFIG["min_ram_gb"]:
    raise SystemExit(f"only {ram:.0f} GB RAM: this run needs ~{CONFIG['min_ram_gb']} GB. Use a TPU VM machine shape.")
threading.Thread(target=watch_memory, daemon=True).start()

# 2. data: the dataset may arrive as folders or as zip files; link the 7 files into /tmp/data/{train,test}
if not find("train_source1.tsv"):
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        sh(f"unzip -q -o '{z}' -d /tmp/unzipped")
for split, names in NEEDED.items():
    os.makedirs(f"{DATA}/{split}", exist_ok=True)
    for n in names:
        src = find(n)
        if src is None:
            raise SystemExit(f"{n} not found under /kaggle/input: attach the private dataset serg4nt/mlc26-ber-data")
        dst = f"{DATA}/{split}/{n}"
        if not os.path.exists(dst):
            os.symlink(src, dst)
sh(f"ls -lL {DATA}/train {DATA}/test")

# 3. code and environment: the pinned requirements, then the pipeline from GitHub
sh("pip install -q polars==1.41.2 lightgbm==4.7.0 rapidfuzz==3.14.5 sparse_dot_topn==1.2.0 anyascii==0.3.3 psutil xgboost==3.4.1 "
   "|| pip install -q polars lightgbm rapidfuzz sparse_dot_topn anyascii psutil xgboost")
sh(f"git clone -q --depth 1 --branch {BRANCH} {REPO} /tmp/repo && git -C /tmp/repo log --oneline -1")
src_dir = "/tmp/repo/code/business_entity_resolution/src"
validator = "/tmp/repo/resources/student_resource/utils/validate_submission.py"

# 4. the run: every stage of the preset, log kept in the notebook output
os.chdir(src_dir)
rc = sh(f"{sys.executable} -u run_pipeline.py --data {DATA} --work {WORK} --out {OUT}/output --preset {CONFIG['preset']} "
        f"--sample {CONFIG['sample']} {CONFIG.get('args', '')} --validator {validator} 2>&1 | tee {OUT}/run.log; exit ${{PIPESTATUS[0]}}", check=False)
for f in glob.glob(f"{WORK}/runs/*/*.json") + glob.glob(f"{WORK}/block/*/*.json") + glob.glob(f"{WORK}/feat/*/*.json"):
    shutil.copy(f, f"{OUT}/{os.path.basename(os.path.dirname(f))}__{os.path.basename(f)}")
for f in glob.glob(f"{WORK}/runs/*/stage*_fold*.*"):  # fold models: a later session can score them frozen
    dst = f"{OUT}/models/{os.path.basename(os.path.dirname(f))}"
    os.makedirs(dst, exist_ok=True)
    shutil.copy(f, dst)
sh(f"ls -la {OUT} {OUT}/output {OUT}/models/*; md5sum {OUT}/output/*.tsv", check=False)
print(f"done in {time.time() - T0:.0f}s, pipeline exit {rc}", flush=True)
sys.exit(rc)
