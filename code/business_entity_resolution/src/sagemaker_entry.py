"""SageMaker training-job entry point for the full pipeline.

The job mounts the competition data as the "dataset" channel and runs run_pipeline.py
with its working files in /tmp (SageMaker's scratch space, never uploaded). It keeps:
  /opt/ml/output/data -> output.tar.gz  matching_results.tsv, candidate_pairs.tsv, metrics.json
  /opt/ml/model       -> model.tar.gz   the stage-1 / stage-2 LightGBM fold models, metrics.json
Hyperparameters arrive as ``--key value`` arguments and are passed through unchanged.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
data = os.environ.get("SM_CHANNEL_DATASET", "/opt/ml/input/data/dataset")
model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
out_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
work = Path("/tmp/work")

cmd = [sys.executable, str(here / "run_pipeline.py"), "--data", data, "--work", str(work), "--out", str(out_dir),
       *sys.argv[1:]]
if (here / "validate_submission.py").exists():
    cmd += ["--validator", str(here / "validate_submission.py")]
print("running:", " ".join(cmd), flush=True)
rc = subprocess.call(cmd, cwd=here)

model_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)
if (work / "metrics.json").exists():
    shutil.copy(work / "metrics.json", out_dir / "metrics.json")
    shutil.copy(work / "metrics.json", model_dir / "metrics.json")
for f in (work / "model").glob("stage*_fold*.txt"):
    shutil.copy(f, model_dir / f.name)
sys.exit(rc)
