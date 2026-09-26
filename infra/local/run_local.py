"""Run the newest pipeline end to end on your own machine (Windows or Linux), one command.

    python run_local.py --data <folder holding the 7 challenge TSVs>            # full run: M-v11, then M-v12
    python run_local.py --data <folder> --sample 0.01                           # 1% smoke run first (~5 min)

Steps: check the machine (RAM, cores, disk, GPU) -> find the 7 TSVs anywhere under --data -> install the pinned
requirements into this Python -> download the pipeline from GitHub (or use --code <repo checkout>) -> run each
preset -> copy each run's matching_results.tsv, candidate_pairs.tsv, metrics.json and log to <out>/<preset>/ and
print a comparison.

Presets (newest first; see code/business_entity_resolution/src/ber/versions.py):
  M-v11  test-like training universe + learned candidate filter (BLK-v5-tlu40) + FEAT-v4 + two-stage LightGBM (MATCH-v6)
  M-v12  the same candidates and features, distractor rows weighted x2, rule chosen on the doubled-distractor eval
A later preset that shares the normalisation, blocking and features of an earlier one only re-runs train and predict.

Hardware: a full run needs about 48 GB of RAM (the candidate search peaks at ~41-44 GB), 8+ cores and ~100 GB of free
disk; it takes 2-3 hours on 8 cores. The pipeline runs on the CPU (LightGBM): an NVIDIA GPU is reported but not used.
Python 3.12 is recommended (the pins were tested on it); other versions fall back to unpinned packages.
Licences: LightGBM MIT; polars MIT; rapidfuzz MIT; sparse_dot_topn Apache-2.0; anyascii ISC; numpy/scipy BSD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

REPO = "https://github.com/9SERG4NT/Amazon-ML-challenge-26"
BRANCH = "sumukh/full-pipeline-aws"
PINS = ["polars==1.41.2", "numpy==2.4.6", "scipy==1.17.1", "lightgbm==4.7.0", "rapidfuzz==3.14.5",
        "sparse_dot_topn==1.2.0", "anyascii==0.3.3", "pyarrow==24.0.0", "psutil==7.2.2"]
NEEDED = {"train": ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"],
          "test": ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]}
FULL_RUN_GB = 48
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------ machine
def total_ram_gb() -> float:
    if sys.platform.startswith("linux"):
        with open("/proc/meminfo") as f:
            return int(f.readline().split()[1]) / 2**20
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MemoryStatus()
        m.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 2**30
    out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout  # macOS
    return int(out) / 2**30 if out.strip() else 0.0


def gpu_info() -> str:
    if not shutil.which("nvidia-smi"):
        return "no nvidia-smi found"
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.stdout.strip().replace("\n", "; ") or r.stderr.strip()


def check_machine(a) -> None:
    ram, cores = total_ram_gb(), os.cpu_count() or 0
    a.work.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(a.work).free / 2**30
    log(f"machine: {platform.platform()}, Python {platform.python_version()}, {cores} logical cores, "
        f"{ram:.0f} GB RAM, {disk:.0f} GB free at {a.work}")
    log(f"GPU: {gpu_info()} (not used: the pipeline runs LightGBM on the CPU)")
    need_ram, need_disk = (FULL_RUN_GB, 100) if a.sample >= 1 else (8, 5)
    problems = []
    if ram < need_ram:
        problems.append(f"{ram:.0f} GB RAM, this run needs ~{need_ram} GB")
    if disk < need_disk:
        problems.append(f"{disk:.0f} GB free disk, this run needs ~{need_disk} GB")
    if problems:
        msg = "; ".join(problems)
        if not a.force:
            raise SystemExit(f"machine too small: {msg}. Try --sample 0.01 first, or pass --force to try anyway "
                             "(a large swap/page file can carry it, slowly).")
        log(f"WARNING {msg}; continuing because of --force")
    if sys.version_info[:2] != (3, 12):
        log("WARNING: Python 3.12 is recommended; the pinned versions were tested on it")


def watch_memory(every: int = 300) -> None:
    """Log used RAM every few minutes: a run killed for memory leaves this trail."""
    try:
        import psutil
    except ImportError:
        return
    while True:
        m = psutil.virtual_memory()
        log(f"mem used {(m.total - m.available) / 2**30:.1f} of {m.total / 2**30:.0f} GB")
        time.sleep(every)


# --------------------------------------------------------------------- data
def find_data(root: Path, work: Path) -> Path:
    """A folder with train/ and test/ holding the 7 TSVs: --data itself, or links/copies under <work>/data."""
    found = {}
    for p in root.rglob("*.tsv"):
        found.setdefault(p.name, p)
    missing = [n for names in NEEDED.values() for n in names if n not in found]
    if missing:
        raise SystemExit(f"not found under {root}: {', '.join(missing)}")
    if all((root / split / n).is_file() for split, names in NEEDED.items() for n in names):
        return root
    data = work / "data"
    for split, names in NEEDED.items():
        (data / split).mkdir(parents=True, exist_ok=True)
        for n in names:
            dst = data / split / n
            if dst.exists():
                continue
            try:
                os.link(found[n], dst)  # same drive: no copy
            except OSError:
                shutil.copy2(found[n], dst)
    return data


# ------------------------------------------------------------ code and deps
def install() -> None:
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    log("installing the pinned requirements")
    if subprocess.run(pip + PINS).returncode != 0:
        log("pinned install failed on this Python; installing unpinned versions")
        subprocess.run(pip + [p.split("==")[0] for p in PINS], check=True)


def get_code(a) -> Path:
    """The pipeline's src/ folder: from --code, else the branch downloaded from GitHub."""
    if a.code:
        hits = list(Path(a.code).rglob("run_pipeline.py"))
    else:
        z = a.work / "code.zip"
        url = f"{REPO}/archive/refs/heads/{a.branch}.zip"
        log(f"downloading {url}")
        urllib.request.urlretrieve(url, z)
        dest = a.work / "code"
        shutil.rmtree(dest, ignore_errors=True)
        keep = ("code/business_entity_resolution/", "resources/student_resource/utils/")
        with zipfile.ZipFile(z) as f:  # only the pipeline and the validator, without the zip's top folder:
            for m in f.infolist():     # short paths stay under Windows' 260-character limit
                rel = m.filename.split("/", 1)[-1]
                if m.is_dir() or not rel.startswith(keep):
                    continue
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(f.read(m))
        hits = list(dest.rglob("run_pipeline.py"))
    hits = [h for h in hits if h.parent.name == "src" and "business_entity_resolution" in str(h)]
    if not hits:
        raise SystemExit("run_pipeline.py not found in the code folder")
    return hits[0].parent


# ---------------------------------------------------------------------- run
def run(cmd: list[str], cwd: Path, log_file: Path) -> int:
    """Run and stream the output to the console and a log file."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with open(log_file, "w", encoding="utf-8") as f, subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", env=env) as p:
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()
        return p.wait()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def summary(metrics: dict) -> dict:
    ev = metrics.get("predict", {}).get("eval", {})
    best, test = ev.get("best", {}), metrics.get("predict", {}).get("test", {})
    block = metrics.get("block", {})
    return {"eval_f05": best.get("f05"), "eval_f05_doubled_distractors": best.get("f05_dup"),
            "precision": best.get("precision"), "recall": best.get("recall"),
            "rule": f"{best.get('score')} {best.get('rule')}={best.get('param')}", "chosen_by": ev.get("chosen_by"),
            "candidates_per_s1_test": block.get("test", {}).get("per_s1"),
            "blocking_recall_eval": block.get("train", {}).get("recall_eval", {}).get("pair_recall"),
            "test_s1_linked": (test.get("s1_with_matches", 0) / test["s1_total"]) if test.get("s1_total") else None,
            "test_links": test.get("links")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="folder holding the 7 challenge TSVs (any layout)")
    ap.add_argument("--work", default=Path("mlc26_work"), type=Path, help="cache of every stage (~60 GB for a full run)")
    ap.add_argument("--out", default=Path("mlc26_out"), type=Path, help="results: <out>/<preset>/")
    ap.add_argument("--presets", default="M-v11,M-v12", help="comma-separated presets, run in this order")
    ap.add_argument("--sample", type=float, default=1.0, help="share of the data; 0.01 = smoke run")
    ap.add_argument("--code", default=None, help="an existing checkout of the repository instead of downloading it")
    ap.add_argument("--branch", default=BRANCH)
    ap.add_argument("--no-install", action="store_true", help="skip pip install (packages already present)")
    ap.add_argument("--force", action="store_true", help="run even if the machine looks too small")
    a = ap.parse_args()
    a.work, a.out = a.work.resolve(), a.out.resolve()
    if a.sample < 1:  # a sample needs its own cache: prep refuses to mix data scopes
        a.work = a.work.with_name(f"{a.work.name}_sample{a.sample:g}")

    check_machine(a)
    threading.Thread(target=watch_memory, daemon=True).start()
    data = find_data(a.data.resolve(), a.work)
    log(f"data: {data}")
    if not a.no_install:
        install()
    src = get_code(a)
    log(f"code: {src}")
    validator = next(iter(src.parents[2].glob("resources/student_resource/utils/validate_submission.py")), None)

    sys.path.insert(0, str(src))
    from ber import versions as V  # the version registry of the downloaded code
    done, results = [], {}
    for preset in [p.strip() for p in a.presets.split(",") if p.strip()]:
        if preset not in V.PRESETS:
            raise SystemExit(f"unknown preset {preset}; known: {', '.join(sorted(V.PRESETS))}")
        shared = any(V.PRESETS[preset][:3] == V.PRESETS[d][:3] for d in done)
        stages = "train,predict" if shared else "prep,block,features,train,predict"
        out = a.out / (preset if a.sample >= 1 else f"{preset}_sample{a.sample:g}")
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "run_pipeline.py", "--data", str(data), "--work", str(a.work), "--out", str(out),
               "--preset", preset, "--sample", str(a.sample), "--stages", stages]
        if validator is not None:
            cmd += ["--validator", str(validator)]
        log(f"==== {preset} ({' + '.join(V.PRESETS[preset])}), stages {stages}")
        rc = run(cmd, src, out / "run.log")
        if rc != 0:
            log(f"{preset} failed (exit {rc}); its log: {out / 'run.log'}")
            return rc
        done.append(preset)
        rv = V.resolve(preset)
        metrics_file = a.work / "runs" / rv.key("train") / "metrics.json"
        shutil.copy2(metrics_file, out / "metrics.json")
        results[preset] = summary(json.loads(metrics_file.read_text(encoding="utf-8")))
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            if (out / f).exists():
                results[preset][f"md5_{f}"] = md5(out / f)
        log(f"{preset}: {json.dumps(results[preset], indent=1)}")

    (a.out / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    log("summary (eval = the held-out 20% of train entities in the test-like universe; higher is better):")
    for p, r in results.items():
        fmt = lambda v, d=4: "-" if v is None else f"{v:.{d}f}"
        log(f"  {p:6s} eval F0.5 {fmt(r['eval_f05'])}  doubled-distractor {fmt(r['eval_f05_doubled_distractors'])}  "
            f"candidates/S1 {fmt(r['candidates_per_s1_test'], 1)}  test S1 linked {fmt(r['test_s1_linked'], 3)}")
    if a.sample >= 1:
        log(f"submit <out>/<preset>/matching_results.tsv of the better run; its candidate_pairs.tsv goes in the package. "
            f"Results in {a.out}")
    else:
        log("smoke run done (the sample's test side is not representative); run again without --sample for the real files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
