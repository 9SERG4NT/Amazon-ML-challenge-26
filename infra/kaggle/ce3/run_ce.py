"""Kaggle GPU (2x T4): a larger multilingual cross-encoder's match probability for every M-v11 candidate pair (ce3).

Inputs (attached datasets):
  serg4nt/mlc26-ber-data   the challenge TSVs
  serg4nt/mlc26-ce-pairs   ids_train.parquet / ids_test.parquet from `experiments.cross_encoder export` on the
                           runner: M-v11's candidate pairs (entity and row ids), labels, roles and cross-fitting halves
Texts are the raw '<business_name> | <business_address>' of both records, so the multilingual model reads the
original scripts (Devanagari, French accents). Each GPU fine-tunes one half's model and scores its share: a fit pair
gets the score of the model that never saw its S1; early-stop, eval and test pairs are split between the two models
by S1 (q_rid parity), so every pair is scored once, by a model that never saw its S1.
Outputs /kaggle/working/ce2_{train,test}.parquet (q_rid, t_rid, ce2) for FEAT-v7 on the runner, plus ce2_eval.json.
The training/scoring code is experiments/cross_encoder.py of our repo, fetched at a fixed commit.

Memory (two workers share ~30 GB): the texts are prepared in a separate process that exits before training; training
tokenises batch by batch; scoring streams 500k-row batches and tokenises 100k pairs at a time; each worker writes its
scores chunk by chunk to /kaggle/working/parts, which Kaggle keeps even if a later step fails. (Run 1 held every text
in both workers and lost a worker while scoring; run 2 tokenised 1.2M training pairs up front in both and lost one
while training.)
"""
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

CONFIG = {"model": "intfloat/multilingual-e5-base",  # MIT, 278M parameters (ce2 used e5-small, 118M)
          "n_train": 1_200_000, "epochs": 1.0, "batch": 256, "lr": 3e-5, "max_len": 64, "score_batch": 512,
          "name": "ce3", "commit": "d347da9f58a2a927702d93341da660a673e1d9a5"}
CONFIG.update(json.loads(os.environ.get("CE_CONFIG", "{}")))  # overrides for a local test
SRC = ("https://raw.githubusercontent.com/9SERG4NT/Amazon-ML-challenge-26/{commit}/code/business_entity_resolution/src/"
       "experiments/cross_encoder.py")
TMP, OUT, INPUT = (os.environ.get("CE_TMP", "/kaggle/tmp"), os.environ.get("CE_OUT", "/kaggle/working"),
                   os.environ.get("CE_INPUT", "/kaggle/input"))
CONFIG.update(tmp=TMP, out=OUT, input=INPUT)
T0 = time.time()

# the texts of every pair, in a process of its own (its memory is returned when it exits)
PREP = r'''
import glob, json, subprocess, sys, time
import polars as pl
c = json.loads(sys.argv[1])
TMP, INPUT = c["tmp"], c["input"]
def find(name):
    hits = sorted(glob.glob(f"{INPUT}/**/{name}", recursive=True) + glob.glob(f"{TMP}/unzipped/**/{name}", recursive=True))
    if not hits:
        raise SystemExit(f"{name} not found under {INPUT}: attach serg4nt/mlc26-ber-data and serg4nt/mlc26-ce-pairs")
    return hits[0]
for z in glob.glob(f"{INPUT}/**/*.zip", recursive=True):
    subprocess.run(["unzip", "-q", "-o", z, "-d", f"{TMP}/unzipped"], check=False)
text = (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null(""))
read = lambda f: pl.read_csv(find(f), separator="\t", quote_char=None, columns=["entity_id", "business_name", "business_address"],
                             schema_overrides={"business_name": pl.Utf8, "business_address": pl.Utf8})
for split in ("train", "test"):
    ids = pl.read_parquet(find(f"ids_{split}.parquet"))
    q = read(f"{split}_source1.tsv").select(pl.col("entity_id").alias("s1_id"), text.alias("qtext"))
    t = pl.concat([read(f"{split}_source{k}.tsv") for k in (2, 3)]).select(pl.col("entity_id").alias("cand_id"), text.alias("ttext"))
    D = ids.join(q, on="s1_id", how="left").join(t, on="cand_id", how="left")
    miss = D["qtext"].null_count() + D["ttext"].null_count()
    D.select("q_rid", "t_rid", "qtext", "ttext", "label", "role", "half").write_parquet(f"{TMP}/{split}.parquet", row_group_size=250_000)
    print(f"{split}: {D.height:,} pairs, {miss} texts missing; by half {D.group_by('half').len().sort('half').rows()}", flush=True)
'''

# one GPU's job, run as its own process: python ce_worker.py <half> '<CONFIG json>'
WORKER = r'''
import json, os, sys, time
import numpy as np, polars as pl, pyarrow.parquet as pq
k, c = int(sys.argv[1]), json.loads(sys.argv[2])
TMP, OUT = c["tmp"], c["out"]
sys.path.insert(0, TMP)
import cross_encoder as CE
T = f"{TMP}/train.parquet"
es = pl.scan_parquet(T).filter(pl.col("role") == CE.ES).select("qtext", "ttext", "label").collect()
es = es.sample(n=min(30_000, es.height), seed=1)
tr = pl.scan_parquet(T).filter(pl.col("half") == k).select("qtext", "ttext", "label").collect()
tr = tr.sample(n=min(c["n_train"], tr.height), seed=k)
CE.log(f"worker {k}: training {c['model']} on {tr.height:,} pairs ({tr['label'].mean():.3f} true)")
model, tok = CE.train_model(tr["qtext"].to_list(), tr["ttext"].to_list(), tr["label"].to_numpy(), c["model"],
                            epochs=c["epochs"], batch=c["batch"], lr=c["lr"], max_len=c["max_len"], device="cuda",
                            half=True, seed=k, val=(es["qtext"].to_list(), es["ttext"].to_list(), es["label"].to_numpy()))
del tr, es
model.save_pretrained(f"{OUT}/{c['name']}_fold{k}")
tok.save_pretrained(f"{OUT}/{c['name']}_fold{k}")
kw = dict(batch=c["score_batch"], max_len=c["max_len"], device="cuda", half=True, chunk=100_000)
os.makedirs(f"{OUT}/parts", exist_ok=True)
for split in ("train", "test"):
    # the other half's fit pairs, and the unseen pairs (early stop, eval, test) of S1 rows with q_rid % 2 == k
    n = 0
    for bi, b in enumerate(pq.ParquetFile(f"{TMP}/{split}.parquet").iter_batches(batch_size=500_000,
                                                                                columns=["q_rid", "t_rid", "qtext", "ttext", "half"])):
        df = pl.from_arrow(b)
        sub = df.filter((pl.col("half") == 1 - k) | ((pl.col("half") < 0) & ((pl.col("q_rid") % 2) == k)))
        del df
        if sub.height:
            p = CE.score(model, tok, sub["qtext"].to_list(), sub["ttext"].to_list(), **kw)
            sub.select("q_rid", "t_rid").with_columns(pl.Series("ce", p)).write_parquet(f"{OUT}/parts/w{k}_{split}_{bi:04d}.parquet")
            n += sub.height
        if bi % 4 == 0:
            CE.log(f"worker {k}: {split} batch {bi}, {n:,} pairs scored")
    CE.log(f"worker {k}: {split} done, {n:,} pairs")
CE.log(f"worker {k}: done")
'''


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def combine():
    import numpy as np
    import polars as pl
    c, res = CONFIG, {}
    for split in ("train", "test"):
        S = pl.read_parquet(f"{TMP}/{split}.parquet", columns=["q_rid", "t_rid", "label", "role", "half"])
        got = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{OUT}/parts/w[01]_{split}_*.parquet"))])
        assert got.height == S.height and got.select("q_rid", "t_rid").n_unique() == S.height, \
            f"{split}: {got.height:,} scores for {S.height:,} pairs"
        S = S.join(got.rename({"ce": c["name"]}), on=["q_rid", "t_rid"], how="left", maintain_order="left")
        S.select("q_rid", "t_rid", c["name"]).write_parquet(f"{OUT}/{c['name']}_{split}.parquet")
        if split == "train":
            ev = (S["role"] == 2).to_numpy()
            y, q = S["label"].to_numpy()[ev], S[c["name"]].to_numpy()[ev]
            res = {"eval_pairs": int(ev.sum()),
                   "eval_logloss": float(-np.mean(y * np.log(np.clip(q, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-6, 1)))),
                   "eval_accuracy": float(np.mean((q >= 0.5) == y)),
                   "half_counts": {int(h): int((S["half"] == h).sum()) for h in (-1, 0, 1)}, **CONFIG}
            log(f"eval pairs: {res}")
        log(f"wrote {OUT}/{c['name']}_{split}.parquet ({S.height:,} pairs)")
    json.dump(res, open(f"{OUT}/{c['name']}_eval.json", "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    try:  # the image's polars may be missing or too old for schema_overrides
        import polars
        assert int(polars.__version__.split(".")[0]) >= 1
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars>=1.0"], check=False)
    urllib.request.urlretrieve(SRC.format(commit=CONFIG["commit"]), f"{TMP}/cross_encoder.py")
    open(f"{TMP}/ce_prep.py", "w").write(PREP)
    open(f"{TMP}/ce_worker.py", "w").write(WORKER)
    subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; free -g | head -2", shell=True)
    rc = subprocess.run([sys.executable, "-u", f"{TMP}/ce_prep.py", json.dumps(CONFIG)]).returncode
    if rc:
        raise SystemExit(f"prepare failed: {rc}")
    log("texts prepared")
    import torch
    n_gpu = torch.cuda.device_count()
    log(f"{n_gpu} GPUs")
    cmd = lambda k: [sys.executable, "-u", f"{TMP}/ce_worker.py", str(k), json.dumps(CONFIG)]
    if n_gpu >= 2:  # one half per GPU, in parallel
        procs = [subprocess.Popen(cmd(k), env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k)}) for k in (0, 1)]
        codes = [p.wait() for p in procs]
    else:
        codes = [subprocess.run(cmd(k)).returncode for k in (0, 1)]
    if any(codes):
        raise SystemExit(f"worker failed: {codes}")
    combine()
    log("done")
