"""A cross-encoder's match probability for every candidate pair, as a feature (FEAT-v6's ce1).

The team pipeline (leaderboard 0.98 against our 0.970) gets 75% of its XGBoost gain from a cross-encoder: a
transformer that reads both records together and scores the pair. This script trains one on our candidate pairs and
scores all of them, cross-fitted so the LightGBM never sees a score from a model that trained on the same S1 entity:

  * the fit and rest entities are split into two halves by a hash of the S1 row id;
  * model A trains on half 0, model B on half 1 (a sample of their candidate pairs, labels from the training links);
  * pairs of half 0 get B's score, pairs of half 1 get A's; early-stop, eval and test pairs, whose entities neither
    model saw, get A's score (or the mean of both with --mean-both).

Usage (from src/):
    python -m experiments.cross_encoder export <work> NORM-v2__BLK-v5-tlu40
    python -m experiments.cross_encoder run <work> NORM-v2__BLK-v5-tlu40 --name ce1 [--model ...] [--n-train ...]
    python -m experiments.cross_encoder bench --model ...

``export`` writes <work>/ce/<key>/{train,test}.parquet (pair ids, '<name> | <address>' texts of both records from the
normalised prep tables, label, half) and ids_{train,test}.parquet (entity ids, for a run elsewhere, e.g. a Kaggle GPU).
``run`` writes <work>/extra/<key>/<name>_{train,test}.parquet (q_rid, t_rid, <name>), which FEAT-v6 joins, and saves
the two models under <work>/ce/<key>/<name>_fold{0,1}/. Models: MIT/Apache-2.0 checkpoints only (default
google/bert_uncased_L-4_H-256_A-4, Apache-2.0, 11M parameters: fast enough for 18M pairs on 8 CPU cores).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl

T0 = time.time()
FIT, ES, EVAL, REST = 0, 1, 2, 3
DEFAULT_MODEL = "google/bert_uncased_L-4_H-256_A-4"


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ export (EC2)
def export(work: Path, key: str) -> None:
    prep = work / "prep" / key.split("__")[0]
    out = work / "ce" / key
    out.mkdir(parents=True, exist_ok=True)
    roles = pl.read_parquet(prep / "split.parquet")
    links = pl.read_parquet(prep / "pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("label"))
    text = lambda: (pl.col("name_n").fill_null("") + " | " + pl.col("addr_n").fill_null(""))
    for split in ("train", "test"):
        C = pl.read_parquet(work / "block" / key / f"cand_{split}.parquet", columns=["q_rid", "t_rid"])
        Q = pl.read_parquet(prep / f"Q_{split}.parquet", columns=["rid", "entity_id", "name_n", "addr_n"])
        T = pl.read_parquet(prep / f"T_{split}.parquet", columns=["rid", "entity_id", "name_n", "addr_n"])
        D = (C.join(Q.select(pl.col("rid").alias("q_rid"), pl.col("entity_id").alias("s1_id"), text().alias("qtext")), on="q_rid", how="left")
             .join(T.select(pl.col("rid").alias("t_rid"), pl.col("entity_id").alias("cand_id"), text().alias("ttext")), on="t_rid", how="left"))
        if split == "train":
            D = (D.join(links, on=["q_rid", "t_rid"], how="left").with_columns(pl.col("label").fill_null(0))
                 .join(roles, on="q_rid", how="left")
                 .with_columns(pl.when(pl.col("role").is_in([FIT, REST])).then((pl.col("q_rid").hash(seed=7) % 2).cast(pl.Int8))
                               .otherwise(pl.lit(-1, pl.Int8)).alias("half")))
        else:
            D = D.with_columns(pl.lit(0, pl.Int8).alias("label"), pl.lit(-1, pl.Int8).alias("role"), pl.lit(-1, pl.Int8).alias("half"))
        D = D.sort("q_rid")
        D.select("q_rid", "t_rid", "qtext", "ttext", "label", "role", "half").write_parquet(out / f"{split}.parquet", compression="zstd")
        D.select("s1_id", "cand_id", "q_rid", "t_rid", "label", "role", "half").write_parquet(out / f"ids_{split}.parquet", compression="zstd")
        by = D.group_by("half").agg(pl.len(), pl.col("label").mean().round(3).alias("pos")).sort("half")
        log(f"export {split}: {D.height:,} pairs; by half {by.rows()}")


# ------------------------------------------------------------------------------------------------ model
def _device(name: str):
    import torch
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _autocast(device, half: bool):
    import torch
    if not half:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16)


def _batches(tok, enc, order: np.ndarray, batch: int):
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        yield idx, tok.pad({k: [enc[k][i] for i in idx] for k in enc.keys()}, return_tensors="pt")


def train_model(qa: list[str], tb: list[str], y: np.ndarray, model_name: str, *, epochs: float = 1.0, batch: int = 128,
                lr: float = 1e-4, max_len: int = 64, device="auto", half: bool = False, seed: int = 0,
                val: tuple | None = None):
    """Fine-tune ``model_name`` as a one-logit pair classifier with binary cross-entropy. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(seed)
    dev = _device(device) if isinstance(device, str) else device
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(dev)
    enc = tok(qa, tb, truncation="longest_first", max_length=max_len)
    n = len(qa)
    steps = int(math.ceil(n * epochs / batch))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=half and dev.type == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    yt = torch.tensor(y, dtype=torch.float32)
    model.train()
    done, t0, run = 0, time.time(), 0.0
    while done < steps:
        order = rng.permutation(n)
        for idx, feats in _batches(tok, enc, order, batch):
            with _autocast(dev, half):
                logits = model(**{k: v.to(dev) for k, v in feats.items()}).logits.float().squeeze(-1)
            loss = lossf(logits, yt[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run = 0.98 * run + 0.02 * loss.item() if done else loss.item()
            done += 1
            if done % 500 == 0 or done == steps:
                log(f"  step {done:,}/{steps:,}: loss {run:.4f} ({done * batch / (time.time() - t0):.0f} pairs/s)")
            if done >= steps:
                break
    model.eval()
    if val is not None:
        p = score(model, tok, val[0], val[1], max_len=max_len, device=dev, half=half)
        vy = val[2]
        ll = -np.mean(vy * np.log(np.clip(p, 1e-6, 1)) + (1 - vy) * np.log(np.clip(1 - p, 1e-6, 1)))
        log(f"  held-out pairs: log loss {ll:.4f}, accuracy {np.mean((p >= 0.5) == vy):.4f} ({len(vy):,} pairs)")
    return model, tok


def score(model, tok, qa: list[str], tb: list[str], *, batch: int = 512, max_len: int = 64, device="auto",
          half: bool = False, chunk: int = 200_000) -> np.ndarray:
    """Match probability of every pair, in input order (pairs sorted by length within chunks: less padding)."""
    import torch
    dev = _device(device) if isinstance(device, str) else device
    out = np.empty(len(qa), np.float32)
    t0 = time.time()
    for s in range(0, len(qa), chunk):
        enc = tok(qa[s:s + chunk], tb[s:s + chunk], truncation="longest_first", max_length=max_len)
        order = np.argsort([len(x) for x in enc["input_ids"]], kind="stable")
        for idx, feats in _batches(tok, enc, order, batch):
            with torch.inference_mode(), _autocast(dev, half):
                logits = model(**{k: v.to(dev) for k, v in feats.items()}).logits.float().squeeze(-1)
            out[s + idx] = torch.sigmoid(logits).cpu().numpy()
        if len(qa) > chunk:
            done = min(s + chunk, len(qa))
            log(f"  scored {done:,}/{len(qa):,} ({done / (time.time() - t0):.0f} pairs/s)")
    return out


# ------------------------------------------------------------------------------------------------ run
def run(work: Path, key: str, a) -> None:
    import torch
    torch.set_num_threads(a.threads) if a.threads else None
    src = work / "ce" / key
    dst = work / "extra" / key
    dst.mkdir(parents=True, exist_ok=True)
    D = pl.read_parquet(src / "train.parquet")
    val = D.filter(pl.col("role") == ES).sample(n=min(a.n_val, D.filter(pl.col("role") == ES).height), seed=1)
    val = (val["qtext"].to_list(), val["ttext"].to_list(), val["label"].to_numpy())
    models = {}
    for h in (0, 1):
        tr = D.filter(pl.col("half") == h)
        tr = tr.sample(n=min(a.n_train, tr.height), seed=h)
        log(f"half {h}: training {a.model} on {tr.height:,} pairs ({tr['label'].mean():.3f} true)")
        models[h] = train_model(tr["qtext"].to_list(), tr["ttext"].to_list(), tr["label"].to_numpy(), a.model,
                                epochs=a.epochs, batch=a.batch, lr=a.lr, max_len=a.max_len, device=a.device,
                                half=a.half, seed=h, val=val)
        models[h][0].save_pretrained(src / f"{a.name}_fold{h}")
        models[h][1].save_pretrained(src / f"{a.name}_fold{h}")
    kw = dict(batch=a.score_batch, max_len=a.max_len, device=a.device, half=a.half)
    for split in ("train", "test"):
        S = D if split == "train" else pl.read_parquet(src / "test.parquet")
        p = np.empty(S.height, np.float32)
        half = S["half"].to_numpy()
        for h, by in ((0, 1), (1, 0)):  # a fit pair is scored by the model of the other half
            m = half == h
            if m.any():
                log(f"{split}: scoring {int(m.sum()):,} pairs of half {h} with the half-{by} model")
                p[m] = score(*models[by], S.filter(pl.Series(m))["qtext"].to_list(), S.filter(pl.Series(m))["ttext"].to_list(), **kw)
        m = half < 0
        if m.any():
            sub = S.filter(pl.Series(m))
            log(f"{split}: scoring {int(m.sum()):,} pairs no model saw" + (" with both models" if a.mean_both else ""))
            p[m] = score(*models[0], sub["qtext"].to_list(), sub["ttext"].to_list(), **kw)
            if a.mean_both:
                p[m] = (p[m] + score(*models[1], sub["qtext"].to_list(), sub["ttext"].to_list(), **kw)) / 2
        out = S.select("q_rid", "t_rid").with_columns(pl.Series(a.name, p))
        out.write_parquet(dst / f"{a.name}_{split}.parquet")
        if split == "train":
            ev = S.with_columns(pl.Series("p", p)).filter(pl.col("role") == EVAL)
            y, q = ev["label"].to_numpy(), ev["p"].to_numpy()
            ll = -np.mean(y * np.log(np.clip(q, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-6, 1)))
            log(f"eval pairs ({ev.height:,}): log loss {ll:.4f}, accuracy {np.mean((q >= 0.5) == y):.4f}")
            (src / f"{a.name}_eval.json").write_text(json.dumps({"eval_logloss": float(ll), "eval_pairs": ev.height,
                                                                "model": a.model, "n_train": a.n_train}))
        log(f"{split}: wrote {dst / f'{a.name}_{split}.parquet'} ({S.height:,} pairs)")


def bench(a) -> None:
    """Pairs per second for training and scoring on this machine (random business-like strings)."""
    import torch
    torch.set_num_threads(a.threads) if a.threads else None
    rng = np.random.default_rng(0)
    words = ["sri", "balaji", "traders", "pvt", "ltd", "main", "rd", "rue", "de", "la", "paris", "street", "suite",
             "123", "45b", "cafe", "holding", "sarl", "inc", "llc", "texas", "nagar", "colony", "avenue", "chennai"]
    mk = lambda k: [" ".join(rng.choice(words, rng.integers(4, 12))) + " | " + " ".join(rng.choice(words, rng.integers(3, 9)))
                    for _ in range(k)]
    qa, tb, y = mk(a.bench_n), mk(a.bench_n), rng.integers(0, 2, a.bench_n)
    t = time.time()
    model, tok = train_model(qa, tb, y, a.model, epochs=1, batch=a.batch, max_len=a.max_len, device=a.device, half=a.half)
    log(f"train: {a.bench_n / (time.time() - t):.0f} pairs/s (batch {a.batch})")
    t = time.time()
    score(model, tok, qa * 4, tb * 4, batch=a.score_batch, max_len=a.max_len, device=a.device, half=a.half)
    log(f"score: {4 * a.bench_n / (time.time() - t):.0f} pairs/s (batch {a.score_batch})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["export", "run", "bench"])
    ap.add_argument("work", nargs="?")
    ap.add_argument("key", nargs="?")
    ap.add_argument("--name", default="ce1")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-train", type=int, default=600_000, help="training pairs per half")
    ap.add_argument("--n-val", type=int, default=30_000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--score-batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--half", action="store_true", help="bf16 autocast on CPU, fp16 on CUDA")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--mean-both", action="store_true")
    ap.add_argument("--bench-n", type=int, default=2048)
    a = ap.parse_args()
    if a.cmd == "export":
        export(Path(a.work), a.key)
    elif a.cmd == "run":
        run(Path(a.work), a.key, a)
    else:
        bench(a)


if __name__ == "__main__":
    main()
