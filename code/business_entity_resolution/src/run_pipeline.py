"""End-to-end pipeline: data -> normalise -> block -> features -> LightGBM -> decision -> output.

Usage (from src/):
    python run_pipeline.py --data <dataset_dir> --work <work_dir> --out <output_dir>   # preset M-v3
    python run_pipeline.py ... --preset M-v4                    # another end-to-end version
    python run_pipeline.py ... --feat FEAT-v3 --stages features,train,predict   # swap one component
    python run_pipeline.py ... --sample 0.01                    # quick end-to-end smoke run
    python run_pipeline.py --list                               # every registered version

Every component comes from a named version in ber/versions.py (NORM, BLK, FEAT, MATCH; the
IDs of method_result.md), and each stage caches its output under the versions it depends on:
  <work>/prep/NORM-v2/                               normalised records, links, split, aliases
  <work>/block/NORM-v2__BLK-v4b@20/                  candidates, region spec
  <work>/feat/NORM-v2__BLK-v4b@20__FEAT-v2/          features, in parts
  <work>/runs/NORM-v2__BLK-v4b@20__FEAT-v2__MATCH-v2/  models, scores, metrics.json
so changing one version recomputes only what depends on it. One work dir holds one data
scope (--sample/--seed/--fit/--es/--eval); prep refuses to be reused with another.

Validation. Train S1 entities are split once (seed 42) into fit / early-stopping / eval /
rest. Every entity stays in the candidate search, so targets are contested as densely as
at test time. Aliases and region merges are learned without the eval entities' links, no
model sees eval or rest entities, and those rows are scored exactly like test rows (the
mean of the fold models), so the eval score estimates the test score.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from ber import model as M
from ber import versions as V
from ber.aliases import learn_aliases
from ber.blocking import (candidates_at_k, feature_frame, generate_candidates, learn_region_merges, noaddr_recall_at_k,
                          recall_at_k, recall_report)
from ber.data import dev_sample, load_split, load_truth
from ber.features import (align_features, context_features, distinctive, frequent_tokens, name_idf, string_features,
                          support_features)
from ber.metrics import macro_f05
from ber.normalize import normalize, token_frames
from ber.partition import country_codes

FIT, ES, EVAL, REST = 0, 1, 2, 3
ROLE_NAMES = {FIT: "fit", ES: "early_stop", EVAL: "eval", REST: "rest"}
STAGES = ("prep", "block", "features", "train", "predict")
PARAMS = {"threshold": np.round(np.arange(0.20, 0.96, 0.025), 3), "expected_f": (0.05, 0.1, 0.2, 0.3, 0.4),
          "gated_ef": np.round(np.arange(0.30, 0.96, 0.05), 3)}

_T0 = time.time()


def peak_gb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # Linux: KiB
    except ImportError:  # Windows
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**30


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.0f}s  peak {peak_gb():5.1f} GB] {msg}", flush=True)


class Paths:
    """Stage directories named by the versions each stage depends on."""

    def __init__(self, work: Path, rv: V.RunVersions):
        self.prep = work / "prep" / rv.key("prep")
        self.block = work / "block" / rv.key("block")
        self.feat = work / "feat" / rv.key("features")
        self.run = work / "runs" / rv.key("train")

    def stage_dir(self, stage: str) -> Path:
        return {"prep": self.prep, "block": self.block, "features": self.feat}.get(stage, self.run)


def read_json(f: Path) -> dict:
    return json.loads(f.read_text()) if f.exists() else {}


def write_json(f: Path, d: dict) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d, indent=2, default=float))


def data_scope(a) -> dict:
    return {"sample": a.sample, "seed": a.seed, "fit": a.fit, "es": a.es, "eval": a.eval}


def check_scope(a, P: Paths) -> None:
    cached = read_json(P.prep / "prep.json").get("scope")
    if cached is None:
        raise SystemExit(f"{P.prep} has no prep output yet: run the prep stage first.")
    if cached != data_scope(a):
        raise SystemExit(f"{P.prep} was built with {cached}, not {data_scope(a)}. Use another --work dir.")


def learning_pairs(P: Paths) -> pl.DataFrame:
    """True links of the non-eval train entities: the only links aliases and region merges may learn from."""
    split = pl.read_parquet(P.prep / "split.parquet")
    return pl.read_parquet(P.prep / "pairs.parquet").join(
        split.filter(pl.col("role") != EVAL).select("q_rid"), on="q_rid", how="semi")


def train_universe(a, bv: V.Block, P: Paths) -> tuple[np.ndarray, np.ndarray] | None:
    """Test-like train universe: (kept S1 rids, kept target rids) as boolean masks, or None for the full one.

    Every eval S1 stays, so the eval slice is the same 441,521 entities in every universe; a share
    ``keep_nonevals`` of the other S1 entities stays too, and the rest leave together with their true
    targets. The distractors all stay, so there are more of them per S1, as in the test set.
    """
    if bv.keep_nonevals >= 1:
        return None
    role = pl.read_parquet(P.prep / "split.parquet")["role"].to_numpy()
    u = np.random.default_rng(a.seed + 2).random(len(role))  # its own draw: the split and folds are untouched
    keep_q = (role == EVAL) | (u < bv.keep_nonevals)
    pairs = pl.read_parquet(P.prep / "pairs.parquet")
    n_t = pl.read_parquet(P.prep / "T_train.parquet", columns=["rid"]).height
    keep_t = np.ones(n_t, bool)
    keep_t[pairs["t_rid"].to_numpy()[~keep_q[pairs["q_rid"].to_numpy()]]] = False
    return keep_q, keep_t


def distractor_copies(T: pl.DataFrame, pairs: pl.DataFrame, copies: int, n_all: int) -> pl.DataFrame:
    """(rid, copy_rid) for the ``copies - 1`` extra copies of every distractor in T (a target no train S1 links to).

    Copy r of a record gets rid + r * n_all (n_all = train targets in prep), so ids never collide; every
    lookup by rid is a join, so the ids need not be dense.
    """
    d = T.select("rid").join(pairs.select(pl.col("t_rid").alias("rid")), on="rid", how="anti")
    return pl.concat([d.select("rid", (pl.col("rid").cast(pl.UInt64) + r * n_all).cast(pl.UInt32).alias("copy_rid"))
                      for r in range(1, copies)])


def train_frames(a, bv: V.Block, P: Paths, with_copies: bool) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
    """The train S1 and target records of the block version's universe, and its distractor copies (or None).

    ``with_copies`` appends the copies to the targets (features stage). Blocking searches the originals
    and copies their candidate rows afterwards (``copy_candidates``): the same lists, less memory.
    """
    Q = pl.read_parquet(P.prep / "Q_train.parquet")
    T = pl.read_parquet(P.prep / "T_train.parquet")
    n_all = T.height
    universe = train_universe(a, bv, P)
    if universe is not None:
        Q, T = Q.filter(pl.Series(universe[0])), T.filter(pl.Series(universe[1]))
    if bv.dup_distractors <= 1:
        return Q, T, None
    copies = distractor_copies(T, pl.read_parquet(P.prep / "pairs.parquet"), bv.dup_distractors, n_all)
    if with_copies:
        T = pl.concat([T, T.join(copies, on="rid").with_columns(pl.col("copy_rid").alias("rid")).select(T.columns)])
    return Q, T, copies


def copy_candidates(C: pl.DataFrame, copies: pl.DataFrame, k: int, k_noaddr: int) -> pl.DataFrame:
    """Add a candidate row for every copy of a candidate distractor, then cut each list back to its top k.

    A copy ties with its original, so in a search over the doubled targets it would take the next slot:
    the lists end up as if the copies had been searched (originals win ties at the cut).
    """
    extra = (C.join(copies.rename({"rid": "t_rid"}), on="t_rid")
             .with_columns(pl.col("copy_rid").alias("t_rid")).select(C.columns))
    r = pl.col("bscore").rank("ordinal", descending=True).over("q_rid", "src", "noaddr_pass")
    return (pl.concat([C, extra]).filter(r <= pl.when(pl.col("noaddr_pass")).then(k_noaddr).otherwise(k))
            .sort("q_rid", "src", "bscore", descending=[False, False, True], maintain_order=True))


def universe_report(Q: pl.DataFrame, T: pl.DataFrame, pairs: pl.DataFrame, copies: int = 1) -> dict:
    """Per country: S1, targets and distractors per S1 in a train universe, copies included (test: ~5.8 and ~2.3).

    Q and T hold the universe's original records; a distractor is a target no S1 of the universe links to.
    """
    linked = pairs.join(Q.select(pl.col("rid").alias("q_rid")), on="q_rid", how="semi").select(pl.col("t_rid").alias("rid"))
    nd_c = dict(T.join(linked, on="rid", how="anti").group_by("country").len().rows())
    nt_c, nq_c = dict(T.group_by("country").len().rows()), dict(Q.group_by("country").len().rows())
    out = {}
    for c in sorted(nq_c):
        nq, nd = nq_c[c], nd_c.get(c, 0) * copies
        nt = nt_c.get(c, 0) + nd_c.get(c, 0) * (copies - 1)
        out[str(c)] = {"s1": nq, "targets": nt, "targets_per_s1": round(nt / nq, 2), "distractors_per_s1": round(nd / nq, 2)}
    return out


# ---------------------------------------------------------------------- prep
def stage_prep(a, rv: V.RunVersions, P: Paths) -> dict:
    nv = rv.norm
    tr_s1, tr_t = load_split(a.data, "train")
    te_s1, te_t = load_split(a.data, "test")
    truth = load_truth(a.data)
    if a.sample < 1:
        tr_s1, tr_t, truth = dev_sample(tr_s1, tr_t, truth, a.sample, seed=a.seed)
        te_s1 = te_s1.sample(fraction=a.sample, seed=a.seed)
        te_t = te_t.sample(fraction=a.sample, seed=a.seed)
    Q, T = tr_s1.with_row_index("rid"), tr_t.with_row_index("rid")
    Qt, Tt = te_s1.with_row_index("rid"), te_t.with_row_index("rid")
    pairs = (truth.join(Q.select(pl.col("entity_id").alias("s1_id"), pl.col("rid").alias("q_rid")), on="s1_id")
             .join(T.select(pl.col("entity_id").alias("t_id"), pl.col("rid").alias("t_rid")), on="t_id")
             .select("q_rid", "t_rid"))
    log(f"train S1 {Q.height:,} targets {T.height:,} links {pairs.height:,} | "
        f"test S1 {Qt.height:,} targets {Tt.height:,}")

    # one fixed split of the train S1 entities
    u = np.random.default_rng(a.seed).random(Q.height)
    role = np.full(Q.height, REST, np.int8)
    role[u < a.fit] = FIT
    role[(u >= a.fit) & (u < a.fit + a.es)] = ES
    role[u >= 1 - a.eval] = EVAL
    split = pl.DataFrame({"q_rid": np.arange(Q.height, dtype=np.uint32), "role": role})
    log("split: " + ", ".join(f"{ROLE_NAMES[r]} {int((role == r).sum()):,}" for r in ROLE_NAMES))

    learn = pairs.join(split.filter(pl.col("role") != EVAL).select("q_rid"), on="q_rid", how="semi")
    _, nq, aq = token_frames(Q)
    _, nt, at = token_frames(T)
    kw = dict(min_count=nv.alias_min_count, min_ratio=nv.alias_min_ratio, max_unaligned=nv.alias_max_unaligned)
    name_alias = learn_aliases(nq, nt, learn, **kw).select("country", "tok", "to")
    addr_alias = learn_aliases(aq, at, learn, **kw).select("country", "tok", "to")
    del nq, aq, nt, at
    log(f"aliases: {name_alias.height} name, {addr_alias.height} address")

    P.prep.mkdir(parents=True, exist_ok=True)
    for name, df in (("Q_train", Q), ("T_train", T), ("Q_test", Qt), ("T_test", Tt)):
        normalize(df, name_alias, addr_alias).drop("business_name", "business_address").write_parquet(
            P.prep / f"{name}.parquet")
        log(f"normalised {name}")
    pairs.write_parquet(P.prep / "pairs.parquet")
    split.write_parquet(P.prep / "split.parquet")
    name_alias.write_parquet(P.prep / "name_alias.parquet")
    addr_alias.write_parquet(P.prep / "addr_alias.parquet")
    return {"scope": data_scope(a), "train_s1": Q.height, "train_targets": T.height, "train_links": pairs.height,
            "test_s1": Qt.height, "test_targets": Tt.height,
            "split": {ROLE_NAMES[r]: int((role == r).sum()) for r in ROLE_NAMES},
            "aliases": {"name": name_alias.height, "address": addr_alias.height}}


# --------------------------------------------------------------------- block
def stage_block(a, rv: V.RunVersions, P: Paths) -> dict:
    bv = rv.block
    spec = None
    if bv.region_split:
        spec = learn_region_merges(pl.read_parquet(P.prep / "Q_train.parquet"),
                                   pl.read_parquet(P.prep / "T_train.parquet"), learning_pairs(P),
                                   min_share=bv.region_min_share, min_cross=bv.region_min_cross,
                                   min_last_ratio=bv.region_min_last_ratio)
        log(f"region spec: {spec}")
    P.block.mkdir(parents=True, exist_ok=True)
    write_json(P.block / "region_spec.json", spec or {})
    out = {"region_spec": spec}
    for split in ("train", "test"):
        copies = None
        if split == "train":
            Q, T, copies = train_frames(a, bv, P, with_copies=False)
            if bv.keep_nonevals < 1 or copies is not None:
                out["universe"] = universe_report(Q, T, pl.read_parquet(P.prep / "pairs.parquet"), bv.dup_distractors)
                log(f"test-like universe: {out['universe']}")
        else:
            Q = pl.read_parquet(P.prep / f"Q_{split}.parquet")
            T = pl.read_parquet(P.prep / f"T_{split}.parquet")
        parts = []
        for country in Q["country"].unique().sort().to_list():  # one country at a time bounds memory
            Qc, Tc = Q.filter(pl.col("country") == country), T.filter(pl.col("country") == country)
            if Tc.height == 0:
                continue
            t0 = time.time()
            ff = lambda df: feature_frame(df, char_grams=bv.char_grams, joined_name=bv.joined_name)
            C = generate_candidates(Qc, Tc, ff(Qc), ff(Tc), k=bv.k, k_noaddr=bv.k_noaddr, max_df_frac=bv.max_df_frac,
                                    min_df_cap=bv.min_df_cap, region_spec=spec)
            parts.append(C)
            log(f"block {split}/{country}: S1 {Qc.height:,} x targets {Tc.height:,} -> "
                f"{C.height:,} candidates ({time.time() - t0:.0f}s)")
        C = pl.concat(parts).sort("q_rid", "src", "bscore", descending=[False, False, True])
        if copies is not None:
            n0 = C.height
            C = copy_candidates(C, copies, bv.k, bv.k_noaddr)
            out["distractor_copies"] = {"records": copies.height, "candidates_before": n0, "candidates_after": C.height}
            log(f"block {split}: {copies.height:,} distractor copies; candidates {n0:,} -> {C.height:,}")
        C.write_parquet(P.block / f"cand_{split}.parquet")
        log(f"block {split}: {C.height:,} candidates, {C.height / Q.height:.1f} per S1")
        out[split] = {"candidates": C.height, "per_s1": C.height / Q.height}
        if split == "train":
            pairs = pl.read_parquet(P.prep / "pairs.parquet").join(Q.select(pl.col("rid").alias("q_rid")), on="q_rid", how="semi")
            ev = pl.read_parquet(P.prep / "split.parquet").filter(pl.col("role") == EVAL).select("q_rid")
            pairs_ev = pairs.join(ev, on="q_rid", how="semi")
            out["train"]["recall_all"] = recall_report(C, pairs)
            out["train"]["recall_eval"] = recall_report(C.join(ev, on="q_rid", how="semi"), pairs_ev)
            Ce = C.join(ev, on="q_rid", how="semi")
            out["train"]["recall_at_k_eval"] = recall_at_k(Ce, pairs_ev)
            out["train"]["recall_at_k_noaddr_eval"] = noaddr_recall_at_k(Ce, pairs_ev)
            out["train"]["cand_per_s1_at_k"] = candidates_at_k(C, Q.height)
            out["train"]["cand_per_s1_at_k_noaddr"] = candidates_at_k(C, Q.height, noaddr=True)
            log(f"blocking recall: all {out['train']['recall_all']['pair_recall']:.4f}, "
                f"eval {out['train']['recall_eval']['pair_recall']:.4f}; eval by main-pass k: "
                + ", ".join(f"{k}:{v:.4f}" for k, v in out["train"]["recall_at_k_eval"].items())
                + "; by name-only k: " + ", ".join(f"{k}:{v:.4f}" for k, v in out["train"]["recall_at_k_noaddr_eval"].items()))
    return out


# ------------------------------------------------------------------ features
def stage_features(a, rv: V.RunVersions, P: Paths) -> dict:
    fv = rv.feat
    out = {}
    for split in ("train", "test"):
        d = P.feat / split
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("part-*.parquet"):
            old.unlink()
        if split == "train":  # name frequencies and IDF count only the records of the universe (copies included)
            Q, T, _ = train_frames(a, rv.block, P, with_copies=True)
        else:
            Q = pl.read_parquet(P.prep / f"Q_{split}.parquet")
            T = pl.read_parquet(P.prep / f"T_{split}.parquet")
        if fv.distinct:  # each country's frequent tokens, learned from this split's own records
            for col, red in (("name_core", "name_red"), ("addr_n", "addr_red")):
                frequent = frequent_tokens(Q, T, col)
                Q, T = (Q.with_columns(distinctive(Q, col, frequent).alias(red)),
                        T.with_columns(distinctive(T, col, frequent).alias(red)))
                out.setdefault("frequent_tokens", {}).setdefault(split, {})[col] = dict(
                    frequent.group_by("country").len().sort("country").rows())
            log(f"features {split}: frequent tokens per country {out['frequent_tokens'][split]}")
        C = pl.read_parquet(P.block / f"cand_{split}.parquet")
        # the context features need every candidate at once; the rest are row-wise
        ctx = context_features(C, Q, T, tfreq=fv.distinct)
        idf = name_idf(Q) if fv.align else None
        base = C.select("q_rid", "t_rid", pl.col("src").cast(pl.Float32), "bscore", "cos_name", "cos_addr",
                        pl.col("noaddr_pass").cast(pl.Float32))
        n, width = a.part_rows, 0
        for i, s in enumerate(range(0, C.height, n)):
            c = C.slice(s, n)
            # same columns, same order as ber.features.build_features
            cols = [base.slice(s, n), string_features(c, Q, T, number_gap=fv.number_gap, distinct=fv.distinct), ctx.slice(s, n)]
            if fv.align:
                cols.append(align_features(c, Q, T, idf))
            part = pl.concat(cols, how="horizontal")
            part.write_parquet(d / f"part-{i:04d}.parquet")
            width = part.width
            log(f"features {split} part {i}: rows {s:,}-{s + c.height:,}")
        out[split] = {"rows": C.height, "parts": -(-C.height // n), "features": width - 2}
    return out


# --------------------------------------------------------------------- train
def _parts(P: Paths, split: str) -> list[Path]:
    return sorted((P.feat / split).glob("part-*.parquet"))


def _labels(df: pl.DataFrame, pairs: pl.DataFrame) -> np.ndarray:
    return (df.select("q_rid", "t_rid")
            .join(pairs.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["q_rid", "t_rid"], how="left",
                  maintain_order="left")["y"].fill_null(0).to_numpy())


def _load_rows(parts: list[Path], keep_q: np.ndarray, cols: list[str], pairs: pl.DataFrame,
               extra: np.ndarray | None = None):
    """Feature rows whose S1 entity is flagged in ``keep_q``, in a fixed order.

    ``extra`` (one row per candidate, in candidate order) is appended as further columns.
    Returns X (float32), y, q_rid, and each row's index in the full candidate order.
    """
    masks, sizes = [], []
    for p in parts:
        q = pl.read_parquet(p, columns=["q_rid"])["q_rid"].to_numpy()
        masks.append(keep_q[q])
        sizes.append(len(q))
    n = int(sum(m.sum() for m in masks))
    width = len(cols) + (0 if extra is None else extra.shape[1])
    X = np.empty((n, width), np.float32)
    y = np.empty(n, np.int8)
    q = np.empty(n, np.uint32)
    rows = np.empty(n, np.int64)
    at = off = 0
    for p, m, size in zip(parts, masks, sizes):
        k = int(m.sum())
        if k:
            df = pl.read_parquet(p).filter(pl.Series(m))
            X[at:at + k, :len(cols)] = df.select(cols).to_numpy()
            y[at:at + k] = _labels(df, pairs)
            q[at:at + k] = df["q_rid"].to_numpy()
            rows[at:at + k] = off + np.flatnonzero(m)
            if extra is not None:
                X[at:at + k, len(cols):] = extra[rows[at:at + k]]
            at += k
        off += size
    return X, y, q, rows


def _predict(models, X: np.ndarray, q: np.ndarray, role: np.ndarray | None, fold: np.ndarray | None,
             margin: float | None = None) -> np.ndarray:
    """Fit rows get the fold model that did not train on their entity; every other row gets the mean.

    ``margin`` turns on LightGBM's prediction early stopping (Match.pred_margin).
    """
    kw = {} if margin is None else {"pred_early_stop": True, "pred_early_stop_freq": 10, "pred_early_stop_margin": margin}
    P = np.stack([m.predict(X, num_threads=0, **kw) for m in models]).astype(np.float32)
    out = P.mean(0)
    if role is not None:
        fit = np.flatnonzero(role[q] == FIT)
        out[fit] = P[fold[q[fit]], fit]
    return out


def _predict_parts(models, parts: list[Path], cols: list[str], role=None, fold=None, extra=None,
                   margin: float | None = None) -> np.ndarray:
    """Predict every candidate in order; ``extra`` holds columns appended to the part features."""
    out, off = [], 0
    for p in parts:
        df = pl.read_parquet(p, columns=["q_rid"] + cols)
        X = df.select(cols).to_numpy()
        if extra is not None:
            X = np.hstack([X, extra[off:off + len(X)]])
        out.append(_predict(models, X, df["q_rid"].to_numpy(), role, fold, margin))
        off += len(X)
    return np.concatenate(out)


def _cross_fit(X, y, q, role, fold, cols, mv: V.Match, tag: str, P: Paths, w: np.ndarray | None = None) -> list:
    is_fit, is_es = role[q] == FIT, role[q] == ES
    X_es, y_es = X[is_es], y[is_es]
    w_es = None if w is None else w[is_es]
    models = []
    train = M.train_arrays_xgb if mv.algo == "xgb" else M.train_arrays
    for f in range(mv.folds):
        tr = is_fit & (fold[q] != f)
        t0 = time.time()
        m = train(X[tr], y[tr], X_es, y_es, cols, params=mv.lgb_params(), threads=0,
                  w_tr=None if w is None else w[tr], w_va=w_es)
        m.save_model(str(P.run / f"{tag}_fold{f}.txt"))
        models.append(m)
        log(f"{tag} fold {f}: {int(tr.sum()):,} rows, best iteration {m.best_iteration} ({time.time() - t0:.0f}s)")
    return models


def _importance(models, top: int = 15) -> list:
    gain = sum(np.asarray(m.feature_importance("gain")) for m in models)
    order = np.argsort(-gain)[:top]
    names = models[0].feature_name()
    return [(names[i], round(float(gain[i] / gain.sum()), 4)) for i in order]


def _stage2_extra(C: pl.DataFrame, p1: np.ndarray, T: pl.DataFrame | None,
                  part: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Stage-2 inputs beyond the stage-1 features: p1 and its context, and optionally support features."""
    S = C.with_columns(pl.Series("p1", p1))
    blocks = [M.probability_context(S, part)]  # p1 + its context columns
    if T is not None:
        blocks.append(support_features(S, T))
    frame = pl.concat(blocks, how="horizontal")
    return frame.to_numpy().astype(np.float32, copy=False), frame.columns  # all columns are Float32 already


def _stage_frozen(a, rv: V.RunVersions, P: Paths) -> dict:
    """Score the train split with the fold models of run <NORM>__<frozen_from>: no training, no test scores.

    Fit rows get the fold model that did not see their entity (the source run's folds: same seed),
    every other row the mean, exactly as in the source run.
    """
    mv = rv.match
    src = P.run.parent / f"{rv.norm.id}__{mv.frozen_from}"
    role = pl.read_parquet(P.prep / "split.parquet")["role"].to_numpy()
    if mv.fit_rest:
        role = np.where(role == REST, FIT, role).astype(np.int8)
    fold = np.random.default_rng(a.seed + 1).integers(0, mv.folds, len(role)).astype(np.int8)
    tr_parts = _parts(P, "train")
    load = lambda tag: [lgb.Booster(model_file=str(src / f"{tag}_fold{f}.txt")) for f in range(mv.folds)]
    s1 = load("stage1")
    # the models' own columns, picked by name: a later FEAT version that only adds columns works too
    cols = s1[0].feature_name()
    missing = sorted(set(cols) - set(pl.read_parquet_schema(tr_parts[0])))
    if missing:
        raise SystemExit(f"{P.feat} lacks features the stage-1 models in {src} need: {missing}")
    p1 = _predict_parts(s1, tr_parts, cols, role, fold)
    log(f"stage 1 scored the train split with the models of {src.name}")
    C = pl.read_parquet(P.block / "cand_train.parquet", columns=["q_rid", "t_rid", "src"])
    scores = C.with_columns(pl.Series("p1", p1))
    if mv.stages == 2:
        text = (pl.read_parquet(P.prep / "T_train.parquet", columns=["rid", "name_core", "name_nosp", "addr_n"])
                if mv.support else None)
        part = country_codes(C["q_rid"].to_numpy(), pl.read_parquet(P.prep / "Q_train.parquet", columns=["rid", "country"]))
        extra, extra_cols = _stage2_extra(C, p1, text, part)
        s2 = load("stage2")
        if s2[0].feature_name() != cols + extra_cols:
            raise SystemExit(f"the stage-2 models in {src} expect other features")
        scores = scores.with_columns(pl.Series("p2", _predict_parts(s2, tr_parts, cols, role, fold, extra=extra)))
        log("stage 2 scored the train split")
    scores.write_parquet(P.run / "scores_train.parquet")
    return {"frozen_from": src.name, "rows": C.height}


def _stage_blend(rv: V.RunVersions, P: Paths) -> dict:
    """Average the stage-1/stage-2 probabilities of runs on the same blocking and features (same rows, same order)."""
    srcs = [P.run.parent / f"{rv.key('features')}__{m}" for m in rv.match.blend_of]
    out = {"blend_of": [s.name for s in srcs]}
    for split in ("train", "test"):
        frames = [pl.read_parquet(s / f"scores_{split}.parquet") for s in srcs]
        base = frames[0].select("q_rid", "t_rid", "src")
        for f in frames[1:]:
            if not f.select("q_rid", "t_rid").equals(base.select("q_rid", "t_rid")):
                raise SystemExit(f"{split} scores of {srcs} are not on the same candidate rows")
        cols = [c for c in ("p1", "p2") if all(c in f.columns for f in frames)]
        blended = base.with_columns([(sum(f[c] for f in frames) / len(frames)).cast(pl.Float32).alias(c) for c in cols])
        blended.write_parquet(P.run / f"scores_{split}.parquet")
        out[split] = {"rows": blended.height, "scores": cols}
    log(f"blended {', '.join(s.name for s in srcs)}")
    return out


def stage_train(a, rv: V.RunVersions, P: Paths) -> dict:
    mv = rv.match
    if mv.blend_of:
        P.run.mkdir(parents=True, exist_ok=True)
        return _stage_blend(rv, P)
    if mv.support and rv.block.dup_distractors > 1:
        raise SystemExit("support features read the prep targets, which lack the distractor copies of "
                         f"{rv.block.id}: use a matcher without support")
    P.run.mkdir(parents=True, exist_ok=True)
    if mv.frozen_from:
        return _stage_frozen(a, rv, P)
    role = pl.read_parquet(P.prep / "split.parquet")["role"].to_numpy()
    if mv.fit_rest:  # rest entities become fit entities: cross-fitted, out-of-fold predictions
        role = np.where(role == REST, FIT, role).astype(np.int8)
    fold = np.random.default_rng(a.seed + 1).integers(0, mv.folds, len(role)).astype(np.int8)
    pairs = pl.read_parquet(P.prep / "pairs.parquet")
    tr_parts, te_parts = _parts(P, "train"), _parts(P, "test")
    cols = [c for c in pl.read_parquet_schema(tr_parts[0]) if c not in M.ID_COLS]

    keep = np.isin(role, [FIT, ES])
    X, y, q, rows = _load_rows(tr_parts, keep, cols, pairs)
    log(f"training rows: {len(y):,} ({y.mean():.4f} positive), {len(cols)} features")
    w = None
    if mv.distractor_weight != 1:  # rows whose target no train S1 links to: the test has twice as many per S1
        t_rid = pl.read_parquet(P.block / "cand_train.parquet", columns=["t_rid"])["t_rid"].to_numpy()[rows]
        linked = np.zeros(int(max(t_rid.max(), pairs["t_rid"].max())) + 1, bool)
        linked[pairs["t_rid"].to_numpy()] = True
        w = np.where(linked[t_rid], 1.0, mv.distractor_weight).astype(np.float32)
        log(f"distractor rows weighted {mv.distractor_weight}: {(~linked[t_rid]).mean():.4f} of training rows")

    s1 = _cross_fit(X, y, q, role, fold, cols, mv, "stage1", P, w)
    del X  # stage 2 re-reads its rows, so the two matrices are never held together
    p1_tr = _predict_parts(s1, tr_parts, cols, role, fold, margin=mv.pred_margin)
    p1_te = _predict_parts(s1, te_parts, cols, margin=mv.pred_margin)
    log("stage 1 predicted train (out-of-fold) and test")
    C_tr = pl.read_parquet(P.block / "cand_train.parquet", columns=["q_rid", "t_rid", "src"])
    C_te = pl.read_parquet(P.block / "cand_test.parquet", columns=["q_rid", "t_rid", "src"])
    scores_tr, scores_te = C_tr.with_columns(pl.Series("p1", p1_tr)), C_te.with_columns(pl.Series("p1", p1_te))
    out = {"rows": len(y), "positive_rate": float(y.mean()), "features_stage1": len(cols),
           "best_iterations_stage1": [m.best_iteration for m in s1], "importance_stage1": _importance(s1)}

    if mv.stages == 2:
        text = lambda split: (pl.read_parquet(P.prep / f"T_{split}.parquet", columns=["rid", "name_core", "name_nosp", "addr_n"])
                              if mv.support else None)
        part = lambda C, split: country_codes(C["q_rid"].to_numpy(),
                                              pl.read_parquet(P.prep / f"Q_{split}.parquet", columns=["rid", "country"]))
        extra_tr, extra_cols = _stage2_extra(C_tr, p1_tr, text("train"), part(C_tr, "train"))
        cols2 = cols + extra_cols
        X2, _, _, _ = _load_rows(tr_parts, keep, cols, pairs, extra=extra_tr)  # same rows, same order as X
        s2 = _cross_fit(X2, y, q, role, fold, cols2, mv, "stage2", P, w)
        del X2
        scores_tr = scores_tr.with_columns(pl.Series("p2", _predict_parts(s2, tr_parts, cols, role, fold, extra=extra_tr, margin=mv.pred_margin)))
        del extra_tr
        extra_te, _ = _stage2_extra(C_te, p1_te, text("test"), part(C_te, "test"))  # built only now, to keep the peak down
        scores_te = scores_te.with_columns(pl.Series("p2", _predict_parts(s2, te_parts, cols, extra=extra_te, margin=mv.pred_margin)))
        del extra_te
        log("stage 2 predicted train (out-of-fold) and test")
        out.update({"features_stage2": len(cols2), "best_iterations_stage2": [m.best_iteration for m in s2],
                    "importance_stage2": _importance(s2)})
    scores_tr.write_parquet(P.run / "scores_train.parquet")
    scores_te.write_parquet(P.run / "scores_test.parquet")
    return out


# ------------------------------------------------------------------- predict
def _exclusive(scores: pl.DataFrame) -> pl.DataFrame:
    """Every S2/S3 record belongs to at most one S1 entity: keep it only where it scores highest."""
    return scores.filter(pl.col("p") == pl.col("p").max().over("t_rid"))


def _decide(ex: pl.DataFrame, rule: str, param: float) -> pl.DataFrame:
    """Links from exclusive scores: a threshold, expected F (param = floor), or gated expected F (param = gate)."""
    if rule == "threshold":
        return ex.filter(pl.col("p") >= param).select("q_rid", "t_rid")
    if rule == "gated_ef":
        return M.decide_gated(ex, gate=param, exclusive=False)
    return M.decide_expected_f(ex, floor=param, exclusive=False)


def _write_lists(Q: pl.DataFrame, T: pl.DataFrame, links: pl.DataFrame, col: str, path: Path) -> None:
    """One row per S1 record (file order), with a comma-separated list of S2/S3 ids ('' if none)."""
    ids = (links.select("q_rid", "t_rid")
           .join(T.select(pl.col("rid").alias("t_rid"), pl.col("entity_id").alias("tid")), on="t_rid")
           .sort("q_rid", "tid").group_by("q_rid", maintain_order=True)
           .agg(pl.col("tid").unique(maintain_order=True).str.join(",").alias(col)))
    (Q.select(pl.col("rid").alias("q_rid"), pl.col("entity_id").alias("source1_entity_id"))
     .join(ids, on="q_rid", how="left", maintain_order="left").with_columns(pl.col(col).fill_null(""))
     .select("source1_entity_id", col).write_csv(path, separator="\t", quote_style="never"))


def stage_predict(a, rv: V.RunVersions, P: Paths) -> dict:
    role = pl.read_parquet(P.prep / "split.parquet")["role"].to_numpy()
    pairs = pl.read_parquet(P.prep / "pairs.parquet")
    S = pl.read_parquet(P.run / "scores_train.parquet")
    score_cols = [c for c in ("p1", "p2") if c in S.columns]
    q_eval = np.flatnonzero(role == EVAL).astype(np.uint32)
    ids_eval = pl.DataFrame({"q_rid": q_eval})
    truth_eval = pairs.join(ids_eval, on="q_rid", how="semi")
    cand_eval = S.select("q_rid", "t_rid").join(ids_eval, on="q_rid", how="semi")
    oracle = macro_f05(cand_eval.join(truth_eval, on=["q_rid", "t_rid"]), truth_eval, q_eval)
    log(f"eval: {len(q_eval):,} S1 entities; perfect matcher on these candidates F0.5={oracle['f05']:.4f}")

    # doubled-distractor eval: links to targets no train S1 links to count twice (the test's lookalike density)
    distractors = S.select("t_rid").unique().join(pairs.select("t_rid"), on="t_rid", how="anti")
    key = "f05_dup" if rv.match.eval_dup else "f05"
    table = []
    for score in score_cols:
        ex = _exclusive(S.select("q_rid", "t_rid", pl.col(score).alias("p"))).join(ids_eval, on="q_rid", how="semi")
        for rule in rv.match.rules:
            for v in PARAMS[rule]:
                links = _decide(ex, rule, float(v))
                dup = macro_f05(links, truth_eval, q_eval, double=distractors)
                table.append({"score": score, "rule": rule, "param": float(v), **macro_f05(links, truth_eval, q_eval),
                              "f05_dup": dup["f05"], "precision_dup": dup["precision"]})
        for k in ("f05", "f05_dup"):
            b = max((r for r in table if r["score"] == score), key=lambda r: r[k])
            log(f"best on {score} by {k}: {b['rule']}={b['param']} F0.5={b['f05']:.4f} doubled-distractor F0.5="
                f"{b['f05_dup']:.4f} P={b['precision']:.4f} R={b['recall']:.4f} singletons={b['f05_singletons']:.4f} "
                f"others={b['f05_non_singletons']:.4f}")
    best = max(table, key=lambda r: r[key])

    # per-country view of the chosen rule
    ex = _exclusive(S.select("q_rid", "t_rid", pl.col(best["score"]).alias("p"))).join(ids_eval, on="q_rid", how="semi")
    pred = _decide(ex, best["rule"], best["param"])
    country = pl.read_parquet(P.prep / "Q_train.parquet", columns=["rid", "country"])
    per_country = {}
    for c in country["country"].unique().sort().to_list():
        qc = np.intersect1d(q_eval, country.filter(pl.col("country") == c)["rid"].to_numpy()).astype(np.uint32)
        if len(qc):
            per_country[c] = macro_f05(pred, truth_eval, qc)
    log(f"chosen by {key}: {best['score']} {best['rule']}={best['param']} -> eval F0.5={best['f05']:.4f}, "
        f"doubled-distractor F0.5={best['f05_dup']:.4f}; " + ", ".join(f"{c} {m['f05']:.4f}" for c, m in per_country.items()))
    result = {"eval": {"oracle": oracle, "best": best, "chosen_by": key, "per_country": per_country,
                       "table": sorted(table, key=lambda r: -r[key])[:25]}}
    if not (P.run / "scores_test.parquet").exists():  # a frozen-model run scores the train split only
        return result

    # test
    St = pl.read_parquet(P.run / "scores_test.parquet")
    links = _decide(_exclusive(St.select("q_rid", "t_rid", pl.col(best["score"]).alias("p"))), best["rule"], best["param"])
    Qt = pl.read_parquet(P.prep / "Q_test.parquet", columns=["rid", "entity_id"])
    Tt = pl.read_parquet(P.prep / "T_test.parquet", columns=["rid", "entity_id"])
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_lists(Qt, Tt, links, "matched_entity_ids", out_dir / "matching_results.tsv")
    _write_lists(Qt, Tt, St, "candidate_entity_ids", out_dir / "candidate_pairs.tsv")
    n_linked = links["q_rid"].n_unique()
    test = {"links": links.height, "s1_with_matches": n_linked, "s1_total": Qt.height,
            "share_empty": 1 - n_linked / Qt.height, "candidates": St.height}
    log(f"test: {links.height:,} links for {n_linked:,} of {Qt.height:,} S1 entities -> {out_dir}")
    # France never appears in train, so compare its prediction profile with the other countries
    # (and with the eval slice's truth: ~94% of S1 have a match, ~3.5 links each)
    qc = pl.read_parquet(P.prep / "Q_test.parquet", columns=["rid", "country"]).rename({"rid": "q_rid"})
    per = (qc.join(links.group_by("q_rid").len(), on="q_rid", how="left").with_columns(pl.col("len").fill_null(0))
           .group_by("country").agg(pl.len().alias("s1"), (pl.col("len") > 0).mean().alias("share_linked"),
                                    pl.col("len").mean().alias("links_per_s1")).sort("country"))
    test["per_country"] = {r["country"]: {k: r[k] for k in ("s1", "share_linked", "links_per_s1")} for r in per.iter_rows(named=True)}
    ev_truth = truth_eval.group_by("q_rid").len()
    test["eval_truth_profile"] = {"share_linked": ev_truth.height / len(q_eval), "links_per_s1": truth_eval.height / len(q_eval)}
    log("test profile: " + ", ".join(f"{c} {v['share_linked']:.3f} linked, {v['links_per_s1']:.2f}/S1"
                                     for c, v in test["per_country"].items())
        + f" | eval truth {test['eval_truth_profile']['share_linked']:.3f}, {test['eval_truth_profile']['links_per_s1']:.2f}/S1")
    if a.validator and a.sample >= 1:
        r = subprocess.run([sys.executable, a.validator, "--matching", str(out_dir / "matching_results.tsv"),
                            "--candidate", str(out_dir / "candidate_pairs.tsv"), "--test-dir", str(Path(a.data) / "test")],
                           capture_output=True, text=True)
        test["validator"] = (r.stdout + r.stderr).strip()[-2000:]
        log(f"validator exit {r.returncode}: {r.stdout.strip()[-300:]}")
    return {**result, "test": test}


# ---------------------------------------------------------------------- main
FNS = {"prep": stage_prep, "block": stage_block, "features": stage_features, "train": stage_train,
       "predict": stage_predict}


def list_versions() -> None:
    for title, reg in (("NORM", V.NORM), ("BLK", V.BLOCK), ("FEAT", V.FEAT), ("MATCH", V.MATCH)):
        print(title)
        for vid, v in reg.items():
            print(f"  {vid:12s} {v.note}")
    print("PRESETS (end-to-end versions)")
    for name, ids in V.PRESETS.items():
        print(f"  {name:6s} {' + '.join(ids)}{'   (default)' if name == V.DEFAULT_PRESET else ''}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print every registered version and exit")
    ap.add_argument("--data", help="dataset dir containing train/ and test/")
    ap.add_argument("--work", help="dir for cached stage outputs (one per data scope)")
    ap.add_argument("--out", help="dir for matching_results.tsv and candidate_pairs.tsv")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--preset", default=V.DEFAULT_PRESET, choices=sorted(V.PRESETS))
    ap.add_argument("--norm", choices=sorted(V.NORM), help="override the preset's normalisation version")
    ap.add_argument("--block", choices=sorted(V.BLOCK), help="override the preset's blocking version")
    ap.add_argument("--feat", choices=sorted(V.FEAT), help="override the preset's feature version")
    ap.add_argument("--match", choices=sorted(V.MATCH), help="override the preset's matcher version")
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of the data, for smoke runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fit", type=float, default=0.30, help="share of train S1 entities used to fit models")
    ap.add_argument("--es", type=float, default=0.05, help="share used for early stopping")
    ap.add_argument("--eval", type=float, default=0.20, help="share held out to score and choose the decision rule")
    ap.add_argument("--part-rows", type=int, default=4_000_000)
    ap.add_argument("--validator", default=None, help="path to utils/validate_submission.py")
    a = ap.parse_args()
    if a.list:
        list_versions()
        return 0
    if not (a.data and a.work and a.out):
        ap.error("--data, --work and --out are required")

    rv = V.resolve(a.preset, a.norm, a.block, a.feat, a.match)
    P = Paths(Path(a.work), rv)
    log(f"versions: {rv.preset or 'custom'} = {rv.key('train').replace('__', ' + ')}")
    run_file = P.run / "metrics.json"
    for stage in a.stages.split(","):
        if stage != "prep":
            check_scope(a, P)
        t0 = time.time()
        log(f"==== {stage}  -> {P.stage_dir(stage)}")
        result = FNS[stage](a, rv, P)
        result.update({"seconds": round(time.time() - t0), "peak_gb": round(peak_gb(), 1)})
        write_json(P.stage_dir(stage) / f"{stage}.json", result)
        # the run's metrics.json gathers every stage it depends on, including cached ones
        write_json(run_file, {"versions": rv.describe(), "scope": data_scope(a),
                              **{s: read_json(P.stage_dir(s) / f"{s}.json") for s in STAGES
                                 if (P.stage_dir(s) / f"{s}.json").exists()}})
        log(f"==== {stage} done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
