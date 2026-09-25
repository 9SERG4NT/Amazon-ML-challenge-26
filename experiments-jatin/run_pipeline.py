#!/usr/bin/env python3
"""
Business Entity Resolution — Production Pipeline (full-data scale)
==================================================================
Multi-scheme token-key blocking -> LightGBM pairwise classifier (GPU/CPU)
-> F0.5-tuned dual-threshold decisions.

Scale-safe design (12.5M records, ~120M candidate pairs):
  - pairs are integer position arrays (no giant string columns)
  - TF-IDF fit on a sample, transform in chunks, bounded vocabulary
  - test features computed + scored in chunks (bounded RAM)
  - negative pairs subsampled for CV; thresholds tuned on full OOF

Usage:
    python3 run_pipeline.py [--data-dir DIR] [--output-dir DIR]
                            [--device gpu|cpu] [--sample N] [--neg-keep F]
"""

import os
import re
import gc
import time
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from anyascii import anyascii
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import GroupKFold
from rapidfuzz import fuzz
import lightgbm as lgb

# ============================================================
# CONFIG
# ============================================================

BLOCK_MAX_BUCKET = 300
BLOCK_MAX_CAND = 80
N_FOLDS = 5
SEED = 42
NEG_KEEP = 0.20          # fraction of negative train pairs kept for CV fit
TFIDF_FIT_SAMPLE = 300_000
TFIDF_CHUNK = 1_000_000
TFIDF_MAX_FEATURES = 1 << 19
PRED_CHUNK = 2_000_000
TOPK_FALLBACK = 3        # always consider top-K pairs per entity in the rule

# ============================================================
# NORMALIZATION
# ============================================================

LEGAL_SUFFIXES = {
    "corp", "corporation", "inc", "incorporated", "ltd", "limited", "llc",
    "llp", "lp", "plc", "co", "company", "pvt", "private", "pte", "gmbh",
    "sa", "sarl", "sas", "bv", "nv", "ag", "kg",
}

TOKEN_MAP = {
    "road": "rd", "street": "st", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "boul": "blvd", "drive": "dr", "lane": "ln",
    "highway": "hwy", "court": "ct", "place": "pl", "square": "sq",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "apartment": "apt", "building": "bldg", "floor": "fl", "number": "no",
    "opposite": "opp", "society": "soc", "sector": "sec", "extension": "ext",
    "mount": "mt", "saint": "st", "fort": "ft", "junction": "jct",
    "center": "ctr", "centre": "ctr", "market": "mkt",
}

DBA_PREFIXES = {"trading", "as", "dba", "d", "b", "a", "f", "k", "m", "s"}


def norm_clean(s: str) -> str:
    """Normalize text: transliterate, lowercase, strip punctuation."""
    if not isinstance(s, str) or not s:
        return ""
    s = anyascii(s).lower().replace("&", " and ")
    s = re.sub(r"\.(com|net|org|in|co|us|gov|edu|fr)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def name_tokens(s: str) -> list:
    """Tokenize name: normalize, strip DBA prefixes and legal suffixes."""
    toks = [TOKEN_MAP.get(t, t) for t in norm_clean(s).split()]
    while len(toks) > 2 and toks[0] in DBA_PREFIXES:
        toks = toks[1:]
    while toks and toks[-1] in LEGAL_SUFFIXES:
        toks.pop()
    return toks


def name_key(s: str) -> str:
    return " ".join(name_tokens(s))


def addr_tokens(s: str) -> list:
    stops = {"near", "opp", "beside", "behind", "adjacent", "next", "to", "null"}
    return [TOKEN_MAP.get(t, t) for t in norm_clean(s).split() if t not in stops]


def addr_key(s: str) -> str:
    return " ".join(addr_tokens(s))


def house_no(s: str) -> str:
    m = re.search(r"\d+", norm_clean(s))
    return m.group(0) if m else ""


def postal_code(s: str) -> str:
    nums = re.findall(r"\b\d{5,6}\b", norm_clean(s))
    return nums[-1] if nums else ""


# ============================================================
# BLOCKING
# ============================================================

def generate_blocking_keys(ntoks, atoks, hn, country, nclean):
    """Generate multi-scheme blocking keys for a record."""
    keys = set()
    sig_atoks = [t for t in atoks if len(t) >= 3]

    if len(ntoks) >= 2:
        keys.add(("n2s", country, tuple(sorted(ntoks[:2]))))
    if ntoks:
        keys.add(("n1", country, ntoks[0]))
    if len(ntoks) >= 2:
        keys.add(("nall", country, tuple(sorted(ntoks))))
    if len(nclean) >= 4:
        keys.add(("c4", country, nclean[:4]))
    if len(nclean) >= 5:
        keys.add(("c5", country, nclean[:5]))
    if len(nclean) >= 6:
        for i in range(0, min(len(nclean) - 3, 10)):
            keys.add(("g4", country, nclean[i:i+4]))

    if hn and len(hn) >= 2:
        for at in sig_atoks:
            if at != hn:
                keys.add(("ha", country, hn, at))
                break
    if len(sig_atoks) >= 2:
        keys.add(("a2s", country, tuple(sorted(sig_atoks[:2]))))
    if ntoks and hn:
        keys.add(("nh", country, ntoks[0], hn))
    if ntoks and sig_atoks:
        keys.add(("na", country, ntoks[0], sig_atoks[0]))

    return keys


def build_inverted_index(records, max_bucket=BLOCK_MAX_BUCKET):
    """Build inverted (pruned) index from S23 records."""
    index = defaultdict(list)
    for eid, ntoks, atoks, hn, country, nclean in records:
        for key in generate_blocking_keys(ntoks, atoks, hn, country, nclean):
            index[key].append(eid)
    pruned = {k: v for k, v in index.items() if len(v) <= max_bucket}
    dropped = len(index) - len(pruned)
    print(f"  Inverted index: {len(index)} keys, pruned {dropped} (>{max_bucket}), "
          f"kept {len(pruned)}", flush=True)
    return pruned


def lookup_candidates(s1_records, index, max_cand=BLOCK_MAX_CAND):
    """Candidate S23 ids per S1 id via the inverted index."""
    candidates = {}
    for eid, ntoks, atoks, hn, country, nclean in s1_records:
        cands = set()
        for key in generate_blocking_keys(ntoks, atoks, hn, country, nclean):
            bucket = index.get(key)
            if bucket:
                cands.update(bucket)
        if len(cands) > max_cand:
            cands = set(list(cands)[:max_cand])
        candidates[eid] = cands
    return candidates


def make_blocking_records(df):
    records = []
    for row in df.itertuples():
        nclean = " ".join(row.name_tokens_list)
        records.append((row.entity_id, row.name_tokens_list, row.addr_tokens_list,
                        row.house_no, row.country, nclean))
    return records


# ============================================================
# PAIRS AS INTEGER POSITIONS
# ============================================================

def id_positions(df):
    """(ids array, {id: pos} map) for a source dataframe."""
    ids = df["entity_id"].to_numpy()
    return ids, {e: i for i, e in enumerate(ids)}


def explode_candidates(candidates, s1_pos, s23_pos):
    """candidates dict {s1_id: set(cand_ids)} -> (a1, a2) int64 position arrays."""
    s1_parts, s23_parts = [], []
    for s1_id, cids in candidates.items():
        i1 = s1_pos[s1_id]
        arr = np.fromiter((s23_pos[c] for c in cids if c in s23_pos), dtype=np.int64)
        if len(arr):
            s1_parts.append(np.full(len(arr), i1, dtype=np.int64))
            s23_parts.append(arr)
    if not s1_parts:
        empty = np.array([], dtype=np.int64)
        return empty, empty
    return np.concatenate(s1_parts), np.concatenate(s23_parts)


def dedup_pairs(a1, a2, n2):
    """Drop duplicate (s1, s23) pairs, keeping first occurrence."""
    comp = a1 * n2 + a2
    _, first = np.unique(comp, return_index=True)
    keep = np.zeros(len(comp), dtype=bool)
    keep[first] = True
    return comp[keep], a1[keep], a2[keep]


def pair_labels(comp, link_comp):
    """bool array: comp[i] is a ground-truth link."""
    if len(link_comp) == 0:
        return np.zeros(len(comp), dtype=bool)
    return np.isin(comp, link_comp)


# ============================================================
# TF-IDF (sample-fit, chunked transform, bounded vocab)
# ============================================================

def _fit_vectorize(texts, sample_n=TFIDF_FIT_SAMPLE):
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(texts))
    if len(texts) > sample_n:
        idx = rng.choice(idx, size=sample_n, replace=False)
    sample = [texts[i] for i in idx]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                          max_features=TFIDF_MAX_FEATURES, sublinear_tf=True,
                          dtype=np.float32)
    vec.fit(sample)
    return vec


def transform_chunked(vec, texts):
    """Transform all texts in chunks -> L2-normalized CSR matrix."""
    blocks = []
    for start in range(0, len(texts), TFIDF_CHUNK):
        block = vec.transform([texts[i] for i in range(start, min(start + TFIDF_CHUNK,
                                                                  len(texts)))])
        blocks.append(normalize(block))
    from scipy.sparse import vstack
    return vstack(blocks).tocsr()


def build_tfidf(s1_texts, s23_texts):
    """Fit on a sample of the joint corpus; transform both sides chunked."""
    joint = list(s1_texts) + list(s23_texts)
    vec = _fit_vectorize(joint)
    del joint
    gc.collect()
    X1 = transform_chunked(vec, s1_texts)
    X2 = transform_chunked(vec, s23_texts)
    return X1, X2


# ============================================================
# FEATURES (vectorized over position arrays)
# ============================================================

FEATURE_NAMES = [
    "name_fuzz_ratio", "name_fuzz_partial", "name_fuzz_token_sort",
    "name_fuzz_token_set", "addr_fuzz_ratio", "addr_fuzz_token_set",
    "name_jac", "name_overlap", "addr_jac", "addr_overlap",
    "name_exact", "name_sorted_eq", "house_eq", "postal_eq",
    "name_len_ratio", "addr_len_ratio", "is_s3",
    "name_tfidf_cos", "addr_tfidf_cos",
]


def source_columns(df):
    """Per-record feature columns as object arrays, aligned with df order."""
    nk = df["name_key"].to_numpy(dtype=object)
    ak = df["addr_key"].to_numpy(dtype=object)
    hn = df["house_no"].to_numpy(dtype=object)
    pc = df["postal_code"].to_numpy(dtype=object)
    nt = np.array([set(t) for t in df["name_tokens_list"]], dtype=object)
    at = np.array([set(t) for t in df["addr_tokens_list"]], dtype=object)
    return dict(nk=nk, ak=ak, hn=hn, pc=pc, nt=nt, at=at)


def _fz_batch(func, a, b):
    return np.array([func(x, y) for x, y in zip(a, b)], dtype=np.float32) / 100.0


def compute_features_chunk(cols1, cols2, a1, a2, Xn1, Xn2, Xa1, Xa2, cand_is_s3):
    """Features for the pair slice given pre-extracted column dicts."""
    m = len(a1)
    F = np.zeros((m, len(FEATURE_NAMES)), dtype=np.float32)

    nk1, nk2 = cols1["nk"][a1], cols2["nk"][a2]
    ak1, ak2 = cols1["ak"][a1], cols2["ak"][a2]
    nt1, nt2 = cols1["nt"][a1], cols2["nt"][a2]
    at1, at2 = cols1["at"][a1], cols2["at"][a2]

    F[:, 0] = _fz_batch(fuzz.ratio, nk1, nk2)
    F[:, 1] = _fz_batch(fuzz.partial_ratio, nk1, nk2)
    F[:, 2] = _fz_batch(fuzz.token_sort_ratio, nk1, nk2)
    F[:, 3] = _fz_batch(fuzz.token_set_ratio, nk1, nk2)
    F[:, 4] = _fz_batch(fuzz.ratio, ak1, ak2)
    F[:, 5] = _fz_batch(fuzz.token_set_ratio, ak1, ak2)

    inter = np.fromiter((len(x & y) for x, y in zip(nt1, nt2)), dtype=np.float32, count=m)
    union = np.fromiter((len(x | y) for x, y in zip(nt1, nt2)), dtype=np.float32, count=m)
    mn = np.fromiter((min(len(x), len(y)) for x, y in zip(nt1, nt2)),
                     dtype=np.float32, count=m)
    F[:, 6] = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    F[:, 7] = np.divide(inter, mn, out=np.zeros_like(inter), where=mn > 0)

    inter = np.fromiter((len(x & y) for x, y in zip(at1, at2)), dtype=np.float32, count=m)
    union = np.fromiter((len(x | y) for x, y in zip(at1, at2)), dtype=np.float32, count=m)
    mn = np.fromiter((min(len(x), len(y)) for x, y in zip(at1, at2)),
                     dtype=np.float32, count=m)
    F[:, 8] = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    F[:, 9] = np.divide(inter, mn, out=np.zeros_like(inter), where=mn > 0)

    F[:, 10] = (nk1 == nk2) & (nk1 != "")
    s1 = np.array([" ".join(sorted(x.split())) for x in nk1], dtype=object)
    s2 = np.array([" ".join(sorted(x.split())) for x in nk2], dtype=object)
    F[:, 11] = (s1 == s2) & (s1 != "")
    del s1, s2

    hn1, hn2 = cols1["hn"][a1], cols2["hn"][a2]
    pc1, pc2 = cols1["pc"][a1], cols2["pc"][a2]
    F[:, 12] = (hn1 == hn2) & (hn1 != "")
    F[:, 13] = (pc1 == pc2) & (pc1 != "")

    l1 = np.fromiter((len(x) for x in nk1), dtype=np.float32, count=m)
    l2 = np.fromiter((len(x) for x in nk2), dtype=np.float32, count=m)
    F[:, 14] = np.minimum(l1 / np.maximum(l2, 1.0), 3.0)
    l1 = np.fromiter((len(x) for x in ak1), dtype=np.float32, count=m)
    l2 = np.fromiter((len(x) for x in ak2), dtype=np.float32, count=m)
    F[:, 15] = np.minimum(l1 / np.maximum(l2, 1.0), 3.0)

    F[:, 16] = cand_is_s3

    if Xn1 is not None:
        F[:, 17] = np.asarray(Xn1[a1].multiply(Xn2[a2]).sum(axis=1)).ravel()
        F[:, 18] = np.asarray(Xa1[a1].multiply(Xa2[a2]).sum(axis=1)).ravel()
    return F


# ============================================================
# MODEL
# ============================================================

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=40,
    colsample_bytree=0.85,
    subsample=0.85,
    subsample_freq=1,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=SEED,
    verbose=-1,
)


def make_model(device, n_estimators):
    params = dict(LGB_PARAMS)
    if device == "gpu":
        params["device"] = "gpu"
        params["gpu_use_dp"] = False
        params["n_jobs"] = max(1, (os.cpu_count() or 8) // 2)
    return lgb.LGBMClassifier(n_estimators=n_estimators, **params)


def subsample_negatives(y, keep=NEG_KEEP, seed=SEED):
    """Keep all positives + a fraction of negatives. Returns index array."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    neg = neg[rng.random(len(neg)) < keep]
    return np.sort(np.concatenate([pos, neg]))


def train_cv(X, y, groups, device, n_splits=N_FOLDS):
    """GroupKFold CV; returns OOF predictions (on the given rows) + mean best iter."""
    oof = np.zeros(len(y), dtype=np.float32)
    iters = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        m = make_model(device, 2000)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        iters.append(m.best_iteration_ or 2000)
        print(f"  Fold {fold+1}: best_iter={iters[-1]}", flush=True)
        del m
        gc.collect()
    return oof, int(np.mean(iters))


def predict_probs_chunked(model, X, n, chunk=PRED_CHUNK):
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        out[start:end] = model.predict_proba(X[start:end])[:, 1]
    return out


# ============================================================
# DECISION RULE + TUNING (vectorized, per-edge)
# ============================================================

def f05_from_counts(tp, pp, gt):
    """Macro F0.5 from per-entity TP / predicted-positive / ground-truth counts."""
    p = np.divide(tp, pp, out=np.zeros_like(tp, dtype=np.float64), where=pp > 0)
    r = np.divide(tp, gt, out=np.zeros_like(tp, dtype=np.float64), where=gt > 0)
    denom = 0.25 * p + r
    f = np.divide(1.25 * p * r, denom, out=np.zeros_like(p), where=denom > 0)
    f[(pp == 0) & (gt == 0)] = 1.0
    return f.mean()


def rule_mask(p, s1pos, t_high, t_low, rel, n_ent, topk=TOPK_FALLBACK):
    """Dual-threshold rule (+ optional top-K per-entity floor) -> boolean edge mask."""
    if len(p) == 0:
        return np.zeros(0, dtype=bool)
    pmax = np.full(n_ent, -np.inf)
    np.maximum.at(pmax, s1pos, p)
    mask = (p >= t_high) | ((p >= t_low) & (p >= rel * pmax[s1pos]))
    if topk and topk > 0:
        # Top-K fallback: sort by (entity asc, p desc) so entities are contiguous
        order = np.lexsort((-p, s1pos))
        s_sorted = s1pos[order]
        new_group = np.empty(len(s_sorted), dtype=bool)
        new_group[0] = True
        new_group[1:] = s_sorted[1:] != s_sorted[:-1]
        gs = np.flatnonzero(new_group)
        rank = np.arange(len(s_sorted)) - np.repeat(gs,
                                                    np.diff(np.append(gs, len(s_sorted))))
        mask[order[rank < topk]] = True
    return mask


def tune_thresholds(p, s1pos, labels, n_ent):
    """Grid search (t_high, t_low, rel, topk) maximizing macro F0.5 on OOF edges.

    The top-K per-entity floor is itself tuned: it guarantees recall on
    non-singletons (valuable for the unseen France country) but forces
    predictions onto true singletons, so the data decides whether it helps.
    """
    gt_counts = np.bincount(s1pos[labels], minlength=n_ent).astype(np.int64)
    best, best_score, best_topk = (0.5, 0.3, 0.7), -1.0, 0
    for topk in (0, TOPK_FALLBACK):
        for th in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
            for tl in [0.25, 0.30, 0.40, 0.50, 0.60]:
                for rl in [0.60, 0.70, 0.80, 0.90]:
                    sel = rule_mask(p, s1pos, th, tl, rl, n_ent, topk=topk)
                    tp = np.bincount(s1pos[sel & labels], minlength=n_ent)
                    pp = np.bincount(s1pos[sel], minlength=n_ent)
                    score = f05_from_counts(tp, pp, gt_counts)
                    if score > best_score:
                        best_score, best, best_topk = score, (th, tl, rl), topk
    return best, best_score, best_topk


# ============================================================
# DATA LOADING
# ============================================================

def load_source(path, max_rows=0):
    """Load a source TSV and preprocess columns. max_rows>0 caps rows (dev mode).

    Raw text columns are dropped after deriving keys/tokens to save RAM at scale.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, nrows=max_rows or None).fillna("")
    df["name_key"] = df["business_name"].map(name_key)
    df["addr_key"] = df["business_address"].map(addr_key)
    df["house_no"] = df["business_address"].map(house_no)
    df["postal_code"] = df["business_address"].map(postal_code)
    df["name_tokens_list"] = df["business_name"].map(name_tokens)
    df["addr_tokens_list"] = df["business_address"].map(addr_tokens)
    return df.drop(columns=["business_name", "business_address"])


def load_gt(path):
    gt = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    return {r.source1_entity_id: set(r.matched_entity_ids.split(",")) - {""}
            for r in gt.itertuples()}


# ============================================================
# OUTPUT
# ============================================================

def write_outputs(s1_ids, s23_ids, a1, a2, sel_mask, output_dir):
    """Write matching_results.tsv + candidate_pairs.tsv from edge arrays."""
    os.makedirs(output_dir, exist_ok=True)
    n1 = len(s1_ids)

    match_path = os.path.join(output_dir, "matching_results.tsv")
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # Sort edges by (a1, a2) once; per-entity slices are then contiguous + sorted
    o = np.lexsort((a2, a1))
    a1s, a2s = a1[o], a2[o]
    sel_s = sel_mask[o]

    starts = np.searchsorted(a1s, np.arange(n1), side="left")
    ends = np.searchsorted(a1s, np.arange(n1), side="right")
    ms = np.searchsorted(a1s[sel_s], np.arange(n1), side="left")
    me = np.searchsorted(a1s[sel_s], np.arange(n1), side="right")
    # positions of selected edges within the sorted arrays
    sel_positions = np.flatnonzero(sel_s)
    a2_sel_sorted = a2s[sel_s]

    with open(match_path, "w", encoding="utf-8") as fm, \
         open(cand_path, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for i in range(n1):
            c_sl = s23_ids[a2s[starts[i]:ends[i]]]
            c_str = ",".join(c_sl) if len(c_sl) else ""
            m_sl = a2_sel_sorted[ms[i]:me[i]]
            m_str = ",".join(s23_ids[m_sl]) if len(m_sl) else ""
            fm.write(f"{s1_ids[i]}\t{m_str}\n")
            fc.write(f"{s1_ids[i]}\t{c_str}\n")
    return match_path, cand_path


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(here, "..", "resources",
                                                          "student_resource", "dataset"))
    parser.add_argument("--output-dir", default=os.path.join(here, "output_dev"))
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample N S1 records for development (0 = full)")
    parser.add_argument("--neg-keep", type=float, default=NEG_KEEP)
    parser.add_argument("--save-model", default=os.path.join(here, "model.txt"))
    args = parser.parse_args()

    data_dir, output_dir = args.data_dir, args.output_dir
    t_total = time.time()
    rng = np.random.default_rng(SEED)

    # ================= PHASE 1: TRAIN =================
    print("=" * 60)
    print("PHASE 1: TRAINING")
    print("=" * 60, flush=True)

    print("\n[1/8] Loading training data...", flush=True)
    t0 = time.time()
    s23_cap = args.sample * 60 if args.sample > 0 else 0
    s1_tr = load_source(os.path.join(data_dir, "train", "train_source1.tsv"))
    s2_tr = load_source(os.path.join(data_dir, "train", "train_source2.tsv"),
                        max_rows=s23_cap)
    s3_tr = load_source(os.path.join(data_dir, "train", "train_source3.tsv"),
                        max_rows=s23_cap)
    s23_tr = pd.concat([s2_tr, s3_tr], ignore_index=True)
    del s2_tr, s3_tr
    gc.collect()
    gt_map = load_gt(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    print(f"  Loaded in {time.time()-t0:.1f}s: S1={len(s1_tr)}, S23={len(s23_tr)}, "
          f"GT={len(gt_map)}", flush=True)

    if args.sample > 0:
        keep = set(s1_tr["entity_id"].head(args.sample))
        s1_tr = s1_tr[s1_tr["entity_id"].isin(keep)].reset_index(drop=True)
        gt_map = {k: v for k, v in gt_map.items() if k in keep}
        print(f"  [DEV MODE] S1 sampled to {len(s1_tr)}", flush=True)

    s1_ids, s1_pos = id_positions(s1_tr)
    s23_ids, s23_pos = id_positions(s23_tr)
    n2 = len(s23_ids)

    # GT links as composite positions
    link_comp = np.array([s1_pos[s] * n2 + s23_pos[c]
                          for s, ms in gt_map.items() for c in ms
                          if c in s23_pos and s in s1_pos], dtype=np.int64)
    print(f"  GT links resolvable in pool: {len(link_comp)}", flush=True)

    print("\n[2/8] Blocking (train)...", flush=True)
    t0 = time.time()
    candidates = {}
    for country in s1_tr["country"].unique():
        c_s1 = s1_tr[s1_tr["country"] == country]
        c_s23 = s23_tr[s23_tr["country"] == country]
        if len(c_s1) == 0:
            continue
        print(f"  Country '{country}': S1={len(c_s1)}, S23={len(c_s23)}", flush=True)
        index = build_inverted_index(make_blocking_records(c_s23))
        candidates.update(lookup_candidates(make_blocking_records(c_s1), index))
        del index
    for eid in s1_ids:
        candidates.setdefault(eid, set())
    a1, a2 = explode_candidates(candidates, s1_pos, s23_pos)
    del candidates
    gc.collect()
    comp, a1, a2 = dedup_pairs(a1, a2, n2)
    labels = pair_labels(comp, link_comp)
    hit = int(labels.sum()); tot = len(link_comp)
    print(f"  {len(a1)} pairs in {time.time()-t0:.1f}s | "
          f"blocking recall = {hit}/{tot} = {hit/max(tot,1):.4f}", flush=True)
    del comp, link_comp
    gc.collect()

    # Subsample pairs BEFORE feature computation: keep all positives + a
    # fraction of negatives. Features/TF-IDF cosines are only needed for the
    # rows the model will actually see; this cuts RAM and compute ~4x.
    sub = subsample_negatives(labels, keep=args.neg_keep)
    a1, a2, labels = a1[sub], a2[sub], labels[sub].astype(np.int8)
    print(f"  Train rows after neg subsample: {len(a1)} "
          f"(pos={int(labels.sum())}, neg_keep={args.neg_keep})", flush=True)
    del sub
    gc.collect()

    print("\n[3/8] TF-IDF (train)...", flush=True)
    t0 = time.time()
    Xn1, Xn2 = build_tfidf(s1_tr["name_key"].to_numpy(), s23_tr["name_key"].to_numpy())
    Xa1, Xa2 = build_tfidf(s1_tr["addr_key"].to_numpy(), s23_tr["addr_key"].to_numpy())
    print(f"  TF-IDF done in {time.time()-t0:.1f}s", flush=True)

    print("\n[4/8] Features (train)...", flush=True)
    t0 = time.time()
    cols1, cols2 = source_columns(s1_tr), source_columns(s23_tr)
    cand_is_s3 = np.fromiter((s23_ids[j].startswith("S3-") for j in a2),
                             dtype=np.float32, count=len(a2))
    X_tr = compute_features_chunk(cols1, cols2, a1, a2, Xn1, Xn2, Xa1, Xa2, cand_is_s3)
    y_tr = labels
    groups_tr = a1  # group by S1 entity position
    print(f"  Features {X_tr.shape} in {time.time()-t0:.1f}s | "
          f"pos_rate={y_tr.mean():.4f}", flush=True)

    del Xn1, Xn2, Xa1, Xa2, cand_is_s3, labels
    gc.collect()

    print(f"\n[5/8] LightGBM CV ({args.device})...", flush=True)
    t0 = time.time()
    oof, best_iter = train_cv(X_tr, y_tr, groups_tr, args.device, n_splits=N_FOLDS)
    print(f"  CV done in {time.time()-t0:.1f}s, mean_best_iter={best_iter}", flush=True)

    print("  Tuning thresholds on OOF...", flush=True)
    rule, oof_score, rule_topk = tune_thresholds(oof, groups_tr, y_tr,
                                                 n_ent=len(s1_ids))
    print(f"  OOF Macro F0.5 = {oof_score:.4f} | rule: t_high={rule[0]}, "
          f"t_low={rule[1]}, rel={rule[2]}, topk={rule_topk}", flush=True)
    del oof
    gc.collect()

    print("\n[6/8] Training full model...", flush=True)
    t0 = time.time()
    clf = make_model(args.device, max(best_iter, 200))
    clf.fit(X_tr, y_tr)
    print(f"  Full model trained in {time.time()-t0:.1f}s", flush=True)
    if args.save_model:
        try:
            clf.booster_.save_model(args.save_model)
            print(f"  Model saved to {args.save_model}", flush=True)
        except Exception as e:  # non-fatal
            print(f"  Model save failed: {e}", flush=True)
    del X_tr, y_tr, groups_tr, labels
    gc.collect()

    # ================= PHASE 2: TEST =================
    print("\n" + "=" * 60)
    print("PHASE 2: TEST INFERENCE")
    print("=" * 60, flush=True)

    print("\n[7/8] Test: load + blocking + pairs...", flush=True)
    t0 = time.time()
    s1_te = load_source(os.path.join(data_dir, "test", "test_source1.tsv"))
    s2_te = load_source(os.path.join(data_dir, "test", "test_source2.tsv"),
                        max_rows=s23_cap)
    s3_te = load_source(os.path.join(data_dir, "test", "test_source3.tsv"),
                        max_rows=s23_cap)
    s23_te = pd.concat([s2_te, s3_te], ignore_index=True)
    del s2_te, s3_te
    gc.collect()
    s1_te = s1_te.reset_index(drop=True)
    s23_te = s23_te.reset_index(drop=True)
    print(f"  Loaded in {time.time()-t0:.1f}s: S1={len(s1_te)}, S23={len(s23_te)}",
          flush=True)

    s1_ids_te, _ = id_positions(s1_te)
    s23_ids_te, s23_pos_te = id_positions(s23_te)
    s1_pos_te = {e: i for i, e in enumerate(s1_ids_te)}
    n2_te = len(s23_ids_te)

    t0 = time.time()
    candidates_te = {}
    for country in s1_te["country"].unique():
        c_s1 = s1_te[s1_te["country"] == country]
        c_s23 = s23_te[s23_te["country"] == country]
        if len(c_s1) == 0:
            continue
        print(f"  Country '{country}': S1={len(c_s1)}, S23={len(c_s23)}", flush=True)
        index = build_inverted_index(make_blocking_records(c_s23))
        candidates_te.update(lookup_candidates(make_blocking_records(c_s1), index))
        del index
        gc.collect()
    for eid in s1_ids_te:
        candidates_te.setdefault(eid, set())
    a1_te, a2_te = explode_candidates(candidates_te, s1_pos_te, s23_pos_te)
    del candidates_te
    gc.collect()
    _, a1_te, a2_te = dedup_pairs(a1_te, a2_te, n2_te)
    print(f"  Test pairs: {len(a1_te)} in {time.time()-t0:.1f}s", flush=True)

    print("\n[8/8] Test: TF-IDF + features + predict (chunked)...", flush=True)
    t0 = time.time()
    Xn1t, Xn2t = build_tfidf(s1_te["name_key"].to_numpy(), s23_te["name_key"].to_numpy())
    Xa1t, Xa2t = build_tfidf(s1_te["addr_key"].to_numpy(), s23_te["addr_key"].to_numpy())
    cols1t, cols2t = source_columns(s1_te), source_columns(s23_te)
    print(f"  TF-IDF + columns ready in {time.time()-t0:.1f}s", flush=True)

    n_te = len(a1_te)
    sel_all = np.zeros(n_te, dtype=bool)
    n_ent_te = len(s1_ids_te)
    t0 = time.time()
    for start in range(0, n_te, PRED_CHUNK):
        end = min(start + PRED_CHUNK, n_te)
        b1, b2 = a1_te[start:end], a2_te[start:end]
        is_s3 = np.fromiter((s23_ids_te[j].startswith("S3-") for j in b2),
                            dtype=np.float32, count=end - start)
        Fb = compute_features_chunk(cols1t, cols2t, b1, b2,
                                    Xn1t, Xn2t, Xa1t, Xa2t, is_s3)
        pb = clf.predict_proba(Fb)[:, 1].astype(np.float32)
        sel_all[start:end] = rule_mask(pb, b1, *rule, n_ent=n_ent_te, topk=rule_topk)
        del Fb, pb, b1, b2, is_s3
        gc.collect()
        print(f"  chunk {start//PRED_CHUNK + 1}/{(n_te + PRED_CHUNK - 1)//PRED_CHUNK} "
              f"done ({time.time()-t0:.0f}s)", flush=True)

    print("  Writing outputs...", flush=True)
    t0 = time.time()
    m_path, c_path = write_outputs(s1_ids_te, s23_ids_te, a1_te, a2_te, sel_all,
                                   output_dir)
    n_pred = int(sel_all.sum())
    print(f"  Wrote {n_pred} matched edges in {time.time()-t0:.1f}s", flush=True)
    print(f"\nDone! Total time: {time.time()-t_total:.1f}s")
    print(f"Output: {m_path} , {c_path}")


if __name__ == "__main__":
    main()
