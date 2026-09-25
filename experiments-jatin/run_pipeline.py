#!/usr/bin/env python3
"""
Business Entity Resolution — Production Pipeline
=================================================
Multi-scheme token-key blocking → pre-filter → LightGBM pairwise classifier → F0.5-tuned decisions.

Usage:
    python3 run_pipeline.py --data-dir dataset --output-dir output [--sample N]

Produces:
    output/matching_results.tsv   — final entity matches (scored on leaderboard)
    output/candidate_pairs.tsv    — blocking candidate set (for auditing)
"""

import os
import sys
import time
import csv
import re
import argparse
import gc
from collections import defaultdict
import numpy as np
import pandas as pd
from anyascii import anyascii
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

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
    # Strip DBA/trading-as prefixes (only if followed by real name tokens)
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

    # Name-based keys
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

    # Char 4-grams from name (catches partial matches/typos), limited count
    if len(nclean) >= 6:
        for i in range(0, min(len(nclean) - 3, 10)):
            keys.add(("g4", country, nclean[i:i+4]))

    # Address-based keys
    if hn and len(hn) >= 2:
        for at in sig_atoks:
            if at != hn:
                keys.add(("ha", country, hn, at))
                break
    if len(sig_atoks) >= 2:
        keys.add(("a2s", country, tuple(sorted(sig_atoks[:2]))))

    # Combo keys
    if ntoks and hn:
        keys.add(("nh", country, ntoks[0], hn))
    if ntoks and sig_atoks:
        keys.add(("na", country, ntoks[0], sig_atoks[0]))

    return keys


def build_inverted_index(records, max_bucket=300):
    """Build inverted index from S23 records. Returns (pruned_index, record_dict)."""
    index = defaultdict(list)
    for eid, ntoks, atoks, hn, country, nclean in records:
        keys = generate_blocking_keys(ntoks, atoks, hn, country, nclean)
        for key in keys:
            index[key].append(eid)

    # Prune large buckets
    pruned = {k: v for k, v in index.items() if len(v) <= max_bucket}
    dropped = len(index) - len(pruned)
    print(f"  Inverted index: {len(index)} keys, pruned {dropped} (>{max_bucket}), kept {len(pruned)}")
    return pruned


def lookup_candidates(s1_records, index, max_cand=80):
    """Look up candidate S23 records for each S1 record using the inverted index."""
    candidates = {}
    for eid, ntoks, atoks, hn, country, nclean in s1_records:
        keys = generate_blocking_keys(ntoks, atoks, hn, country, nclean)
        cands = set()
        for key in keys:
            bucket = index.get(key, [])
            cands.update(bucket)
        # Cap candidates per S1 entity
        if len(cands) > max_cand:
            cands = set(list(cands)[:max_cand])
        candidates[eid] = cands
    return candidates


# ============================================================
# FEATURES
# ============================================================

FEATURE_NAMES = [
    "name_fuzz_ratio", "name_fuzz_partial", "name_fuzz_token_sort",
    "name_fuzz_token_set",
    "addr_fuzz_ratio", "addr_fuzz_token_set",
    "name_jac", "name_overlap",
    "addr_jac", "addr_overlap",
    "name_exact", "name_sorted_eq",
    "house_eq", "postal_eq",
    "name_len_ratio", "addr_len_ratio",
    "is_s3",
    "name_tfidf_cos", "addr_tfidf_cos",
]


def compute_features_batch(pairs_df, s1_data, s23_data, name_vecs=None, addr_vecs=None):
    """Compute pairwise features for all candidate pairs.

    s1_data, s23_data: dict of eid -> (name_key, addr_key, house_no, postal_code, country, name_tokens_set, addr_tokens_set)
    """
    n = len(pairs_df)
    feats = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)

    s1_ids = pairs_df["s1_id"].values
    cand_ids = pairs_df["cand_id"].values

    for i in range(n):
        d1 = s1_data[s1_ids[i]]
        d2 = s23_data[cand_ids[i]]
        nk1, ak1, hn1, pc1, c1, nt1, at1 = d1
        nk2, ak2, hn2, pc2, c2, nt2, at2 = d2

        # Rapidfuzz name features
        feats[i, 0] = fuzz.ratio(nk1, nk2) / 100.0
        feats[i, 1] = fuzz.partial_ratio(nk1, nk2) / 100.0
        feats[i, 2] = fuzz.token_sort_ratio(nk1, nk2) / 100.0
        feats[i, 3] = fuzz.token_set_ratio(nk1, nk2) / 100.0

        # Rapidfuzz addr features
        feats[i, 4] = fuzz.ratio(ak1, ak2) / 100.0
        feats[i, 5] = fuzz.token_set_ratio(ak1, ak2) / 100.0

        # Token Jaccard & overlap for name
        u = len(nt1 | nt2)
        inter = len(nt1 & nt2)
        feats[i, 6] = inter / u if u else 0.0
        mn = min(len(nt1), len(nt2))
        feats[i, 7] = inter / mn if mn else 0.0

        # Token Jaccard & overlap for addr
        ua = len(at1 | at2)
        intera = len(at1 & at2)
        feats[i, 8] = intera / ua if ua else 0.0
        ma = min(len(at1), len(at2))
        feats[i, 9] = intera / ma if ma else 0.0

        # Exact matches
        feats[i, 10] = float(nk1 == nk2 and len(nk1) > 0)
        sorted1 = " ".join(sorted(nk1.split()))
        sorted2 = " ".join(sorted(nk2.split()))
        feats[i, 11] = float(sorted1 == sorted2 and len(sorted1) > 0)

        feats[i, 12] = float(hn1 == hn2 and len(hn1) > 0)
        feats[i, 13] = float(pc1 == pc2 and len(pc1) > 0)

        # Length ratios
        feats[i, 14] = min(len(nk1) / max(len(nk2), 1), 3.0)
        feats[i, 15] = min(len(ak1) / max(len(ak2), 1), 3.0)

        # Source flag
        feats[i, 16] = float(cand_ids[i].startswith("S3-"))

    # TF-IDF cosine features (precomputed sparse vectors)
    if name_vecs is not None:
        s1_idx = pairs_df["s1_idx"].values
        s23_idx = pairs_df["s23_idx"].values
        # Batch sparse dot products
        for start in range(0, n, 50000):
            end = min(start + 50000, n)
            i1 = s1_idx[start:end]
            i2 = s23_idx[start:end]
            # Name TF-IDF cosine
            cos_n = np.asarray(name_vecs[0][i1].multiply(name_vecs[1][i2]).sum(axis=1)).ravel()
            feats[start:end, 17] = cos_n
            # Addr TF-IDF cosine
            cos_a = np.asarray(addr_vecs[0][i1].multiply(addr_vecs[1][i2]).sum(axis=1)).ravel()
            feats[start:end, 18] = cos_a

    return feats


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
    random_state=42,
    verbose=-1,
)


def train_cv(X, y, groups, n_splits=5):
    """GroupKFold CV by S1 entity. Returns OOF predictions and mean best iteration."""
    oof = np.zeros(len(y), dtype=np.float32)
    iters = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        m = lgb.LGBMClassifier(n_estimators=2000, **LGB_PARAMS)
        m.fit(X[tr], y[tr],
              eval_X=X[va], eval_y=y[va],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        iters.append(m.best_iteration_ or 2000)
        print(f"  Fold {fold+1}: best_iter={iters[-1]}")
    return oof, int(np.mean(iters))


def train_full(X, y, n_iter):
    """Train on full data with given iteration count."""
    m = lgb.LGBMClassifier(n_estimators=max(n_iter, 200), **LGB_PARAMS)
    m.fit(X, y)
    return m


# ============================================================
# EVALUATION
# ============================================================

def per_entity_f05(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    tp = len(pred & true)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(true) if true else 0.0
    if p + r == 0.0:
        return 0.0
    return (1.25 * p * r) / (0.25 * p + r)


def macro_f05(pred_dict, true_dict, entities):
    tot = sum(per_entity_f05(set(pred_dict.get(e, ())), set(true_dict.get(e, ())))
              for e in entities)
    return tot / len(entities)


# ============================================================
# DECISION RULE
# ============================================================

def select_matches(prob_df, t_high, t_low, rel):
    """Apply dual-threshold decision rule."""
    res = {}
    for s1_id, g in prob_df.groupby("s1_id", sort=False):
        pmax = g["p"].max()
        m = (g["p"] >= t_high) | ((g["p"] >= t_low) & (g["p"] >= rel * pmax))
        res[s1_id] = g.loc[m, "cand_id"].tolist()
    return res


def tune_thresholds(oof_df, gt_map, entities):
    """Grid search for best (t_high, t_low, rel) on OOF predictions."""
    best, best_score = (0.5, 0.3, 0.7), -1.0
    for th in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        for tl in [0.25, 0.30, 0.40, 0.50, 0.60]:
            for rl in [0.60, 0.70, 0.80, 0.90]:
                pred = select_matches(oof_df, th, tl, rl)
                score = macro_f05(pred, gt_map, entities)
                if score > best_score:
                    best_score, best = score, (th, tl, rl)
    return best, best_score


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def load_source(path):
    """Load a source TSV file and preprocess columns."""
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    df["name_key"] = df["business_name"].map(name_key)
    df["addr_key"] = df["business_address"].map(addr_key)
    df["house_no"] = df["business_address"].map(house_no)
    df["postal_code"] = df["business_address"].map(postal_code)
    df["name_tokens_list"] = df["business_name"].map(name_tokens)
    df["addr_tokens_list"] = df["business_address"].map(addr_tokens)
    return df


def make_blocking_records(df):
    """Convert dataframe to list of tuples for blocking."""
    records = []
    for row in df.itertuples():
        nclean = " ".join(row.name_tokens_list)
        records.append((row.entity_id, row.name_tokens_list, row.addr_tokens_list,
                        row.house_no, row.country, nclean))
    return records


def make_data_dict(df):
    """Create lookup dict for feature computation."""
    d = {}
    for row in df.itertuples():
        nt_set = set(row.name_tokens_list)
        at_set = set(row.addr_tokens_list)
        d[row.entity_id] = (row.name_key, row.addr_key, row.house_no,
                            row.postal_code, row.country, nt_set, at_set)
    return d


def build_tfidf_vectors(s1_df, s23_df):
    """Build TF-IDF vectors for name and address."""
    all_names = pd.concat([s1_df["name_key"], s23_df["name_key"]], ignore_index=True)
    all_addrs = pd.concat([s1_df["addr_key"], s23_df["addr_key"]], ignore_index=True)
    n1 = len(s1_df)

    name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                               min_df=2, max_df=0.1, sublinear_tf=True,
                               dtype=np.float32)
    X_name = name_vec.fit_transform(all_names)
    from sklearn.preprocessing import normalize
    X_name = normalize(X_name)

    addr_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                               min_df=2, max_df=0.1, sublinear_tf=True,
                               dtype=np.float32)
    X_addr = addr_vec.fit_transform(all_addrs)
    X_addr = normalize(X_addr)

    name_vecs = (X_name[:n1], X_name[n1:])
    addr_vecs = (X_addr[:n1], X_addr[n1:])
    return name_vecs, addr_vecs


def build_pairs_df(candidates, s1_df, s23_df, gt_map=None):
    """Build pairs DataFrame from candidates dict."""
    s1_pos = {e: i for i, e in enumerate(s1_df["entity_id"])}
    s23_pos = {e: i for i, e in enumerate(s23_df["entity_id"])}
    s23_valid = set(s23_df["entity_id"])

    rows = []
    for s1_id, cids in candidates.items():
        i1 = s1_pos[s1_id]
        for cid in cids:
            if cid not in s23_valid:
                continue
            i2 = s23_pos[cid]
            label = int(cid in gt_map.get(s1_id, set())) if gt_map is not None else -1
            rows.append((s1_id, cid, i1, i2, label))

    df = pd.DataFrame(rows, columns=["s1_id", "cand_id", "s1_idx", "s23_idx", "label"])
    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

def load_gt(path):
    """Load ground truth file."""
    gt = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    return {r.source1_entity_id: set(r.matched_entity_ids.split(",")) - {""}
            for r in gt.itertuples()}


def write_output(s1_entities, pred_dict, cand_dict, output_dir):
    """Write matching_results.tsv and candidate_pairs.tsv."""
    os.makedirs(output_dir, exist_ok=True)

    rows_m, rows_c = [], []
    for eid in s1_entities:
        matched = sorted(pred_dict.get(eid, []))
        cands = sorted(cand_dict.get(eid, []))
        rows_m.append((eid, ",".join(matched)))
        rows_c.append((eid, ",".join(cands)))

    pd.DataFrame(rows_m, columns=["source1_entity_id", "matched_entity_ids"]) \
        .to_csv(os.path.join(output_dir, "matching_results.tsv"), sep="\t", index=False)
    pd.DataFrame(rows_c, columns=["source1_entity_id", "candidate_entity_ids"]) \
        .to_csv(os.path.join(output_dir, "candidate_pairs.tsv"), sep="\t", index=False)


def process_country(s1_df, s23_df, country, gt_map=None, max_bucket=300, max_cand=80):
    """Process a single country: blocking + pairs."""
    c_s1 = s1_df[s1_df["country"] == country].reset_index(drop=True)
    c_s23 = s23_df[s23_df["country"] == country].reset_index(drop=True)

    if len(c_s1) == 0:
        return {}, c_s1, c_s23

    print(f"  Country '{country}': S1={len(c_s1)}, S23={len(c_s23)}")

    s23_recs = make_blocking_records(c_s23)
    index = build_inverted_index(s23_recs, max_bucket=max_bucket)

    s1_recs = make_blocking_records(c_s1)
    candidates = lookup_candidates(s1_recs, index, max_cand=max_cand)

    return candidates, c_s1, c_s23


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample N S1 records for development (0 = full)")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    t_total = time.time()

    # ---- TRAIN ----
    print("=" * 60)
    print("PHASE 1: TRAINING")
    print("=" * 60)

    print("\n[1/7] Loading training data...")
    t0 = time.time()
    s1_tr = load_source(os.path.join(data_dir, "train", "train_source1.tsv"))
    s2_tr = load_source(os.path.join(data_dir, "train", "train_source2.tsv"))
    s3_tr = load_source(os.path.join(data_dir, "train", "train_source3.tsv"))
    s23_tr = pd.concat([s2_tr, s3_tr], ignore_index=True)
    del s2_tr, s3_tr; gc.collect()
    gt_map = load_gt(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    print(f"  Loaded in {time.time()-t0:.1f}s: S1={len(s1_tr)}, S23={len(s23_tr)}, GT={len(gt_map)}")

    if args.sample > 0:
        print(f"  [DEV MODE] Sampling {args.sample} S1 records")
        s1_tr = s1_tr.head(args.sample).reset_index(drop=True)
        # Filter GT to sampled S1
        sampled_ids = set(s1_tr["entity_id"])
        gt_map = {k: v for k, v in gt_map.items() if k in sampled_ids}

    # Blocking per country
    print("\n[2/7] Candidate generation (blocking)...")
    t0 = time.time()
    all_candidates_tr = {}
    countries = s1_tr["country"].unique()
    for country in countries:
        cands, _, _ = process_country(s1_tr, s23_tr, country, gt_map,
                                      max_bucket=300, max_cand=80)
        all_candidates_tr.update(cands)

    # Ensure all S1 entities have entries
    for eid in s1_tr["entity_id"]:
        if eid not in all_candidates_tr:
            all_candidates_tr[eid] = set()

    total_pairs = sum(len(v) for v in all_candidates_tr.values())
    avg_cand = total_pairs / max(len(s1_tr), 1)
    print(f"  Blocking done in {time.time()-t0:.1f}s: total_pairs={total_pairs}, avg_cand/S1={avg_cand:.1f}")

    # Check blocking recall
    hit = sum(1 for s, ms in gt_map.items() for m in ms
              if m in all_candidates_tr.get(s, set()))
    tot = sum(len(ms) for ms in gt_map.values())
    print(f"  Blocking recall = {hit}/{tot} = {hit/max(tot,1):.4f}")

    # Build pairs DataFrame
    print("\n[3/7] Building pairs DataFrame...")
    t0 = time.time()
    pairs_tr = build_pairs_df(all_candidates_tr, s1_tr, s23_tr, gt_map)
    print(f"  {len(pairs_tr)} pairs built in {time.time()-t0:.1f}s, positive_rate={pairs_tr['label'].mean():.4f}")

    # Features
    print("\n[4/7] Computing features...")
    t0 = time.time()
    s1_data = make_data_dict(s1_tr)
    s23_data = make_data_dict(s23_tr)

    # Build TF-IDF vectors
    print("  Building TF-IDF vectors...")
    t1 = time.time()
    name_vecs_tr, addr_vecs_tr = build_tfidf_vectors(s1_tr, s23_tr)
    print(f"  TF-IDF done in {time.time()-t1:.1f}s")

    print("  Computing pairwise features...")
    X_tr = compute_features_batch(pairs_tr, s1_data, s23_data, name_vecs_tr, addr_vecs_tr)
    y_tr = pairs_tr["label"].values
    groups_tr = pairs_tr["s1_id"].values
    print(f"  Features done in {time.time()-t0:.1f}s, shape={X_tr.shape}")

    # Free memory
    del s1_data, s23_data, name_vecs_tr, addr_vecs_tr; gc.collect()

    # Model training with CV
    print("\n[5/7] Training LightGBM with GroupKFold CV...")
    t0 = time.time()
    oof_preds, best_iter = train_cv(X_tr, y_tr, groups_tr, n_splits=5)
    print(f"  CV done in {time.time()-t0:.1f}s, mean_best_iter={best_iter}")

    # Tune decision thresholds
    oof_df = pairs_tr[["s1_id", "cand_id"]].copy()
    oof_df["p"] = oof_preds
    all_s1_list = list(s1_tr["entity_id"])
    rule, oof_score = tune_thresholds(oof_df, gt_map, all_s1_list)
    print(f"  OOF Macro F0.5 = {oof_score:.4f}, rule: t_high={rule[0]}, t_low={rule[1]}, rel={rule[2]}")

    # Train full model
    print("\n[6/7] Training full model...")
    t0 = time.time()
    clf = train_full(X_tr, y_tr, best_iter)
    print(f"  Full model trained in {time.time()-t0:.1f}s")

    # Free train memory
    del X_tr, y_tr, pairs_tr, oof_df, oof_preds; gc.collect()

    # ---- TEST ----
    print("\n" + "=" * 60)
    print("PHASE 2: TEST INFERENCE")
    print("=" * 60)

    print("\n[7/7] Loading test data and predicting...")
    t0 = time.time()
    s1_te = load_source(os.path.join(data_dir, "test", "test_source1.tsv"))
    s2_te = load_source(os.path.join(data_dir, "test", "test_source2.tsv"))
    s3_te = load_source(os.path.join(data_dir, "test", "test_source3.tsv"))
    s23_te = pd.concat([s2_te, s3_te], ignore_index=True)
    del s2_te, s3_te; gc.collect()
    print(f"  Loaded in {time.time()-t0:.1f}s: S1={len(s1_te)}, S23={len(s23_te)}")

    # Blocking
    print("  Blocking test data...")
    t0 = time.time()
    all_candidates_te = {}
    for country in s1_te["country"].unique():
        cands, _, _ = process_country(s1_te, s23_te, country, max_bucket=300, max_cand=80)
        all_candidates_te.update(cands)
    for eid in s1_te["entity_id"]:
        if eid not in all_candidates_te:
            all_candidates_te[eid] = set()
    total_pairs_te = sum(len(v) for v in all_candidates_te.values())
    print(f"  Test blocking done in {time.time()-t0:.1f}s: pairs={total_pairs_te}")

    # Build test pairs
    print("  Building test pairs...")
    t0 = time.time()
    pairs_te = build_pairs_df(all_candidates_te, s1_te, s23_te)
    print(f"  {len(pairs_te)} pairs in {time.time()-t0:.1f}s")

    # Features
    print("  Computing test features...")
    t0 = time.time()
    s1_data_te = make_data_dict(s1_te)
    s23_data_te = make_data_dict(s23_te)

    print("  Building test TF-IDF vectors...")
    name_vecs_te, addr_vecs_te = build_tfidf_vectors(s1_te, s23_te)

    X_te = compute_features_batch(pairs_te, s1_data_te, s23_data_te, name_vecs_te, addr_vecs_te)
    print(f"  Test features done in {time.time()-t0:.1f}s, shape={X_te.shape}")

    # Predict
    print("  Predicting...")
    if len(pairs_te) > 0:
        prob = clf.predict_proba(X_te)[:, 1]
    else:
        prob = np.array([])

    prob_df = pairs_te[["s1_id", "cand_id"]].copy()
    prob_df["p"] = prob

    pred = select_matches(prob_df, *rule)

    # Validate: only keep S2/S3 IDs that exist in test set
    valid_ids = set(s23_te["entity_id"])
    pred = {k: [x for x in v if x in valid_ids] for k, v in pred.items()}

    # Convert candidate sets to lists for output
    cand_out = {k: sorted(v) for k, v in all_candidates_te.items()}

    # Write output
    write_output(list(s1_te["entity_id"]), pred, cand_out, output_dir)

    print(f"\nDone! Total time: {time.time()-t_total:.1f}s")
    print(f"Output written to {output_dir}/matching_results.tsv and {output_dir}/candidate_pairs.tsv")


if __name__ == "__main__":
    main()
