"""Which feature groups hurt an unseen country? Stage-1 LightGBM trained on one country, scored on the other.

Usage (from src/):
    python -m experiments.transfer_ablation <work_dir> <features key>

For the full feature set and for each group dropped in turn, trains on the US and scores India, and trains on India
and scores the US (fit rows, early stopping on the training country's early-stopping rows, current parameters). It
reports the log loss on the unseen country (fit and early-stopping rows only: the eval slice is never read) and in the
training country. Then it tries the union of the groups whose removal helped. Writes <work>/tune/<key>__ablation.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from ber import model as M
from experiments.tune_transfer import logloss
from run_pipeline import ES, FIT, REST, _load_rows

GROUPS = {
    "lengths": ["a_name_len", "b_name_len", "a_addr_ntok", "b_addr_ntok", "nr_len_a", "ar_len_a"],
    "counts": ["a_name_freq", "b_name_freq", "a_name_tfreq", "b_name_tfreq", "t_indegree", "n_cand_q"],
    "flags": ["b_has_addr", "b_is_web", "b_has_alias", "b_non_ascii", "alias_best"],
    "numbers": ["num_jacc", "num_b_in_a", "num_first_in_b", "num_trunc", "num_x_edit", "num_x_loggap", "num_x_relgap"],
    "raw_address": ["a_ratio", "a_tset", "a_tsort", "a_partial", "a_b_in_a", "a_a_in_b"],
    "blocking": ["bscore", "cos_name", "cos_addr"],
    "ranks": ["rank_in_src", "rank_in_q", "gap_to_best_src", "rank_for_t", "margin_vs_other_s1"],
}


def main(work: str, key: str) -> None:
    work = Path(work)
    prep = work / "prep" / key.split("__")[0]
    parts = sorted((work / "feat" / key / "train").glob("part-*.parquet"))
    role = pl.read_parquet(prep / "split.parquet")["role"].to_numpy()
    role = np.where(role == REST, FIT, role).astype(np.int8)
    country = pl.read_parquet(prep / "Q_train.parquet", columns=["country"])["country"].to_numpy()
    cols = [c for c in pl.read_parquet_schema(parts[0]) if c not in M.ID_COLS]
    X, y, q, _ = _load_rows(parts, np.isin(role, [FIT, ES]), cols, pl.read_parquet(prep / "pairs.parquet"))
    cq, rq = country[q], role[q]
    rows = {src: ((cq == src) & (rq == FIT), (cq == src) & (rq == ES), cq == tgt)
            for src, tgt in (("US", "India"), ("India", "US"))}

    def run(drop: list[str]) -> dict:
        keep = [j for j, c in enumerate(cols) if c not in drop]
        names = [cols[j] for j in keep]
        res, t0 = {}, time.time()
        for src, (tr, va, te) in rows.items():
            p = {**M.DEFAULT_PARAMS, "learning_rate": 0.1, "num_threads": 0}
            dtr = lgb.Dataset(X[np.ix_(tr, keep)], label=y[tr], feature_name=names)
            dva = lgb.Dataset(X[np.ix_(va, keep)], label=y[va], reference=dtr)
            b = lgb.train(p, dtr, 3000, valid_sets=[dva], valid_names=["val"], callbacks=[lgb.early_stopping(50, verbose=False)])
            res[src] = {"in_country": float(b.best_score["val"]["binary_logloss"]),
                        "unseen": logloss(y[te], b.predict(X[np.ix_(te, keep)], num_iteration=b.best_iteration))}
        res["unseen_mean"] = float(np.mean([res[s]["unseen"] for s in rows]))
        res["in_country_mean"] = float(np.mean([res[s]["in_country"] for s in rows]))
        print(f"drop {drop or 'nothing'}: unseen {res['unseen_mean']:.5f} (US->India {res['US']['unseen']:.5f}, "
              f"India->US {res['India']['unseen']:.5f}); in-country {res['in_country_mean']:.5f} ({time.time() - t0:.0f}s)",
              flush=True)
        return res

    out = {"baseline": run([])}
    for g, feats in GROUPS.items():
        out[g] = run(feats)
    helped = [g for g in GROUPS if out[g]["unseen_mean"] < out["baseline"]["unseen_mean"] - 0.002]
    if len(helped) > 1:
        out["union:" + "+".join(helped)] = run([f for g in helped for f in GROUPS[g]])
    f = work / "tune" / f"{key}__ablation.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1))
    print(f"groups whose removal helped the unseen country by > 0.002: {helped} -> {f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
