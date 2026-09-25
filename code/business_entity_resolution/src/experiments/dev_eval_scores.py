"""Score saved DEV-10 predictions under each decision rule.

Usage: python -m experiments.dev_eval_scores <cache_dir> <tag>   (reads dev_scores_<tag>.parquet)
"""
import sys
import numpy as np, polars as pl
from ber import model as M
from ber.metrics import macro_f05

cache, tag = sys.argv[1], sys.argv[2]
S = pl.read_parquet(f"{cache}/dev_scores_{tag}.parquet")
P = pl.read_parquet(f"{cache}/dev_pairs.parquet"); part = np.load(f"{cache}/dev_part.npy")
q_eval = np.flatnonzero(part == 2)
truth_eval = P.join(pl.DataFrame({"q_rid": q_eval.astype(np.uint32)}), on="q_rid", how="semi")
fmt = lambda m: (f"F0.5={m['f05']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
                 f"single={m['f05_singletons']:.4f} multi={m['f05_non_singletons']:.4f}")
for excl in (False, True):
    thr, table = M.tune_threshold(S, truth_eval, q_eval, exclusive=excl)
    print(f"{tag} threshold exclusive={excl}: thr={thr} " + fmt([r for r in table if r["thr"] == thr][0]), flush=True)
for floor in (0.05, 0.2, 0.4):
    print(f"{tag} expected-F floor={floor}: " + fmt(macro_f05(M.decide_expected_f(S, floor=floor), truth_eval, q_eval)), flush=True)
