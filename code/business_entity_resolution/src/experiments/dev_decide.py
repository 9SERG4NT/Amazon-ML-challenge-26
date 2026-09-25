"""DEV-10 experiment: decision rules on saved scores.

Threshold is best for singletons, expected-F for entities with matches; the gated rule
tries to get both: say nothing unless the best candidate reaches ``gate``, then pick
the set size by expected F_0.5.

Usage: python -m experiments.dev_decide <cache_dir> <scores_file> <score_col>
"""
import sys, time
import numpy as np, polars as pl
from ber import model as M
from ber.metrics import macro_f05

tic = time.time(); lap = lambda msg: print(f"[{time.time()-tic:6.0f}s] {msg}", flush=True)
cache, file, col = sys.argv[1], sys.argv[2], sys.argv[3]
S = pl.read_parquet(f"{cache}/{file}").select("q_rid", "t_rid", pl.col(col).alias("p"))
P = pl.read_parquet(f"{cache}/dev_pairs.parquet"); part = np.load(f"{cache}/dev_part.npy")
q_eval = np.flatnonzero(part == 2)
ids = pl.DataFrame({"q_rid": q_eval.astype(np.uint32)})
truth = P.join(ids, on="q_rid", how="semi")
ex = S.filter(pl.col("p") == pl.col("p").max().over("t_rid")).join(ids, on="q_rid", how="semi")
fmt = lambda m: (f"F0.5={m['f05']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
                 f"single={m['f05_singletons']:.4f} multi={m['f05_non_singletons']:.4f}")

rows = []
for thr in np.round(np.arange(0.40, 0.91, 0.025), 3):
    rows.append(("threshold", thr, macro_f05(ex.filter(pl.col("p") >= thr), truth, q_eval)))
for floor in (0.02, 0.05, 0.1, 0.2):
    rows.append(("expected_f", floor, macro_f05(M.decide_expected_f(ex, floor=floor, exclusive=False), truth, q_eval)))
for gate in np.round(np.arange(0.30, 0.96, 0.05), 3):
    rows.append(("gated_ef", gate, macro_f05(M.decide_gated(ex, gate=gate), truth, q_eval)))
for rule in ("threshold", "expected_f", "gated_ef"):
    r = max((x for x in rows if x[0] == rule), key=lambda x: x[2]["f05"])
    lap(f"{col} best {rule}={r[1]}: " + fmt(r[2]))
print("gated_ef grid:", " ".join(f"{g}:{m['f05']:.4f}" for rule, g, m in rows if rule == "gated_ef"))
