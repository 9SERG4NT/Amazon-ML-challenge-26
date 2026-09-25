"""DEV-10 experiment: 2-stage stacking + decision rules, on saved M-v2 features.

Usage: python -m experiments.dev_stack <cache_dir>
"""
import sys, time
import numpy as np, polars as pl
from ber import model as M
from ber.metrics import macro_f05

cache = sys.argv[1]
tic = time.time(); lap = lambda msg: print(f"[{time.time()-tic:6.0f}s] {msg}", flush=True)
F = pl.read_parquet(f"{cache}/dev_features_v2.parquet"); P = pl.read_parquet(f"{cache}/dev_pairs.parquet")
part = np.load(f"{cache}/dev_part.npy")
y = F.select("q_rid", "t_rid").join(P.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["q_rid", "t_rid"],
                                    how="left", maintain_order="left")["y"].fill_null(0).to_numpy()
qr = F["q_rid"].to_numpy(); qpart = part[qr]
fit, es, ev = qpart == 0, qpart == 1, qpart == 2
fold = np.random.default_rng(7).integers(0, 3, len(part))[qr]
q_eval = np.flatnonzero(part == 2)
truth_eval = P.join(pl.DataFrame({"q_rid": q_eval.astype(np.uint32)}), on="q_rid", how="semi")

def report(name, p):
    sc = F.select("q_rid", "t_rid").with_columns(pl.Series("p", p))
    thr, table = M.tune_threshold(sc, truth_eval, q_eval)
    b = [r for r in table if r["thr"] == thr][0]
    lap(f"{name} | threshold {thr}: F0.5={b['f05']:.4f} P={b['precision']:.4f} R={b['recall']:.4f} single={b['f05_singletons']:.4f} multi={b['f05_non_singletons']:.4f}")
    for floor in (0.05, 0.2, 0.4):
        m = macro_f05(M.decide_expected_f(sc, floor=floor), truth_eval, q_eval)
        lap(f"{name} | expected-F (floor {floor}): F0.5={m['f05']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} single={m['f05_singletons']:.4f} multi={m['f05_non_singletons']:.4f}")

p1, models = M.cross_fit_stage1(F, y, fit, es, fold, n_folds=3, threads=0)
lap("stage 1 cross-fitted")
report("stage1", p1)
ctx = M.probability_context(F.select("q_rid", "t_rid", "src").with_columns(pl.Series("p1", p1)))  # includes p1
F2 = pl.concat([F, ctx], how="horizontal")
m2 = M.train(F2.filter(fit), y[fit], F2.filter(es), y[es], threads=0)
lap(f"stage 2 trained, best_iter={m2.best_iteration}")
p2 = M.predict(m2, F2)
report("stage2", p2)
imp = sorted(zip(m2.feature_name(), m2.feature_importance("gain")), key=lambda x: -x[1]); tot = sum(v for _, v in imp)
print("stage-2 top features:", ", ".join(f"{n}={v/tot:.3f}" for n, v in imp[:12]))
F.select("q_rid", "t_rid").with_columns(pl.Series("p1", p1), pl.Series("p2", p2)).write_parquet(f"{cache}/dev_scores_v3.parquet")
m2.save_model(f"{cache}/dev_lgbm_v3_stage2.txt")
