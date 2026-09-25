"""DEV-10 experiment: number-gap features (stage 1) and support features (stage 2).

Baselines from dev_stack on the same splits: stage 1 F0.5=0.9894 (threshold 0.725),
stage 2 F0.5=0.9904 (threshold) / 0.9906 (expected-F).

Usage: python -m experiments.dev_v3 <cache_dir>
"""
import sys, time
import numpy as np, polars as pl
from ber import model as M
from ber.features import number_gap_features, support_features
from ber.metrics import macro_f05

tic = time.time(); lap = lambda msg: print(f"[{time.time()-tic:6.0f}s] {msg}", flush=True)
cache = sys.argv[1]
F = pl.read_parquet(f"{cache}/dev_features_v2.parquet"); P = pl.read_parquet(f"{cache}/dev_pairs.parquet")
part = np.load(f"{cache}/dev_part.npy")
Q = pl.read_parquet(f"{cache}/dev_Q.parquet", columns=["rid", "nums"])
T = pl.read_parquet(f"{cache}/dev_T.parquet", columns=["rid", "name_core", "name_nosp", "addr_n", "nums"])
y = F.select("q_rid", "t_rid").join(P.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["q_rid", "t_rid"],
                                    how="left", maintain_order="left")["y"].fill_null(0).to_numpy()
qr = F["q_rid"].to_numpy(); qpart = part[qr]
fit, es = qpart == 0, qpart == 1
fold = np.random.default_rng(7).integers(0, 3, len(part))[qr]  # same folds as dev_stack
q_eval = np.flatnonzero(part == 2)
truth_eval = P.join(pl.DataFrame({"q_rid": q_eval.astype(np.uint32)}), on="q_rid", how="semi")

nums = (F.select("q_rid", "t_rid")
        .join(Q.rename({"rid": "q_rid", "nums": "a_nums"}), on="q_rid", how="left", maintain_order="left")
        .join(T.select(pl.col("rid").alias("t_rid"), pl.col("nums").alias("b_nums")), on="t_rid", how="left",
              maintain_order="left"))
num = pl.DataFrame(number_gap_features(nums["a_nums"].to_list(), nums["b_nums"].to_list()))
lap(f"number-gap features: share with an unmatched pair {(num['num_x_edit'] >= 0).mean():.3f}")
F1 = pl.concat([F, num], how="horizontal")


def report(name, p):
    sc = F.select("q_rid", "t_rid").with_columns(pl.Series("p", p))
    thr, table = M.tune_threshold(sc, truth_eval, q_eval)
    b = [r for r in table if r["thr"] == thr][0]
    lap(f"{name} | threshold {thr}: F0.5={b['f05']:.4f} P={b['precision']:.4f} R={b['recall']:.4f} "
        f"single={b['f05_singletons']:.4f} multi={b['f05_non_singletons']:.4f}")
    m = macro_f05(M.decide_expected_f(sc, floor=0.05), truth_eval, q_eval)
    lap(f"{name} | expected-F: F0.5={m['f05']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
        f"single={m['f05_singletons']:.4f} multi={m['f05_non_singletons']:.4f}")


def top(m, k=12):
    imp = sorted(zip(m.feature_name(), m.feature_importance("gain")), key=lambda x: -x[1]); tot = sum(v for _, v in imp)
    return ", ".join(f"{n}={v/tot:.3f}" for n, v in imp[:k])


p1, s1 = M.cross_fit_stage1(F1, y, fit, es, fold, n_folds=3, threads=0)
report("stage1 + number gap", p1)
print("stage-1 top features:", top(s1[0]), flush=True)

S = F.select("q_rid", "t_rid", "src").with_columns(pl.Series("p1", p1))
ctx = M.probability_context(S)  # p1 + its context columns
sup = support_features(S, T)
lap(f"support features: rows with anchors {(sup['sup_n'] > 0).mean():.3f}")
for name, F2 in (("stage2 + number gap", pl.concat([F1, ctx], how="horizontal")),
                 ("stage2 + number gap + support", pl.concat([F1, ctx, sup], how="horizontal"))):
    m = M.train(F2.filter(fit), y[fit], F2.filter(es), y[es], threads=0)
    lap(f"{name}: best_iter={m.best_iteration}")
    report(name, M.predict(m, F2))
    print(f"{name} top features:", top(m), flush=True)
