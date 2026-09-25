"""DEV-10 experiment: blocking -> features -> LightGBM -> F_0.5 decision.

Usage: python -m experiments.dev_matcher <cache_dir> [k]
The cache dir holds dev_Q/dev_T/dev_pairs parquet files and dev_region_spec.json.
"""
import json, sys, time
import numpy as np, polars as pl
from ber.blocking import feature_frame, generate_candidates, recall_report
from ber.features import build_features
from ber import model as M
from ber.metrics import macro_f05

cache = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
tag = sys.argv[3] if len(sys.argv) > 3 else "v1"
tic = time.time(); lap = lambda msg: print(f"[{time.time()-tic:6.0f}s] {msg}", flush=True)
Q = pl.read_parquet(f"{cache}/dev_Q.parquet"); T = pl.read_parquet(f"{cache}/dev_T.parquet")
P = pl.read_parquet(f"{cache}/dev_pairs.parquet"); spec = json.load(open(f"{cache}/dev_region_spec.json"))
C = generate_candidates(Q, T, feature_frame(Q), feature_frame(T), k=k, k_noaddr=5, region_spec=spec)
lap(f"blocking k={k}: {recall_report(C, P)}")
F = build_features(C, Q, T, align=(tag != "v1")); lap(f"features: {F.shape}")
y = F.select("q_rid", "t_rid").join(P.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["q_rid", "t_rid"],
                                    how="left", maintain_order="left")["y"].fill_null(0).to_numpy()
# split S1 entities: 70% fit / 10% early stopping / 20% evaluation
rng = np.random.default_rng(42); u = rng.random(Q.height)
part = np.where(u < 0.7, 0, np.where(u < 0.8, 1, 2)); qpart = part[F["q_rid"].to_numpy()]
fit, es, ev = (qpart == 0), (qpart == 1), (qpart == 2)
lap(f"rows fit/es/eval = {fit.sum()}/{es.sum()}/{ev.sum()}, positive rate {y.mean():.4f}")
model = M.train(F.filter(fit), y[fit], F.filter(es), y[es], threads=0)
lap(f"trained: best_iter={model.best_iteration}")
p = M.predict(model, F)
scores = F.select("q_rid", "t_rid").with_columns(pl.Series("p", p))
q_eval = np.flatnonzero(part == 2)
truth_eval = P.join(pl.DataFrame({"q_rid": q_eval.astype(np.uint32)}), on="q_rid", how="semi")
ceiling = macro_f05(C.select("q_rid","t_rid").join(truth_eval, on=["q_rid","t_rid"]), truth_eval, q_eval)
lap(f"oracle on candidates (perfect matcher): F0.5={ceiling['f05']:.4f}")
for excl in (False, True):
    thr, table = M.tune_threshold(scores, truth_eval, q_eval, exclusive=excl)
    best = [r for r in table if r["thr"] == thr][0]
    lap(f"exclusive={excl}: best thr={thr}  " + "  ".join(f"{k2}={v:.4f}" for k2, v in best.items() if k2 not in ("thr","n_entities")))
imp = sorted(zip(model.feature_name(), model.feature_importance("gain")), key=lambda x: -x[1])
tot = sum(v for _, v in imp); print("top features (gain share):", ", ".join(f"{n}={v/tot:.3f}" for n, v in imp[:15]))
model.save_model(f"{cache}/dev_lgbm_{tag}.txt"); scores.write_parquet(f"{cache}/dev_scores_{tag}.parquet")
F.write_parquet(f"{cache}/dev_features_{tag}.parquet"); np.save(f"{cache}/dev_part.npy", part)
