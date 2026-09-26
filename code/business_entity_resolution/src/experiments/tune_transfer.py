"""Tune the stage-1 LightGBM for a country it has never seen (France's situation), with Optuna TPE (MIT licence).

Usage (from src/):
    python -m experiments.tune_transfer <work_dir> <features key> [trials] [timeout_s]

Each trial trains twice: on the US (fit rows, early stopping on US early-stopping rows) scored on India, and on India
scored on the US. The objective is the mean log loss on the unseen country's fit and early-stopping rows, so the eval
slice is never read. Besides the usual parameters the search can switch on what should help an unseen country:
monotone constraints taken from the training country's own correlations (``model.monotone_signs``), extremely
randomised splits (``extra_trees``) and ``path_smooth``. Trial 0 is the current matcher (the reference). Writes
<work>/tune/<key>__transfer.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl

from ber import model as M
from run_pipeline import ES, FIT, REST, _load_rows

LR = 0.1


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main(work: str, key: str, trials: int = 25, timeout: int = 1500) -> None:
    work = Path(work)
    prep = work / "prep" / key.split("__")[0]
    parts = sorted((work / "feat" / key / "train").glob("part-*.parquet"))
    role = pl.read_parquet(prep / "split.parquet")["role"].to_numpy()
    role = np.where(role == REST, FIT, role).astype(np.int8)
    country = pl.read_parquet(prep / "Q_train.parquet", columns=["country"])["country"].to_numpy()
    pairs = pl.read_parquet(prep / "pairs.parquet")
    cols = [c for c in pl.read_parquet_schema(parts[0]) if c not in M.ID_COLS]
    X, y, q, _ = _load_rows(parts, np.isin(role, [FIT, ES]), cols, pairs)
    cq, rq = country[q], role[q]
    dirs = {}
    for src, tgt in (("US", "India"), ("India", "US")):
        tr, va, te = (cq == src) & (rq == FIT), (cq == src) & (rq == ES), cq == tgt
        dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=cols, params={"max_bin": 255, "feature_pre_filter": False},
                          free_raw_data=False)
        dirs[src] = {"dtr": dtr, "dva": lgb.Dataset(X[va], label=y[va], reference=dtr, free_raw_data=False),
                     "Xt": X[te], "yt": y[te], "Xtr": X[tr], "ytr": y[tr], "tgt": tgt}
        print(f"{src} -> {tgt}: train {int(tr.sum()):,}, early stop {int(va.sum()):,}, unseen {int(te.sum()):,} rows", flush=True)
    del X
    signs = {}  # monotone signs per training country and threshold, computed once

    def fit(params: dict, mono: float) -> dict:
        res = {}
        for src, d in dirs.items():
            p = {**M.DEFAULT_PARAMS, "learning_rate": LR, "num_threads": 0, "feature_pre_filter": False, **params}
            if mono > 0:
                if (src, mono) not in signs:
                    signs[(src, mono)] = M.monotone_signs(d["Xtr"], d["ytr"], mono)
                p["monotone_constraints"] = signs[(src, mono)]
            b = lgb.train(p, d["dtr"], 3000, valid_sets=[d["dva"]], valid_names=["val"],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            res[src] = {"in_country": float(b.best_score["val"]["binary_logloss"]),
                        "unseen": logloss(d["yt"], b.predict(d["Xt"], num_iteration=b.best_iteration)),
                        "trees": int(b.best_iteration)}
        return res

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 511, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 20000, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
            "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 100.0),
        }
        mono = trial.suggest_categorical("monotone_min_corr", [0.0, 0.05, 0.1, 0.2])
        t0 = time.time()
        res = fit(params, mono)
        unseen = float(np.mean([r["unseen"] for r in res.values()]))
        trial.set_user_attr("result", res)
        print(f"trial {trial.number}: unseen logloss {unseen:.5f} "
              + " ".join(f"{s}->{d['tgt']} {res[s]['unseen']:.5f} (in {res[s]['in_country']:.5f}, {res[s]['trees']} trees)"
                         for s, d in dirs.items())
              + f" {time.time() - t0:.0f}s {params} mono={mono}", flush=True)
        return unseen

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    d = M.DEFAULT_PARAMS
    study.enqueue_trial({"num_leaves": d["num_leaves"], "min_data_in_leaf": d["min_data_in_leaf"],
                         "feature_fraction": d["feature_fraction"], "bagging_fraction": d["bagging_fraction"],
                         "lambda_l2": d["lambda_l2"], "min_gain_to_split": 0.0, "extra_trees": False,
                         "path_smooth": 0.0, "monotone_min_corr": 0.0})
    study.optimize(objective, n_trials=trials, timeout=timeout)
    base = study.trials[0]
    out = {"key": key, "learning_rate": LR, "optuna": optuna.__version__, "baseline_unseen_logloss": base.value,
           "baseline": base.user_attrs["result"], "best_unseen_logloss": study.best_value,
           "best_params": study.best_params, "best": study.best_trial.user_attrs["result"],
           "trials": [{"n": t.number, "unseen_logloss": t.value, **t.params, "result": t.user_attrs.get("result")}
                      for t in study.trials if t.value is not None]}
    f = work / "tune" / f"{key}__transfer.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1))
    print(f"baseline unseen logloss {base.value:.5f}; best {study.best_value:.5f} (trial {study.best_trial.number}): "
          f"{study.best_params}; {len(out['trials'])} trials -> {f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(v) for v in sys.argv[3:5]))
