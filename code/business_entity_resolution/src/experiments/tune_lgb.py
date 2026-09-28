"""Bayesian hyperparameter search for the stage-1 LightGBM (Optuna TPE, MIT licence): the same kind of search as
SageMaker Automatic Model Tuning, run on our own machine.

Usage (from src/):
    python -m experiments.tune_lgb <work_dir> <features key> [trials] [timeout_s]
    e.g. python -m experiments.tune_lgb /opt/mlc26/runs/full NORM-v2__BLK-v5-tlu40__FEAT-v4 40 1800

Same rows as the matcher (fit and rest entities, fit_rest), same fold draw: each trial fits on folds 1-2 of the fit
entities and early-stops on the early-stopping slice; its score is the best log loss there. The eval slice is never
read, so it stays clean for comparing the tuned matcher (registered as a MATCH version) with M-v11. Trial 0 is the
current parameters (model.DEFAULT_PARAMS with learning rate 0.1), the reference. Writes <work>/tune/<key>.json.
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

LR = 0.1  # the matchers' learning rate (MATCH-v6); fixed so trials stay comparable and quick


def main(work: str, key: str, trials: int = 40, timeout: int = 1800) -> None:
    work = Path(work)
    prep = work / "prep" / key.split("__")[0]
    parts = sorted((work / "feat" / key / "train").glob("part-*.parquet"))
    role = pl.read_parquet(prep / "split.parquet")["role"].to_numpy()
    role = np.where(role == REST, FIT, role).astype(np.int8)
    fold = np.random.default_rng(42 + 1).integers(0, 3, len(role)).astype(np.int8)
    pairs = pl.read_parquet(prep / "pairs.parquet")
    cols = [c for c in pl.read_parquet_schema(parts[0]) if c not in M.ID_COLS]
    X, y, q, _ = _load_rows(parts, np.isin(role, [FIT, ES]), cols, pairs)
    tr, va = (role[q] == FIT) & (fold[q] != 0), role[q] == ES
    print(f"train rows {int(tr.sum()):,}, early-stopping rows {int(va.sum()):,}, {len(cols)} features", flush=True)
    # one binned dataset for every trial (feature_pre_filter off, so min_data_in_leaf may change between trials)
    dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=cols, params={"max_bin": 255, "feature_pre_filter": False},
                      free_raw_data=False)
    dva = lgb.Dataset(X[va], label=y[va], reference=dtr, free_raw_data=False)
    del X

    def fit(params: dict) -> tuple[float, int]:
        p = {**M.DEFAULT_PARAMS, "learning_rate": LR, "num_threads": 0, "feature_pre_filter": False, **params}
        ev = {}
        b = lgb.train(p, dtr, 3000, valid_sets=[dva], valid_names=["val"],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.record_evaluation(ev)])
        return float(min(ev["val"]["binary_logloss"])), int(b.best_iteration)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 31, 1023, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 5000, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100.0, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        }
        t0 = time.time()
        loss, it = fit(params)
        trial.set_user_attr("best_iteration", it)
        trial.set_user_attr("seconds", round(time.time() - t0))
        print(f"trial {trial.number}: logloss {loss:.6f}, {it} trees, {time.time() - t0:.0f}s, {params}", flush=True)
        return loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    d = M.DEFAULT_PARAMS
    study.enqueue_trial({"num_leaves": d["num_leaves"], "min_data_in_leaf": d["min_data_in_leaf"],
                         "feature_fraction": d["feature_fraction"], "bagging_fraction": d["bagging_fraction"],
                         "lambda_l2": d["lambda_l2"], "lambda_l1": 1e-3, "min_gain_to_split": 0.0})
    study.optimize(objective, n_trials=trials, timeout=timeout)
    base = study.trials[0].value
    out = {"key": key, "learning_rate": LR, "optuna": optuna.__version__, "baseline_logloss": base,
           "best_logloss": study.best_value, "improvement": base - study.best_value, "best_params": study.best_params,
           "best_iteration": study.best_trial.user_attrs.get("best_iteration"),
           "trials": [{"n": t.number, "logloss": t.value, **t.params, **t.user_attrs} for t in study.trials
                      if t.value is not None]}
    f = work / "tune" / f"{key}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1))
    print(f"baseline logloss {base:.6f}; best {study.best_value:.6f} (trial {study.best_trial.number}): "
          f"{study.best_params}; {len(out['trials'])} trials -> {f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(v) for v in sys.argv[3:5]))
