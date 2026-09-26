"""Pair classifier (LightGBM, MIT licence) and the F_0.5 decision step.

Loss: binary cross-entropy (log loss) on (S1, candidate) pairs, which gives
calibrated match probabilities. The set of matches per S1 entity is then
chosen by the decision step below, tuned directly on validation macro F_0.5.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from .metrics import macro_f05
from .partition import by_partition

ID_COLS = ("q_rid", "t_rid")
DEFAULT_PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "max_bin": 255, "verbose": -1, "seed": 42,
}


def feature_names(F: pl.DataFrame) -> list[str]:
    return [c for c in F.columns if c not in ID_COLS]


def train(F_tr: pl.DataFrame, y_tr: np.ndarray, F_va: pl.DataFrame, y_va: np.ndarray,
          params: dict | None = None, rounds: int = 3000, threads: int = 0) -> lgb.Booster:
    cols = feature_names(F_tr)
    return train_arrays(F_tr.select(cols).to_numpy(), y_tr, F_va.select(cols).to_numpy(), y_va, cols,
                        params=params, rounds=rounds, threads=threads)


def train_arrays(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, cols: list[str],
                 params: dict | None = None, rounds: int = 3000, threads: int = 0,
                 w_tr: np.ndarray | None = None, w_va: np.ndarray | None = None) -> lgb.Booster:
    """``train`` on float32 feature matrices whose columns are named ``cols`` (optional row weights)."""
    p = {**DEFAULT_PARAMS, **(params or {}), "num_threads": threads}
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=cols, free_raw_data=True)
    dva = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=dtr)
    return lgb.train(p, dtr, rounds, valid_sets=[dva], valid_names=["val"],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])


XGB_PARAMS = {  # the LightGBM defaults above, in XGBoost terms: leaf-wise trees of up to 255 leaves
    "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "grow_policy": "lossguide",
    "max_depth": 0, "max_leaves": 255, "min_child_weight": 5.0, "eta": 0.05, "subsample": 0.8,
    "colsample_bytree": 0.8, "lambda": 1.0, "max_bin": 256, "seed": 42,
}


class XGBModel:
    """An XGBoost booster behind the LightGBM Booster methods the pipeline uses."""

    def __init__(self, booster, cols: list[str]):
        self.booster, self.cols = booster, cols
        self.best_iteration = int(booster.best_iteration) + 1  # LightGBM counts trees, XGBoost indexes them

    def predict(self, X: np.ndarray, num_threads: int = 0, **_) -> np.ndarray:
        if num_threads > 0:
            self.booster.set_param({"nthread": num_threads})
        return self.booster.inplace_predict(X, iteration_range=(0, self.best_iteration))

    def save_model(self, path: str) -> None:
        self.booster.save_model(str(path).removesuffix(".txt") + ".json")

    def feature_name(self) -> list[str]:
        return list(self.cols)

    def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        g = self.booster.get_score(importance_type="total_gain" if importance_type == "gain" else "weight")
        return np.array([g.get(c, 0.0) for c in self.cols])


def train_arrays_xgb(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, cols: list[str],
                     params: dict | None = None, rounds: int = 3000, threads: int = 0,
                     w_tr: np.ndarray | None = None, w_va: np.ndarray | None = None) -> XGBModel:
    """``train_arrays`` with XGBoost (Apache-2.0, hist trees): same early stopping on the same slice."""
    import xgboost as xgb
    p = {**XGB_PARAMS, **(params or {})}
    if threads > 0:
        p["nthread"] = threads
    dtr = xgb.QuantileDMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=cols, max_bin=p["max_bin"])
    dva = xgb.QuantileDMatrix(X_va, label=y_va, weight=w_va, feature_names=cols, ref=dtr)
    b = xgb.train(p, dtr, rounds, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=200)
    return XGBModel(b, cols)


def _f05(p, r):
    return np.where(p + r > 0, 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12), 0.0)


def fp_cost(k: np.ndarray) -> np.ndarray:
    """F0.5 an S1 with ``k`` true links (all found) loses to one wrong link; a no-match S1 loses everything."""
    return np.where(k == 0, 1.0, 1 - _f05(k / (k + 1), 1.0))


def fn_cost(k: np.ndarray) -> np.ndarray:
    """F0.5 an S1 with ``k`` true links loses to one missed link (no wrong ones); with one link, everything."""
    return 1 - _f05(1.0, (k - 1) / np.maximum(k, 1))


CAT_PARAMS = {  # CatBoost (Apache-2.0): symmetric (oblivious) trees, ordered boosting; same log loss as LightGBM
    "loss_function": "Logloss", "learning_rate": 0.1, "depth": 8, "l2_leaf_reg": 3.0, "border_count": 254,
    "random_seed": 42, "od_type": "Iter", "od_wait": 100, "use_best_model": True,
}


class CatModel:
    """A CatBoost classifier behind the LightGBM Booster methods the pipeline uses."""

    def __init__(self, model, cols: list[str]):
        self.model, self.cols = model, cols
        self.best_iteration = int(model.get_best_iteration()) + 1

    def predict(self, X: np.ndarray, num_threads: int = 0, **_) -> np.ndarray:
        return self.model.predict(X, prediction_type="Probability", thread_count=num_threads or -1)[:, 1].astype(np.float32)

    def save_model(self, path: str) -> None:
        self.model.save_model(str(path).removesuffix(".txt") + ".cbm")

    def feature_name(self) -> list[str]:
        return list(self.cols)

    def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        return np.asarray(self.model.get_feature_importance())


def train_arrays_cat(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, cols: list[str],
                     params: dict | None = None, rounds: int = 3000, threads: int = 0,
                     w_tr: np.ndarray | None = None, w_va: np.ndarray | None = None) -> CatModel:
    """``train_arrays`` with CatBoost: same early stopping on the same slice."""
    from catboost import CatBoostClassifier, Pool
    p = {**CAT_PARAMS, **(params or {}), "iterations": rounds, "thread_count": threads or -1, "verbose": 200}
    m = CatBoostClassifier(**p)
    m.fit(Pool(X_tr, y_tr, weight=w_tr, feature_names=cols), eval_set=Pool(X_va, y_va, weight=w_va, feature_names=cols))
    return CatModel(m, cols)


MLP_PARAMS = {  # a feed-forward network written here in numpy: no deep-learning framework, nothing pretrained
    "hidden": (256, 128), "lr": 1e-3, "batch": 2048, "epochs": 40, "patience": 4, "l2": 1e-6, "seed": 42,
}


def _mlp_stats(X: np.ndarray) -> dict:
    """Input scaling learned on the training rows: signed log1p, then mean/std; columns with gaps get a flag."""
    Z = _signed_log(X)
    mean, std = np.nanmean(Z, 0), np.nanstd(Z, 0)
    return {"mean": np.nan_to_num(mean).astype(np.float32), "std": np.where(std > 1e-6, std, 1).astype(np.float32),
            "nan_cols": np.flatnonzero(np.isnan(Z).any(0))}


def _signed_log(X: np.ndarray) -> np.ndarray:
    Z = np.sign(X) * np.log1p(np.abs(X))
    Z[~np.isfinite(Z)] = np.nan  # an infinite input counts as missing
    return Z


def _mlp_inputs(X: np.ndarray, stats: dict) -> np.ndarray:
    Z = _signed_log(X)
    miss = np.isnan(Z)
    Z = (Z - stats["mean"]) / stats["std"]
    Z[miss] = 0.0
    if len(stats["nan_cols"]):
        Z = np.hstack([Z, miss[:, stats["nan_cols"]]])
    return Z.astype(np.float32)


def _mlp_forward(layers: list, Z: np.ndarray) -> tuple[np.ndarray, list]:
    acts = [Z]
    for W, b in layers[:-1]:
        acts.append(np.maximum(acts[-1] @ W + b, 0))  # ReLU
    W, b = layers[-1]
    return (acts[-1] @ W + b)[:, 0], acts  # logits


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return (0.5 * (1 + np.tanh(0.5 * z))).astype(np.float32)


class MLPModel:
    """A trained feed-forward network behind the LightGBM Booster methods the pipeline uses."""

    def __init__(self, layers: list, stats: dict, cols: list[str], best_iteration: int):
        self.layers, self.stats, self.cols, self.best_iteration = layers, stats, cols, best_iteration

    def predict(self, X: np.ndarray, num_threads: int = 0, **_) -> np.ndarray:
        out = np.empty(len(X), np.float32)
        for s in range(0, len(X), 1 << 19):
            out[s:s + (1 << 19)] = _sigmoid(_mlp_forward(self.layers, _mlp_inputs(X[s:s + (1 << 19)], self.stats))[0])
        return out

    def save_model(self, path: str) -> None:
        arrays = {f"{k}{i}": a for i, (W, b) in enumerate(self.layers) for k, a in (("W", W), ("b", b))}
        np.savez(str(path).removesuffix(".txt") + ".npz", mean=self.stats["mean"], std=self.stats["std"],
                 nan_cols=self.stats["nan_cols"], cols=np.array(self.cols), **arrays)

    def feature_name(self) -> list[str]:
        return list(self.cols)

    def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        """Weight mass of each (standardised) input in the first layer: a rough proxy, not a gain."""
        return np.abs(self.layers[0][0][:len(self.cols)]).sum(1)


def train_arrays_mlp(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, cols: list[str],
                     params: dict | None = None, rounds: int = 3000, threads: int = 0,
                     w_tr: np.ndarray | None = None, w_va: np.ndarray | None = None) -> MLPModel:
    """``train_arrays`` with a neural network: weighted binary cross-entropy, Adam, He initialisation, mini-batches.

    The learning rate halves whenever the early-stopping loss (the same slice LightGBM stops on) fails to improve,
    and training stops after ``patience`` such epochs; the best epoch's weights are kept.
    """
    p = {**MLP_PARAMS, **(params or {})}
    rng = np.random.default_rng(p["seed"])
    stats = _mlp_stats(X_tr)
    Z, Zv = _mlp_inputs(X_tr, stats), _mlp_inputs(X_va, stats)
    y, yv = y_tr.astype(np.float32), y_va.astype(np.float32)
    w = np.ones(len(y), np.float32) if w_tr is None else w_tr.astype(np.float32)
    wv = np.ones(len(yv), np.float32) if w_va is None else w_va.astype(np.float32)
    sizes = [Z.shape[1], *p["hidden"], 1]
    layers = [[(rng.standard_normal((a, b)) * np.sqrt(2 / a)).astype(np.float32), np.zeros(b, np.float32)]
              for a, b in zip(sizes[:-1], sizes[1:])]
    m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
    v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
    b1, b2, eps, lr, t = 0.9, 0.999, 1e-8, p["lr"], 0

    def val_loss() -> float:
        z = np.concatenate([_mlp_forward(layers, Zv[s:s + (1 << 19)])[0] for s in range(0, len(Zv), 1 << 19)])
        ll = np.maximum(z, 0) - z * yv + np.log1p(np.exp(-np.abs(z)))
        return float((ll * wv).sum() / wv.sum())

    best, best_layers, best_epoch, bad = np.inf, None, 0, 0
    for epoch in range(1, p["epochs"] + 1):
        order = rng.permutation(len(y))
        for s in range(0, len(order), p["batch"]):
            idx = order[s:s + p["batch"]]
            z, acts = _mlp_forward(layers, Z[idx])
            g = ((_sigmoid(z) - y[idx]) * w[idx] / w[idx].sum())[:, None]  # d loss / d logit
            t += 1
            for i in reversed(range(len(layers))):
                W, bias = layers[i]
                grads = (acts[i].T @ g + p["l2"] * W, g.sum(0))
                if i:
                    g = (g @ W.T) * (acts[i] > 0)
                for j, gr in enumerate(grads):  # Adam
                    m[i][j] = b1 * m[i][j] + (1 - b1) * gr
                    v[i][j] = b2 * v[i][j] + (1 - b2) * gr * gr
                    layers[i][j] -= lr * (m[i][j] / (1 - b1 ** t)) / (np.sqrt(v[i][j] / (1 - b2 ** t)) + eps)
        loss = val_loss()
        print(f"[mlp epoch {epoch}] val logloss {loss:.6f} lr {lr:.2e}", flush=True)
        if loss < best - 1e-6:
            best, best_epoch, bad = loss, epoch, 0
            best_layers = [[W.copy(), b.copy()] for W, b in layers]
        else:
            bad += 1
            lr /= 2
            if bad >= p["patience"]:
                break
    return MLPModel(best_layers, stats, cols, best_epoch)


def predict(model: lgb.Booster, F: pl.DataFrame) -> np.ndarray:
    return model.predict(F.select(model.feature_name()).to_numpy(), num_threads=0)


def decide(scores: pl.DataFrame, thr: float, exclusive: bool = True) -> pl.DataFrame:
    """Choose links from (q_rid, t_rid, p).

    exclusive: each S2/S3 record goes only to the S1 entity that scores it
    highest (every record belongs to at most one entity).
    """
    s = scores
    if exclusive:
        s = s.filter(pl.col("p") == pl.col("p").max().over("t_rid"))
    return s.filter(pl.col("p") >= thr).select("q_rid", "t_rid")


def tune_threshold(scores: pl.DataFrame, truth: pl.DataFrame, q_ids: np.ndarray,
                   grid=None, exclusive: bool = True) -> tuple[float, list[dict]]:
    grid = grid if grid is not None else np.round(np.arange(0.20, 0.96, 0.025), 3)
    table = []
    for thr in grid:
        m = macro_f05(decide(scores, float(thr), exclusive), truth, q_ids)
        table.append({"thr": float(thr), **m})
    best = max(table, key=lambda r: r["f05"])
    return best["thr"], table


def decide_expected_f(scores: pl.DataFrame, floor: float = 0.05, exclusive: bool = True) -> pl.DataFrame:
    """Per S1 entity, predict the top-k candidates with the k that maximises expected F_0.5.

    With independent match probabilities p_i (candidates sorted by p):
      k = 0 : E[F] = P(no true match) = prod(1 - p_i)
      k >= 1: E[F] ~ 1.25 * sum_{i<=k} p_i / (0.25 * sum_i p_i + k)
    so "no match" is chosen whenever that is the better bet, which is how
    singletons earn their 1.0.
    """
    s = scores
    if exclusive:
        s = s.filter(pl.col("p") == pl.col("p").max().over("t_rid"))
    s = s.filter(pl.col("p") >= floor).sort(["q_rid", "p"], descending=[False, True])
    s = s.with_columns(
        pl.col("p").cum_sum().over("q_rid").alias("cum_p"),
        pl.int_range(1, pl.len() + 1).over("q_rid").alias("k"),
        pl.col("p").sum().over("q_rid").alias("sum_p"),
        (1 - pl.col("p")).log().sum().over("q_rid").exp().alias("ef0"),
    ).with_columns((1.25 * pl.col("cum_p") / (0.25 * pl.col("sum_p") + pl.col("k"))).alias("ef"))
    best = s.group_by("q_rid").agg(pl.col("ef").max().alias("ef_best"), pl.col("ef0").first(),
                                   pl.col("k").get(pl.col("ef").arg_max()).alias("k_best"))
    best = best.filter(pl.col("ef_best") > pl.col("ef0"))
    return s.join(best.select("q_rid", "k_best"), on="q_rid").filter(pl.col("k") <= pl.col("k_best")).select("q_rid", "t_rid")


def decide_gated(scores: pl.DataFrame, gate: float, floor: float = 0.05, exclusive: bool = True) -> pl.DataFrame:
    """Expected-F selection for entities whose best candidate reaches ``gate``; nothing for the rest.

    A threshold protects singletons (an entity with no true match scores 1.0 only when
    nothing is predicted), while expected F picks the set size better for entities that
    do have matches. Gating on the top probability keeps both.
    """
    s = scores
    if exclusive:
        s = s.filter(pl.col("p") == pl.col("p").max().over("t_rid"))
    confident = s.group_by("q_rid").agg(pl.col("p").max().alias("p_top")).filter(pl.col("p_top") >= gate)
    return decide_expected_f(s.join(confident.select("q_rid"), on="q_rid", how="semi"), floor=floor, exclusive=False)


def probability_context(scores: pl.DataFrame, part: np.ndarray | None = None) -> pl.DataFrame:
    """Second-stage features from first-stage probabilities (row order preserved).

    ``scores`` has q_rid, t_rid, src, p1. ``part`` (e.g. ``partition.country_codes``)
    computes them one partition at a time, with the same result and a lower peak memory.
    """
    s = scores.select("q_rid", "t_rid", "src", "p1")
    if part is not None:
        return by_partition(s, part, _probability_context)
    return _probability_context(s)


def _probability_context(s: pl.DataFrame) -> pl.DataFrame:
    best_other = lambda over: (pl.when(pl.col("p1") == pl.col("p1").max().over(over))
                               .then(pl.col("p1").sort(descending=True).slice(1, 1).first().over(over))
                               .otherwise(pl.col("p1").max().over(over)).fill_null(0))
    out = s.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("q_rid").alias("p1_rank_q"),
        pl.col("p1").rank("ordinal", descending=True).over("q_rid", "src").alias("p1_rank_src"),
        pl.col("p1").rank("ordinal", descending=True).over("t_rid").alias("p1_rank_t"),
        best_other("q_rid").alias("p1_best_other_q"),
        best_other("t_rid").alias("p1_best_other_t"),
        best_other(["q_rid", "src"]).alias("p1_best_other_src"),
        pl.col("p1").sum().over("q_rid").alias("p1_sum_q"),
        (pl.col("p1") > 0.5).sum().over("q_rid").alias("p1_n05_q"),
        pl.len().over("t_rid").alias("p1_n_t"),
    ).with_columns(
        (pl.col("p1") - pl.col("p1_best_other_t")).alias("p1_margin_t"),
        (pl.col("p1") - pl.col("p1_best_other_q")).alias("p1_margin_q"),
    )
    return out.drop("q_rid", "t_rid", "src").with_columns(pl.all().cast(pl.Float32))


def cross_fit_stage1(F: pl.DataFrame, y: np.ndarray, fit: np.ndarray, es: np.ndarray, fold: np.ndarray,
                     n_folds: int = 3, threads: int = 0, params: dict | None = None):
    """Out-of-fold first-stage probabilities.

    fit rows get the prediction of the fold model that did not see their S1
    entity; every other row gets the mean of the fold models. Returns (p1, models).
    """
    p1 = np.zeros(F.height)
    rest = ~fit
    models = []
    for f in range(n_folds):
        tr = fit & (fold != f)
        m = train(F.filter(tr), y[tr], F.filter(es), y[es], params=params, threads=threads)
        models.append(m)
        held = fit & (fold == f)
        p1[held] = predict(m, F.filter(held))
        p1[rest] += predict(m, F.filter(rest)) / n_folds
    return p1, models
