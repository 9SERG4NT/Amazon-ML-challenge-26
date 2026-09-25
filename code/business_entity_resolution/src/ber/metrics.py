"""The challenge metric: macro-averaged F_0.5 over Source 1 entities."""
from __future__ import annotations

import numpy as np
import polars as pl


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, q_ids: np.ndarray) -> dict:
    """Score predicted (q_rid, t_rid) links against true links over the S1 ids ``q_ids``.

    Per entity: no true matches -> 1.0 if nothing is predicted, else 0.0.
    True matches but nothing (or nothing correct) predicted -> 0.0.
    Otherwise F_0.5 = 1.25 P R / (0.25 P + R). The mean over ``q_ids`` includes
    entities with no candidates at all.
    """
    ids = pl.DataFrame({"q_rid": np.asarray(q_ids, dtype=np.uint32)})
    p = pred.select("q_rid", "t_rid").join(ids, on="q_rid", how="semi")
    t = truth.select("q_rid", "t_rid").join(ids, on="q_rid", how="semi")
    tp = p.join(t, on=["q_rid", "t_rid"]).group_by("q_rid").agg(pl.len().alias("tp"))
    df = (ids.join(p.group_by("q_rid").agg(pl.len().alias("n_pred")), on="q_rid", how="left")
          .join(t.group_by("q_rid").agg(pl.len().alias("n_true")), on="q_rid", how="left")
          .join(tp, on="q_rid", how="left").fill_null(0))
    P = pl.col("tp") / pl.col("n_pred")
    R = pl.col("tp") / pl.col("n_true")
    df = df.with_columns(
        pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
          .when(pl.col("tp") == 0).then(0.0)
          .otherwise(1.25 * P * R / (0.25 * P + R)).alias("f05"),
        pl.when(pl.col("n_pred") > 0).then(P).alias("prec"),
        pl.when(pl.col("n_true") > 0).then(R).alias("rec"),
    )
    single = df.filter(pl.col("n_true") == 0)
    multi = df.filter(pl.col("n_true") > 0)
    return {
        "f05": float(df["f05"].mean()),
        "precision": float(df["prec"].mean() or 0),   # over entities with a prediction
        "recall": float(df["rec"].mean() or 0),       # over entities with true matches
        "f05_singletons": float(single["f05"].mean()) if single.height else float("nan"),
        "f05_non_singletons": float(multi["f05"].mean()) if multi.height else float("nan"),
        "n_entities": df.height,
    }
