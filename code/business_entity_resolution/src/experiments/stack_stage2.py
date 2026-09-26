"""Second stage on top of any first-stage pair scorer: our stage 2, as a standalone script.

Built for the team's bi-encoder + cross-encoder + XGBoost run (leaderboard 0.98), whose write-up lists "a
second-stage stacking model" as not yet tried. In our pipeline the same stage lifts F0.5 from 0.9825 (stage-1
probabilities) to 0.9850 (full data). It learns from the first-stage probabilities of the *rival* candidates: how
far a candidate stands above the next best for its S1, and above the best other S1 that claims the same record.

Inputs (parquet, or TSV/CSV with a header):
  --val   scored pairs of held-out validation S1 entities: s1_id, cand_id, p   (p never trained on these entities)
  --test  scored pairs of the test S1 entities:            s1_id, cand_id, p
  --truth train_ground_truth.tsv (labels the --val pairs and gives each validation S1 its true link count)
  --s1    test_source1.tsv (every test S1 gets a row in the output, and its country for the French shift)
Other numeric columns present in both --val and --test (e.g. the cross-encoder score) become stage-2 features too.
Pass p *before* any French calibration: --france-shift auto redoes that calibration on the stage-2 output.

Steps:
  1. Context features from p, per S1 entity, per (S1, source) and per target record.
  2. LightGBM (binary log loss) cross-fitted over the validation S1 entities (5 folds by entity): out-of-fold p2 on
     validation, the mean of the fold models on test.
  3. The decision rule (threshold, expected F0.5, gated expected F0.5; exclusive assignment: each record goes to
     the S1 that scores it highest) is chosen on validation, for p and for p2 alike. Validation F0.5 of both is
     printed; the output is written from p2 only if p2 wins on validation (or with --force).
  4. --france-shift auto: shift French logits until France's no-match share on test equals the other countries'
     share (the team's empty-share calibration), then apply the chosen rule.

Per-target context needs every S1 that competes for a record. If the validation pairs hold only the validation S1
(their rivals from training entities left out), the per-target features see fewer rivals than on test: the script
prints S1-per-record for both files, and --no-target-context drops those features.

Usage:
  python stack_stage2.py --val val_scored.parquet --test test_scored.parquet \
      --truth dataset/train/train_ground_truth.tsv --s1 dataset/test/test_source1.tsv --out out_dir [--france-shift auto]
Requires numpy, polars, lightgbm.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

T0 = time.time()
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1,
          "seed": 42}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def read(path: str) -> pl.DataFrame:
    if path.endswith(".parquet"):
        return pl.read_parquet(path)
    return pl.read_csv(path, separator="," if path.endswith(".csv") else "\t", quote_char='"' if path.endswith(".csv") else None,
                       infer_schema_length=10000)


def exploded_truth(path: str, ids: pl.Series) -> pl.DataFrame:
    """(s1_id, cand_id) true links of the given S1 entities, and each entity's true link count."""
    gt = pl.read_csv(path, separator="\t", quote_char=None, schema_overrides={"matched_entity_ids": pl.Utf8})
    gt = gt.rename({"source1_entity_id": "s1_id"}).filter(pl.col("s1_id").is_in(ids.implode()))
    links = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
             .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
             .select("s1_id", pl.col("matched_entity_ids").alias("cand_id")))
    n_true = links.group_by("s1_id").len("n_true")
    return links, n_true


def context(s: pl.DataFrame, target: bool) -> pl.DataFrame:
    """Stage-2 features from p (row order preserved): our ``probability_context``, keyed by IDs."""
    s = s.with_columns(pl.col("cand_id").str.slice(0, 2).alias("src"))
    best_other = lambda over: (pl.when(pl.col("p") == pl.col("p").max().over(over))
                               .then(pl.col("p").sort(descending=True).slice(1, 1).first().over(over))
                               .otherwise(pl.col("p").max().over(over)).fill_null(0))
    cols = [pl.col("p").rank("ordinal", descending=True).over("s1_id").alias("p_rank_q"),
            pl.col("p").rank("ordinal", descending=True).over("s1_id", "src").alias("p_rank_src"),
            best_other("s1_id").alias("p_best_other_q"),
            best_other(["s1_id", "src"]).alias("p_best_other_src"),
            pl.col("p").sum().over("s1_id").alias("p_sum_q"),
            (pl.col("p") > 0.5).sum().over("s1_id").alias("p_n05_q"),
            pl.len().over("s1_id").alias("p_n_q")]
    if target:
        cols += [pl.col("p").rank("ordinal", descending=True).over("cand_id").alias("p_rank_t"),
                 best_other("cand_id").alias("p_best_other_t"),
                 pl.len().over("cand_id").alias("p_n_t")]
    out = s.with_columns(cols).with_columns((pl.col("p") - pl.col("p_best_other_q")).alias("p_margin_q"))
    if target:
        out = out.with_columns((pl.col("p") - pl.col("p_best_other_t")).alias("p_margin_t"))
    return out.drop("src")


def exclusive(s: pl.DataFrame, col: str) -> pl.DataFrame:
    return s.filter(pl.col(col) == pl.col(col).max().over("cand_id"))


def expected_f(s: pl.DataFrame, col: str, floor: float) -> pl.DataFrame:
    """Per S1, the top-k candidates with the k that maximises expected F0.5 (empty when that is the better bet)."""
    s = s.filter(pl.col(col) >= floor).sort(["s1_id", col], descending=[False, True])
    s = s.with_columns(pl.col(col).cum_sum().over("s1_id").alias("cum_p"),
                       pl.int_range(1, pl.len() + 1).over("s1_id").alias("k"),
                       pl.col(col).sum().over("s1_id").alias("sum_p"),
                       (1 - pl.col(col)).log().sum().over("s1_id").exp().alias("ef0"),
                       ).with_columns((1.25 * pl.col("cum_p") / (0.25 * pl.col("sum_p") + pl.col("k"))).alias("ef"))
    best = (s.group_by("s1_id").agg(pl.col("ef").max().alias("ef_best"), pl.col("ef0").first(),
                                    pl.col("k").get(pl.col("ef").arg_max()).alias("k_best"))
            .filter(pl.col("ef_best") > pl.col("ef0")))
    return s.join(best.select("s1_id", "k_best"), on="s1_id").filter(pl.col("k") <= pl.col("k_best")).select("s1_id", "cand_id")


def decide(s: pl.DataFrame, col: str, rule: str, param: float) -> pl.DataFrame:
    s = exclusive(s.select("s1_id", "cand_id", col), col)
    if rule == "threshold":
        return s.filter(pl.col(col) >= param).select("s1_id", "cand_id")
    if rule == "expected_f":
        return expected_f(s, col, param)
    top = s.group_by("s1_id").agg(pl.col(col).max().alias("top")).filter(pl.col("top") >= param)  # gated
    return expected_f(s.join(top.select("s1_id"), on="s1_id", how="semi"), col, 0.05)


RULES = ([("threshold", t) for t in np.round(np.arange(0.30, 0.91, 0.05), 2)]
         + [("expected_f", f) for f in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)]
         + [("gated", g) for g in (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7)])


def macro_f05(pred: pl.DataFrame, links: pl.DataFrame, n_true: pl.DataFrame, ids: pl.Series) -> dict:
    """Macro F0.5 over ``ids`` (singletons score 1 only when nothing is predicted)."""
    hit = pred.join(links, on=["s1_id", "cand_id"], how="semi").group_by("s1_id").len("tp")
    n_pred = pred.group_by("s1_id").len("n_pred")
    e = (pl.DataFrame({"s1_id": ids}).join(n_true, on="s1_id", how="left").join(n_pred, on="s1_id", how="left")
         .join(hit, on="s1_id", how="left").fill_null(0))
    e = e.with_columns(
        pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
        .when(pl.col("n_pred") == 0).then(0.0)
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred"))).alias("f"),
        pl.when(pl.col("n_pred") > 0).then(pl.col("tp") / pl.col("n_pred")).alias("prec"),
        pl.when(pl.col("n_true") > 0).then(pl.col("tp") / pl.col("n_true")).alias("rec"))
    single = e.filter(pl.col("n_true") == 0)
    return {"f05": e["f"].mean(), "precision": e["prec"].mean(), "recall": e["rec"].mean(),
            "f05_singletons": single["f"].mean() if single.height else None, "n": e.height}


def best_rule(s: pl.DataFrame, col: str, links, n_true, ids) -> tuple[tuple, dict]:
    res = [((r, p), macro_f05(decide(s, col, r, p), links, n_true, ids)) for r, p in RULES]
    return max(res, key=lambda x: x[1]["f05"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--s1", help="test_source1.tsv: all test S1 ids and their country")
    ap.add_argument("--out", default="stack_out")
    ap.add_argument("--val-ids", help="every validation S1 id (column s1_id), so entities without candidates count too")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-target-context", action="store_true")
    ap.add_argument("--france-shift", default="none", help="'auto' (the team's empty-share calibration), a number, or 'none'")
    ap.add_argument("--force", action="store_true", help="write the p2 output even if p2 loses on validation")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    V = read(a.val).with_columns(pl.col("p").cast(pl.Float64))
    ids = read(a.val_ids)["s1_id"].unique() if a.val_ids else V["s1_id"].unique()
    links, n_true = exploded_truth(a.truth, ids)
    V = V.join(links.with_columns(pl.lit(1, pl.Int8).alias("label")), on=["s1_id", "cand_id"], how="left").with_columns(
        pl.col("label").fill_null(0))
    log(f"validation: {V.height:,} pairs for {ids.len():,} S1; {V['label'].mean():.3f} true; "
        f"{V['label'].sum():,} of {links.height:,} true links are candidates")
    T = read(a.test).with_columns(pl.col("p").cast(pl.Float64)) if a.test else None
    extra = [c for c in V.columns if c not in ("s1_id", "cand_id", "p", "label", "country")
             and V[c].dtype.is_numeric() and (T is None or c in T.columns)]
    target = not a.no_target_context
    V = context(V, target)
    feats = ["p"] + [c for c in V.columns if c.startswith("p_")] + extra
    log(f"stage-2 features ({len(feats)}): {feats}")
    if T is not None:
        T = context(T, target)
        for name, D in (("validation", V), ("test", T)):
            if target:
                log(f"{name}: S1 per record mean {D['p_n_t'].mean():.3f}; candidates per S1 {D.height / D['s1_id'].n_unique():.2f}")

    # cross-fitted stage 2 over the validation entities
    rng = np.random.default_rng(42)
    fold_of = pl.DataFrame({"s1_id": ids, "fold": rng.integers(0, a.folds, ids.len())}).with_columns(pl.col("fold").cast(pl.Int64))
    V = V.join(fold_of, on="s1_id", how="left")
    X = V.select(feats).to_numpy().astype(np.float32)
    y = V["label"].to_numpy()
    fold = V["fold"].to_numpy()
    p2 = np.zeros(V.height)
    p2_test = np.zeros(T.height) if T is not None else None
    Xt = T.select(feats).to_numpy().astype(np.float32) if T is not None else None
    for f in range(a.folds):
        tr, te = fold != f, fold == f
        es = tr & (rng.random(V.height) < 0.1)  # early stopping on 10% of the training rows' pairs
        m = lgb.train({**PARAMS, "num_threads": a.threads}, lgb.Dataset(X[tr & ~es], label=y[tr & ~es], feature_name=feats),
                      3000, valid_sets=[lgb.Dataset(X[es], label=y[es])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        p2[te] = m.predict(X[te], num_iteration=m.best_iteration, num_threads=a.threads)
        if Xt is not None:
            p2_test += m.predict(Xt, num_iteration=m.best_iteration, num_threads=a.threads) / a.folds
        log(f"fold {f}: {m.best_iteration} trees")
    V = V.with_columns(pl.Series("p2", p2))

    (r1, m1) = best_rule(V, "p", links, n_true, ids)
    (r2, m2) = best_rule(V, "p2", links, n_true, ids)
    fmt = lambda r, m: (f"{r[0]} {r[1]}: F0.5 {m['f05']:.4f} P {m['precision']:.4f} R {m['recall']:.4f} "
                        f"singletons {m['f05_singletons']}")
    log("validation, first stage p  -> " + fmt(r1, m1))
    log("validation, stage 2 p2      -> " + fmt(r2, m2))
    use_p2 = m2["f05"] > m1["f05"] or a.force
    col, rule = ("p2", r2) if use_p2 else ("p", r1)
    log(f"using {col} with {rule[0]} {rule[1]}" + ("" if use_p2 or T is None else " (p2 did not win on validation)"))
    if T is None:
        return

    T = T.with_columns(pl.Series("p2", p2_test))
    S1 = pl.read_csv(a.s1, separator="\t", quote_char=None, columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    T = T.join(S1, on="s1_id", how="left")
    if a.france_shift != "none":
        fr = pl.col("country") == "France"
        logit = (pl.col(col).clip(1e-6, 1 - 1e-6) / (1 - pl.col(col).clip(1e-6, 1 - 1e-6))).log()
        shifted = lambda d: T.with_columns(pl.when(fr).then(1 / (1 + (-(logit - d)).exp())).otherwise(pl.col(col)).alias(col))

        def empty_share(d: float) -> tuple[float, float]:
            linked = decide(shifted(d), col, *rule).select("s1_id").unique().join(S1, on="s1_id")
            n_fr, n_ot = S1.filter(pl.col("country") == "France").height, S1.filter(pl.col("country") != "France").height
            l_fr = linked.filter(pl.col("country") == "France").height
            return 1 - l_fr / n_fr, 1 - (linked.height - l_fr) / n_ot

        if a.france_shift == "auto":  # bisection on the logit shift: France's no-match share = the others'
            lo, hi = -2.0, 6.0
            for _ in range(14):
                mid = (lo + hi) / 2
                fr_e, ot_e = empty_share(mid)
                lo, hi = (mid, hi) if fr_e < ot_e else (lo, mid)
            d = (lo + hi) / 2
        else:
            d = float(a.france_shift)
        fr_e, ot_e = empty_share(d)
        log(f"French logit shift {d:.3f}: no-match share France {fr_e:.4f}, others {ot_e:.4f}")
        T = shifted(d)
    pred = decide(T, col, *rule)
    res = (S1.select("s1_id").join(pred.group_by("s1_id").agg(pl.col("cand_id").sort().str.join(",").alias("m")),
                                   on="s1_id", how="left")
           .select(pl.col("s1_id").alias("source1_entity_id"), pl.col("m").fill_null("").alias("matched_entity_ids")))
    res.write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
    prof = (res.join(S1.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id")
            .with_columns(pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element() != ""))
                          .list.len().alias("n"))
            .group_by("country").agg((pl.col("n") > 0).mean().round(4).alias("linked"), pl.col("n").mean().round(3).alias("links_per_s1"))
            .sort("country"))
    log(f"test: {pred.height:,} links -> {out / 'matching_results.tsv'}")
    for r in prof.iter_rows(named=True):
        log(f"  {r['country']}: {r['linked']:.2%} of S1 linked, {r['links_per_s1']:.3f} links per S1")


if __name__ == "__main__":
    main()
