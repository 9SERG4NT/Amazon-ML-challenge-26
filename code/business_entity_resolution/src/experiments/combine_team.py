"""Combine our pipeline with the team's (bi-encoder + cross-encoder + XGBoost) on the union of both candidate sets.

The two pipelines score 0.978 (ours, M-v28) and 0.980 (the team's) on the public leaderboard, with different candidate
searches (sparse IDF vs dense embeddings), features and learners. A second-stage model learns how to weigh them from
held-out entities that *both* pipelines scored out-of-sample: the team's validation S1 (10% of train, never used by
its models) that are also in our test-like universe (our scores there are out-of-fold or from models that never saw
the entity).

Per union pair (S1, record): each side's probability (missing if the pair is not in that side's candidate set), flags,
and each side's competition context (rank within the S1, margin over the S1's next candidate, margin over the best
other S1 claiming the record). A LightGBM (binary log loss) is cross-fitted over the shared validation entities
(5 folds by S1); the decision rule (threshold, expected F0.5, gated expected F0.5, exclusive assignment) is chosen on
its out-of-fold scores and compared, on the same entities, with each side alone. The output is written only if the
combination wins (or with --force). The candidate file is the union of both candidate sets: every match is in it.

Inputs (--team: the team's stage2 output folder; --ours: two parquet files of ours):
  team:  val_scores.parquet (s1, b, p, y, is_val), test_scores.parquet (s1, b, p): row indices into
         train_/test_s1_ids.parquet and train_/test_pool_ids.parquet (entity ids)
  ours:  ours_train.parquet, ours_test.parquet (s1_id, cand_id, p): out-of-fold / test probabilities
Usage:
  python -m experiments.combine_team --team <dir> --ours-train ours_train.parquet --ours-test ours_test.parquet \
      --truth train_ground_truth.tsv --s1 test_source1.tsv --out <dir> [--threads 8]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

T0 = time.time()
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1,
          "seed": 42, "deterministic": True, "force_row_wise": True}  # same file on every run
RULES = ([("threshold", t) for t in np.round(np.arange(0.30, 0.91, 0.05), 2)]
         + [("expected_f", f) for f in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)]
         + [("gated", g) for g in (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7)])


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def team_pairs(d: Path, split: str) -> pl.DataFrame:
    """The team's scored pairs with entity ids (s1_id, cand_id, p[, y, is_val])."""
    S = pl.read_parquet(d / f"{split if split == 'test' else 'val'}_scores.parquet")
    s1 = pl.read_parquet(d / f"{split}_s1_ids.parquet")["entity_id"]
    pool = pl.read_parquet(d / f"{split}_pool_ids.parquet")["entity_id"]
    S = S.with_columns(pl.Series("s1_id", s1.gather(S["s1"].to_numpy())), pl.Series("cand_id", pool.gather(S["b"].to_numpy())))
    return S.drop("s1", "b").with_columns(pl.col("p").cast(pl.Float64))


def side_context(S: pl.DataFrame, name: str) -> pl.DataFrame:
    """Competition context of one side's probability over that side's own pairs."""
    p = pl.col("p")
    best_other = lambda over: (pl.when(p == p.max().over(over)).then(p.sort(descending=True).slice(1, 1).first().over(over))
                               .otherwise(p.max().over(over)).fill_null(0))
    return S.select("s1_id", "cand_id",
                    p.alias(f"p_{name}"),
                    p.rank("ordinal", descending=True).over("s1_id").cast(pl.Float32).alias(f"{name}_rank_q"),
                    (p - best_other("s1_id")).alias(f"{name}_margin_q"),
                    (p - best_other("cand_id")).alias(f"{name}_margin_t"),
                    p.max().over("s1_id").alias(f"{name}_top_q"),
                    pl.len().over("s1_id").cast(pl.Float32).alias(f"{name}_n_q"))


def union_features(T: pl.DataFrame, O: pl.DataFrame, keep_s1: pl.Series | None) -> pl.DataFrame:
    """Union of both sides' pairs (context computed on each side's full pair set, then restricted to keep_s1)."""
    Tc, Oc = side_context(T, "team"), side_context(O, "ours")
    if keep_s1 is not None:
        k = keep_s1.implode()
        Tc, Oc = Tc.filter(pl.col("s1_id").is_in(k)), Oc.filter(pl.col("s1_id").is_in(k))
    U = Tc.join(Oc, on=["s1_id", "cand_id"], how="full", coalesce=True)
    return U.with_columns(
        pl.col("p_team").is_not_null().cast(pl.Float32).alias("in_team"),
        pl.col("p_ours").is_not_null().cast(pl.Float32).alias("in_ours"),
        pl.mean_horizontal("p_team", "p_ours").alias("p_mean"),
        (pl.col("p_team") - pl.col("p_ours")).alias("p_diff"))


def exclusive(s: pl.DataFrame, col: str) -> pl.DataFrame:
    return s.filter(pl.col(col) == pl.col(col).max().over("cand_id"))


def expected_f(s: pl.DataFrame, col: str, floor: float) -> pl.DataFrame:
    s = s.filter(pl.col(col) >= floor).sort(["s1_id", col, "cand_id"], descending=[False, True, False])
    s = s.with_columns(pl.col(col).cum_sum().over("s1_id").alias("cum_p"), pl.int_range(1, pl.len() + 1).over("s1_id").alias("k"),
                       pl.col(col).sum().over("s1_id").alias("sum_p"),
                       (1 - pl.col(col)).log().sum().over("s1_id").exp().alias("ef0"),
                       ).with_columns((1.25 * pl.col("cum_p") / (0.25 * pl.col("sum_p") + pl.col("k"))).alias("ef"))
    best = (s.group_by("s1_id").agg(pl.col("ef").max().alias("ef_best"), pl.col("ef0").first(),
                                    pl.col("k").get(pl.col("ef").arg_max()).alias("k_best"))
            .filter(pl.col("ef_best") > pl.col("ef0")))
    return s.join(best.select("s1_id", "k_best"), on="s1_id").filter(pl.col("k") <= pl.col("k_best")).select("s1_id", "cand_id")


def decide(s: pl.DataFrame, col: str, rule: str, param: float) -> pl.DataFrame:
    s = exclusive(s.select("s1_id", "cand_id", col).filter(pl.col(col).is_not_null()), col)
    if rule == "threshold":
        return s.filter(pl.col(col) >= param).select("s1_id", "cand_id")
    if rule == "expected_f":
        return expected_f(s, col, param)
    top = s.group_by("s1_id").agg(pl.col(col).max().alias("top")).filter(pl.col("top") >= param)
    return expected_f(s.join(top.select("s1_id"), on="s1_id", how="semi"), col, 0.05)


def macro_f05(pred: pl.DataFrame, links: pl.DataFrame, ids: pl.Series) -> dict:
    n_true = links.group_by("s1_id").len("n_true")
    hit = pred.join(links, on=["s1_id", "cand_id"], how="semi").group_by("s1_id").len("tp")
    e = (pl.DataFrame({"s1_id": ids}).join(n_true, on="s1_id", how="left")
         .join(pred.group_by("s1_id").len("n_pred"), on="s1_id", how="left").join(hit, on="s1_id", how="left").fill_null(0))
    e = e.with_columns(pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
                       .when(pl.col("n_pred") == 0).then(0.0)
                       .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred"))).alias("f"),
                       pl.when(pl.col("n_pred") > 0).then(pl.col("tp") / pl.col("n_pred")).alias("prec"),
                       pl.when(pl.col("n_true") > 0).then(pl.col("tp") / pl.col("n_true")).alias("rec"))
    return {"f05": e["f"].mean(), "precision": e["prec"].mean(), "recall": e["rec"].mean(),
            "f05_singletons": e.filter(pl.col("n_true") == 0)["f"].mean(), "n": e.height}


def best_rule(U: pl.DataFrame, col: str, links, ids) -> tuple:
    res = [((r, p), macro_f05(decide(U, col, r, p), links, ids)) for r, p in RULES]
    return max(res, key=lambda x: x[1]["f05"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", required=True)
    ap.add_argument("--ours-train", required=True)
    ap.add_argument("--ours-test", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--s1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--france-shift", default="none", help="'auto' (the team's empty-share calibration), a number, or 'none'")
    a = ap.parse_args()
    out, team = Path(a.out), Path(a.team)
    out.mkdir(parents=True, exist_ok=True)

    Tv = team_pairs(team, "train")
    Ov = pl.read_parquet(a.ours_train).select("s1_id", "cand_id", pl.col("p").cast(pl.Float64))
    shared = (Tv.filter(pl.col("is_val")).select("s1_id").unique()
              .join(Ov.select("s1_id").unique(), on="s1_id", how="semi")["s1_id"].sort())  # sorted: same folds on every run
    log(f"team validation S1 {Tv.filter(pl.col('is_val'))['s1_id'].n_unique():,}; in our universe too: {shared.len():,}")
    gt = (pl.read_csv(a.truth, separator="\t", quote_char=None, schema_overrides={"matched_entity_ids": pl.Utf8})
          .rename({"source1_entity_id": "s1_id"}).filter(pl.col("s1_id").is_in(shared.implode())))
    links = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
             .filter(pl.col("matched_entity_ids") != "").select("s1_id", pl.col("matched_entity_ids").alias("cand_id")))
    V = union_features(Tv.drop("y", "is_val"), Ov, shared)
    V = V.join(links.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1_id", "cand_id"], how="left").with_columns(pl.col("y").fill_null(0))
    feats = [c for c in V.columns if c not in ("s1_id", "cand_id", "y")]
    cover = {k: int(V[k].sum()) for k in ("in_team", "in_ours")}
    found = links.join(V.select("s1_id", "cand_id"), on=["s1_id", "cand_id"], how="semi").height
    log(f"validation union: {V.height:,} pairs ({cover}), true links found {found:,} of {links.height:,}; "
        f"{len(feats)} features")

    rng = np.random.default_rng(42)
    fold_of = pl.DataFrame({"s1_id": shared, "fold": rng.integers(0, a.folds, shared.len())})
    V = V.join(fold_of, on="s1_id", how="left").sort(["s1_id", "cand_id"])
    X, y, fold = V.select(feats).to_numpy().astype(np.float32), V["y"].to_numpy(), V["fold"].to_numpy()
    p = np.zeros(V.height)
    models = []
    for f in range(a.folds):
        tr, te = fold != f, fold == f
        es = tr & (rng.random(V.height) < 0.1)
        m = lgb.train({**PARAMS, "num_threads": a.threads}, lgb.Dataset(X[tr & ~es], label=y[tr & ~es], feature_name=feats), 3000,
                      valid_sets=[lgb.Dataset(X[es], label=y[es])], callbacks=[lgb.early_stopping(100, verbose=False)])
        p[te] = m.predict(X[te], num_iteration=m.best_iteration, num_threads=a.threads)
        models.append(m)
        log(f"fold {f}: {m.best_iteration} trees")
    V = V.with_columns(pl.Series("p_comb", p))

    res = {}
    for col in ("p_team", "p_ours", "p_mean", "p_comb"):
        r, m = best_rule(V, col, links, shared)
        res[col] = {"rule": r, **m}
        log(f"shared validation, {col:7s}: {r[0]} {r[1]} -> F0.5 {m['f05']:.5f} P {m['precision']:.4f} R {m['recall']:.4f} "
            f"singletons {m['f05_singletons']:.4f}")
    (out / "combine_validation.json").write_text(json.dumps(res, indent=1, default=str))
    best_single = max(res["p_team"]["f05"], res["p_ours"]["f05"])
    if res["p_comb"]["f05"] <= best_single and not a.force:
        log("the combination does not beat the better single pipeline on validation: no output written")
        return

    Tt = team_pairs(team, "test")
    Ot = pl.read_parquet(a.ours_test).select("s1_id", "cand_id", pl.col("p").cast(pl.Float64))
    U = union_features(Tt, Ot, None).sort(["s1_id", "cand_id"])
    Xt = U.select(feats).to_numpy().astype(np.float32)
    pt = np.mean([m.predict(Xt, num_iteration=m.best_iteration, num_threads=a.threads) for m in models], axis=0)
    U = U.with_columns(pl.Series("p_comb", pt))
    U.select("s1_id", "cand_id", "p_comb").write_parquet(out / "test_scores.parquet")
    rule = res["p_comb"]["rule"]
    S1 = pl.read_csv(a.s1, separator="\t", quote_char=None, columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    if a.france_shift != "none":  # label-free: shift French logits until France's no-match share equals the others'
        base = U.join(S1, on="s1_id", how="left")
        q = pl.col("p_comb").clip(1e-6, 1 - 1e-6)
        shifted = lambda d: base.with_columns(pl.when(pl.col("country") == "France")
                                              .then(1 / (1 + (-((q / (1 - q)).log() - d)).exp()))
                                              .otherwise(pl.col("p_comb")).alias("p_comb"))
        n_fr = S1.filter(pl.col("country") == "France").height

        def empty_share(d: float) -> tuple[float, float]:
            linked = decide(shifted(d), "p_comb", *rule).select("s1_id").unique().join(S1, on="s1_id")
            l_fr = linked.filter(pl.col("country") == "France").height
            return 1 - l_fr / n_fr, 1 - (linked.height - l_fr) / (S1.height - n_fr)

        if a.france_shift == "auto":  # bisection on the logit shift
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
        U = shifted(d).drop("country")
    pred = decide(U, "p_comb", *rule)
    lists = lambda d, name: (S1.select("s1_id").join(d.group_by("s1_id").agg(pl.col("cand_id").sort().str.join(",").alias("m")),
                                                     on="s1_id", how="left")
                             .select(pl.col("s1_id").alias("source1_entity_id"), pl.col("m").fill_null("").alias(name)))
    lists(pred, "matched_entity_ids").write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
    lists(U.select("s1_id", "cand_id"), "candidate_entity_ids").write_csv(out / "candidate_pairs.tsv", separator="\t", quote_style="never")
    prof = (S1.join(pred.group_by("s1_id").len("n"), on="s1_id", how="left").with_columns(pl.col("n").fill_null(0))
            .group_by("country").agg((pl.col("n") > 0).mean().alias("linked"), pl.col("n").mean().alias("links")).sort("country"))
    log(f"test: {U.height:,} union pairs ({U.height / S1.height:.2f} per S1; team {int(U['in_team'].sum()):,}, "
        f"ours {int(U['in_ours'].sum()):,}), {pred.height:,} links with {rule[0]} {rule[1]}")
    for r in prof.iter_rows(named=True):
        log(f"  {r['country']}: {r['linked']:.2%} of S1 linked, {r['links']:.3f} links per S1")


if __name__ == "__main__":
    main()
