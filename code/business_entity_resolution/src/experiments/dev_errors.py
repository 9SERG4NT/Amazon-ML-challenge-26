"""Break down DEV-10 eval-slice errors and print examples, to guide feature work.

Usage: python -m experiments.dev_errors <cache_dir> <scores_file> <score_col> <threshold> [n_examples]
e.g.   python -m experiments.dev_errors cache dev_scores_v3.parquet p2 0.7 25
"""
import sys
import numpy as np, polars as pl

cache, file, col, thr = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
n_ex = int(sys.argv[5]) if len(sys.argv) > 5 else 25
S = pl.read_parquet(f"{cache}/{file}").select("q_rid", "t_rid", pl.col(col).alias("p"))
P = pl.read_parquet(f"{cache}/dev_pairs.parquet"); part = np.load(f"{cache}/dev_part.npy")
Q = pl.read_parquet(f"{cache}/dev_Q.parquet", columns=["rid", "business_name", "business_address", "country"])
T = pl.read_parquet(f"{cache}/dev_T.parquet", columns=["rid", "business_name", "business_address", "src",
                                                        "has_addr", "non_ascii", "is_web", "has_alias"])
ids = pl.DataFrame({"q_rid": np.flatnonzero(part == 2).astype(np.uint32)})
truth = P.join(ids, on="q_rid", how="semi")

best_q = S.group_by("t_rid").agg(pl.col("p").max().alias("p_best"))
S = S.join(best_q, on="t_rid").with_columns((pl.col("p") < pl.col("p_best")).alias("lost_excl"))
pred = S.filter(~pl.col("lost_excl") & (pl.col("p") >= thr)).join(ids, on="q_rid", how="semi")

fn = truth.join(pred, on=["q_rid", "t_rid"], how="anti").join(S, on=["q_rid", "t_rid"], how="left")
fp = pred.join(truth, on=["q_rid", "t_rid"], how="anti")
n_fn, n_fp = fn.height, fp.height
print(f"eval links {truth.height:,}; predicted {pred.height:,}; FN {n_fn:,}; FP {n_fp:,}")
print("FN breakdown:")
print(f"  not a candidate (blocking miss): {fn['p'].is_null().sum():,}")
print(f"  lost to another S1 (exclusivity): {fn.filter(pl.col('lost_excl').fill_null(False)).height:,}")
print(f"  below threshold:                 {fn.filter(~pl.col('lost_excl').fill_null(True) & (pl.col('p') < thr)).height:,}")

def show(df, title):
    df = (df.join(Q.rename({"rid": "q_rid", "business_name": "q_name", "business_address": "q_addr"}), on="q_rid")
          .join(T.rename({"rid": "t_rid", "business_name": "t_name", "business_address": "t_addr"}), on="t_rid"))
    for flag in ("has_addr", "non_ascii", "is_web", "has_alias"):
        print(f"  {title}: share with t.{flag} = {df[flag].mean():.3f}")
    print(f"  {title}: by country " + ", ".join(f"{c}={n}" for c, n in df.group_by("country").len().sort("country").rows()))
    for r in df.sample(min(n_ex, df.height), seed=1).sort("p", descending=True, nulls_last=True).iter_rows(named=True):
        p = "  -  " if r["p"] is None else f"{r['p']:.3f}"
        print(f"  p={p} S{r['src']} | {r['q_name'][:40]:40s} | {r['t_name'][:40]:40s}")
        print(f"        {'':3s} | {(r['q_addr'] or '')[:40]:40s} | {(r['t_addr'] or '')[:40]:40s}")

print(f"\n--- false negatives scored in [0.2, {thr}) (the ones a better model could recover)")
show(fn.filter(pl.col("p").is_between(0.2, thr, closed="left") & ~pl.col("lost_excl")), "FN")
print("\n--- false negatives lost to another S1 entity")
show(fn.filter(pl.col("lost_excl").fill_null(False)), "FN-excl")
print("\n--- false positives (highest scored first)")
show(fp, "FP")
