"""Where a full-data run loses F0.5 on its eval slice: per-entity loss and per-link error breakdown.

Reads a run_pipeline.py work dir (prep split and links, saved stage scores, normalised records),
re-applies the decision rule and splits the eval loss (1 - macro F0.5) by entity outcome, then
breaks the missed links down by why they were missed (never a candidate, lost to another S1
under exclusive assignment, not chosen by the rule) and prints examples of each kind.

Usage: python -m experiments.full_errors <work_dir> <run_key> [score] [rule] [param] [n_examples]
e.g.   python -m experiments.full_errors /opt/mlc26/runs/full NORM-v2__BLK-v4b@20__FEAT-v2__MATCH-v2 p2 gated_ef 0.5 12
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

from ber import model as M
from ber.metrics import macro_f05

EVAL = 2
work, key = Path(sys.argv[1]), sys.argv[2]
score = sys.argv[3] if len(sys.argv) > 3 else "p2"
rule = sys.argv[4] if len(sys.argv) > 4 else "gated_ef"
param = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
n_ex = int(sys.argv[6]) if len(sys.argv) > 6 else 12
prep = work / "prep" / key.split("__")[0]

role = pl.read_parquet(prep / "split.parquet")["role"].to_numpy()
q_eval = np.flatnonzero(role == EVAL).astype(np.uint32)
ids = pl.DataFrame({"q_rid": q_eval})
pairs = pl.read_parquet(prep / "pairs.parquet")
owner = pairs.rename({"q_rid": "owner"})  # every target has at most one true S1
truth = pairs.join(ids, on="q_rid", how="semi")

S = pl.read_parquet(work / "runs" / key / "scores_train.parquet", columns=["q_rid", "t_rid", "src", score]).rename({score: "p"})
best = S.group_by("t_rid").agg(pl.col("p").max().alias("p_best"),
                               pl.col("q_rid").get(pl.col("p").arg_max()).alias("winner"))
S = S.join(ids, on="q_rid", how="semi").join(best, on="t_rid", how="left")
ex = S.filter(pl.col("p") == pl.col("p_best"))
if rule == "threshold":
    pred = ex.filter(pl.col("p") >= param).select("q_rid", "t_rid")
elif rule == "gated_ef":
    pred = M.decide_gated(ex.select("q_rid", "t_rid", "p"), gate=param, exclusive=False)
else:
    pred = M.decide_expected_f(ex.select("q_rid", "t_rid", "p"), floor=param, exclusive=False)
m = macro_f05(pred, truth, q_eval)
print(f"eval S1 {len(q_eval):,}, true links {truth.height:,}, predicted {pred.height:,} | {score} {rule}={param}: "
      f"F0.5={m['f05']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")

# ---- 1. per-entity loss: 1 - macro F0.5 split by what went wrong for the entity
tp = pred.join(truth, on=["q_rid", "t_rid"]).group_by("q_rid").len().rename({"len": "tp"})
E = (ids.join(pred.group_by("q_rid").len().rename({"len": "n_pred"}), on="q_rid", how="left")
     .join(truth.group_by("q_rid").len().rename({"len": "n_true"}), on="q_rid", how="left")
     .join(tp, on="q_rid", how="left").fill_null(0)
     .with_columns(fp=pl.col("n_pred") - pl.col("tp"), fn=pl.col("n_true") - pl.col("tp")))
E = E.with_columns(
    pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
      .when(pl.col("tp") == 0).then(0.0)
      .otherwise(1.25 * (pl.col("tp") / pl.col("n_pred")) * (pl.col("tp") / pl.col("n_true"))
                 / (0.25 * pl.col("tp") / pl.col("n_pred") + pl.col("tp") / pl.col("n_true"))).alias("f"),
    pl.when(pl.col("n_true") == 0).then(pl.when(pl.col("n_pred") == 0).then(pl.lit("singleton ok")).otherwise(pl.lit("singleton, predicted links")))
      .when(pl.col("n_pred") == 0).then(pl.lit("has matches, predicted nothing"))
      .when(pl.col("tp") == 0).then(pl.lit("has matches, all predictions wrong"))
      .when((pl.col("fn") > 0) & (pl.col("fp") > 0)).then(pl.lit("partial: missed and wrong links"))
      .when(pl.col("fn") > 0).then(pl.lit("partial: missed links only"))
      .when(pl.col("fp") > 0).then(pl.lit("partial: wrong links only"))
      .otherwise(pl.lit("perfect")).alias("outcome"))
loss = (E.group_by("outcome").agg(pl.len().alias("entities"), ((1 - pl.col("f")).sum() / len(q_eval)).alias("loss"))
        .sort("loss", descending=True))
print(f"\n1. Loss by entity outcome (total loss {1 - m['f05']:.4f})")
for r in loss.iter_rows(named=True):
    print(f"   {r['outcome']:38s} {r['entities']:>9,} entities  loss {r['loss']:.4f}")

# ---- 2. missed links: why
T = pl.read_parquet(prep / "T_train.parquet", columns=["rid", "src", "has_addr", "name_core", "addr_n"])
Q = pl.read_parquet(prep / "Q_train.parquet", columns=["rid", "country", "name_core", "addr_n"])
fn = (truth.join(pred, on=["q_rid", "t_rid"], how="anti")
      .join(S.select("q_rid", "t_rid", "p", "p_best", "winner"), on=["q_rid", "t_rid"], how="left")
      .join(T.select(pl.col("rid").alias("t_rid"), "src", "has_addr"), on="t_rid", how="left")
      .join(Q.select(pl.col("rid").alias("q_rid"), "country"), on="q_rid", how="left")
      .with_columns(pl.when(pl.col("p").is_null()).then(pl.lit("never a candidate"))
                      .when(pl.col("p") < pl.col("p_best")).then(pl.lit("lost to another S1"))
                      .otherwise(pl.lit("not chosen by the rule")).alias("why")))
print(f"\n2. Missed links: {fn.height:,} of {truth.height:,} ({fn.height / truth.height:.2%})")
for r in (fn.group_by("why").agg(pl.len().alias("n"), pl.col("has_addr").not_().mean().alias("noaddr"),
                                  (pl.col("src") == 2).mean().alias("s2"), (pl.col("country") == "India").mean().alias("india"))
          .sort("n", descending=True).iter_rows(named=True)):
    print(f"   {r['why']:24s} {r['n']:>8,} ({r['n'] / fn.height:5.1%})  no address {r['noaddr']:.2f}  S2 {r['s2']:.2f}  India {r['india']:.2f}")
bins = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]
for why in ("lost to another S1", "not chosen by the rule"):
    h = fn.filter(pl.col("why") == why).select(pl.col("p").cut(bins).alias("bin")).group_by("bin").len().sort("bin")
    print(f"   {why}: own {score} by bin " + ", ".join(f"{b}:{n:,}" for b, n in h.iter_rows()))
lost = fn.filter(pl.col("why") == "lost to another S1")
if lost.height:
    linked = pairs.select(pl.col("q_rid").unique().alias("winner"), pl.lit(True).alias("w_linked"))
    w = (lost.join(linked, on="winner", how="left")
         .join(Q.select(pl.col("rid").alias("q_rid"), pl.col("name_core").alias("qn")), on="q_rid", how="left")
         .join(Q.select(pl.col("rid").alias("winner"), pl.col("name_core").alias("wn")), on="winner", how="left"))
    wroles = role[lost["winner"].to_numpy()]
    print(f"   lost to another S1: winner {score} mean {lost['p_best'].mean():.3f}, gap to winner median "
          f"{(lost['p_best'] - lost['p']).median():.3f}; winner has the same core name {(w['qn'] == w['wn']).mean():.2f}; "
          f"winner is a singleton {w['w_linked'].is_null().mean():.2f}; winner is an eval entity {(wroles == EVAL).mean():.2f}")
nothing = E.filter(pl.col("outcome") == "has matches, predicted nothing").select("q_rid")
if nothing.height:
    top = S.join(nothing, on="q_rid", how="semi").group_by("q_rid").agg(pl.col("p").max().alias("top"))
    why = fn.join(nothing, on="q_rid", how="semi").group_by("why").len().sort("len", descending=True)
    print(f"   entities with matches but nothing predicted: {nothing.height:,}; their best {score}: "
          + ", ".join(f"{b}:{n:,}" for b, n in top.select(pl.col("top").cut(bins).alias("b")).group_by("b").len().sort("b").iter_rows())
          + " | their missed links: " + ", ".join(f"{w_}={n:,}" for w_, n in why.iter_rows()))
by_n = (E.filter(pl.col("n_true") > 0).group_by(pl.col("n_true").clip(upper_bound=7))
        .agg(pl.len().alias("n"), pl.col("f").mean().alias("f"), ((1 - pl.col("f")).sum() / len(q_eval)).alias("loss")).sort("n_true"))
print("   F0.5 by true links per S1: " + ", ".join(f"{r['n_true']}{'+' if r['n_true'] == 7 else ''}: {r['f']:.4f} "
                                               f"(loss {r['loss']:.4f})" for r in by_n.iter_rows(named=True)))

# ---- 3. wrong links: whose target was it?
fp = (pred.join(truth, on=["q_rid", "t_rid"], how="anti").join(owner, on="t_rid", how="left")
      .join(S.select("q_rid", "t_rid", "p"), on=["q_rid", "t_rid"], how="left")
      .join(T.select(pl.col("rid").alias("t_rid"), "has_addr"), on="t_rid", how="left")
      .join(E.select("q_rid", "n_true"), on="q_rid", how="left"))
print(f"\n3. Wrong links: {fp.height:,}; target belongs to another S1 {fp['owner'].is_not_null().mean():.2f}, "
      f"to nobody (distractor) {fp['owner'].is_null().mean():.2f}; no address {fp['has_addr'].not_().mean():.2f}; "
      f"predicted for a singleton {(fp['n_true'] == 0).mean():.2f}")
h = fp.select(pl.col("p").cut(bins).alias("bin")).group_by("bin").len().sort("bin")
print(f"   wrong links by {score}: " + ", ".join(f"{b}:{n:,}" for b, n in h.iter_rows()))

# ---- 4. examples
names = lambda df, qcol, tcol: (df.join(Q.select(pl.col("rid").alias(qcol), pl.col("name_core").alias("qn"), pl.col("addr_n").alias("qa")), on=qcol, how="left")
                                  .join(T.select(pl.col("rid").alias(tcol), pl.col("name_core").alias("tn"), pl.col("addr_n").alias("ta"), pl.col("src").alias("tsrc")), on=tcol, how="left"))


def show(df, title, other=None):
    """Sampled rows as S1 | target; ``other`` names a column holding a rival S1 to print below."""
    print(f"\n--- {title}")
    if df.height == 0:
        return
    for r in names(df.sample(min(n_ex, df.height), seed=1), "q_rid", "t_rid").iter_rows(named=True):
        p = "  -  " if r["p"] is None else f"{r['p']:.3f}"
        print(f"  {score}={p} S{r['tsrc']} | {r['qn'][:42]:42s} | {r['tn'][:42]:42s}")
        print(f"  {'':9s} | {r['qa'][:42]:42s} | {r['ta'][:42]:42s}")
        if other and r.get(other) is not None:
            o = Q.filter(pl.col("rid") == r[other]).row(0, named=True)
            print(f"  {other:9s} | {o['name_core'][:42]:42s} | {o['addr_n'][:42]}")


show(fn.filter(pl.col("why") == "lost to another S1"), "missed: lost to another S1 (S1 | target; winner = the S1 that took it)", "winner")
show(fn.filter((pl.col("why") == "not chosen by the rule") & (pl.col("p") >= 0.2)), f"missed: not chosen by the rule, {score} >= 0.2")
show(fn.filter((pl.col("why") == "not chosen by the rule") & (pl.col("p") < 0.2)), f"missed: not chosen by the rule, {score} < 0.2")
show(fn.filter(pl.col("why") == "never a candidate"), "missed: never a candidate")
show(fp.filter(pl.col("owner").is_null()), "wrong: target belongs to nobody")
show(fp.filter(pl.col("owner").is_not_null()), "wrong: target belongs to another S1 (owner = its true S1)", "owner")
