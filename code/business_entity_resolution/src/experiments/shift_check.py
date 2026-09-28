"""Train/test shift diagnostics for a full-data work dir (the leaderboard scored 0.96, the eval slice 0.9813).

Sections (pick with the second argument, comma-separated):
  leak   does file row order or the numeric part of the IDs carry the ground truth? (train)
  stats  per split and country: records, targets per S1, record flags, exact core-name overlap
  preds  per split and country: prediction profile of a run (train rows restricted to the eval slice)

Usage: python -m experiments.shift_check <work_dir> leak,stats[,preds] [data_dir] [run_key] [score] [gate]
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

from ber import model as M
from ber.data import load_truth

work, what = Path(sys.argv[1]), sys.argv[2].split(",")
data = sys.argv[3] if len(sys.argv) > 3 else "/data/dataset"
key = sys.argv[4] if len(sys.argv) > 4 else "NORM-v2__BLK-v4b@20__FEAT-v2__MATCH-v2"
score = sys.argv[5] if len(sys.argv) > 5 else "p2"
gate = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
prep = work / "prep" / key.split("__")[0]
EVAL = 2


def fmean(x: pl.Series) -> float:
    """Mean of a boolean or numeric series, NaN when it is empty."""
    return float(x.mean()) if len(x) else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


if "leak" in what:
    ids = lambda s: pl.read_csv(Path(data) / "train" / f"train_source{s}.tsv", separator="\t", quote_char=None,
                                infer_schema=False, columns=["entity_id"])  # the ID column only: light next to a big job
    Q = ids(1).with_row_index("q_row")
    s2, s3 = ids(2).with_columns(pl.lit(2, pl.UInt8).alias("src")), ids(3).with_columns(pl.lit(3, pl.UInt8).alias("src"))
    n2 = s2.height
    T = pl.concat([s2, s3]).with_row_index("t_row")  # same order as ber.data.load_split
    L = (load_truth(data).join(Q, left_on="s1_id", right_on="entity_id").join(T, left_on="t_id", right_on="entity_id")
         .with_columns(pl.when(pl.col("src") == 3).then(pl.col("t_row") - n2).otherwise(pl.col("t_row")).alias("t_pos"),
                       pl.col("s1_id").str.extract(r"(\d+)$").cast(pl.Int64).alias("q_num"),
                       pl.col("t_id").str.extract(r"(\d+)$").cast(pl.Int64).alias("t_num")))
    print(f"leak check on {L.height:,} train links (Spearman; about 0 means no leak)")
    for s in (2, 3):
        x = L.filter(pl.col("src") == s)
        print(f"  S{s}: row order S1 vs target {spearman(x['q_row'].to_numpy(), x['t_pos'].to_numpy()):+.4f}; "
              f"ID number S1 vs target {spearman(x['q_num'].to_numpy(), x['t_num'].to_numpy()):+.4f}")
    # a leak can also be local: do an entity's S2 and S3 records sit near each other in their files?
    both = (L.group_by("s1_id").agg(pl.col("t_pos").filter(pl.col("src") == 2).first().alias("p2"),
                                    pl.col("t_pos").filter(pl.col("src") == 3).first().alias("p3")).drop_nulls())
    print(f"  S2 vs S3 row of the same entity ({both.height:,} entities): "
          f"{spearman(both['p2'].to_numpy(), both['p3'].to_numpy()):+.4f}")
    idl = L.select(pl.col("s1_id").str.extract(r"^(\w+)-").alias("prefix_q"), pl.col("t_id").str.extract(r"^(\w+)-").alias("prefix_t"),
                   pl.col("s1_id").str.len_chars().alias("len_q"), pl.col("t_id").str.len_chars().alias("len_t"))
    print("  ID shapes: " + str(idl.group_by("prefix_q", "prefix_t", "len_q", "len_t").len().sort("len", descending=True).head(6).rows()))

if "stats" in what:
    rows = []
    for split in ("train", "test"):
        Q = pl.read_parquet(prep / f"Q_{split}.parquet", columns=["rid", "country", "name_core"])
        T = pl.read_parquet(prep / f"T_{split}.parquet",
                            columns=["rid", "country", "src", "name_core", "has_addr", "non_ascii", "is_web", "has_alias", "nums"])
        qn = Q.group_by("country", "name_core").len().rename({"len": "q_freq"})
        for c in sorted(Q["country"].unique().to_list()):
            q, t = Q.filter(pl.col("country") == c), T.filter(pl.col("country") == c)
            tq = t.join(qn.filter(pl.col("country") == c).drop("country"), on="name_core", how="left")
            qf = q.join(qn.filter(pl.col("country") == c).drop("country"), on="name_core", how="left")
            rows.append({"split": split, "country": c, "S1": q.height, "targets": t.height,
                         "targets/S1": t.height / q.height,
                         "S1 name shared": fmean(qf["q_freq"] > 1),
                         "t name = some S1": fmean(tq["q_freq"].is_not_null()),
                         "t no addr": fmean(t["has_addr"].not_()), "t non-ascii": fmean(t["non_ascii"]),
                         "t web": fmean(t["is_web"]), "t alias": fmean(t["has_alias"]),
                         "t no number": fmean(t["nums"].list.len().fill_null(0) == 0)})
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=250, float_precision=3):
        print(pl.DataFrame(rows))

if "preds" in what:
    role = pl.read_parquet(prep / "split.parquet")["role"].to_numpy()
    rows = []
    for split in ("train", "test"):
        S = pl.read_parquet(work / "runs" / key / f"scores_{split}.parquet", columns=["q_rid", "t_rid", score]).rename({score: "p"})
        ex = S.filter(pl.col("p") == pl.col("p").max().over("t_rid"))
        tmax = S.group_by("t_rid").agg(pl.col("p").max().alias("tmax"))
        Q = pl.read_parquet(prep / f"Q_{split}.parquet", columns=["rid", "country"]).rename({"rid": "q_rid"})
        T = pl.read_parquet(prep / f"T_{split}.parquet", columns=["rid", "country"]).rename({"rid": "t_rid"})
        if split == "train":  # the eval slice only, scored like test rows
            Q = Q.filter(pl.Series(role == EVAL))
            ex = ex.join(Q, on="q_rid", how="semi")
        links = M.decide_gated(ex, gate=gate, exclusive=False)
        top = ex.group_by("q_rid").agg(pl.col("p").max().alias("top"))
        per = (Q.join(top, on="q_rid", how="left").join(links.group_by("q_rid").len(), on="q_rid", how="left")
               .with_columns(pl.col("len").fill_null(0), pl.col("top").fill_null(0)))
        lk = links.join(ex, on=["q_rid", "t_rid"]).join(Q, on="q_rid")
        tm = T.join(tmax, on="t_rid", how="left").with_columns(pl.col("tmax").fill_null(0))
        for c in sorted(Q["country"].unique().to_list()):
            p, l, t = per.filter(pl.col("country") == c), lk.filter(pl.col("country") == c), tm.filter(pl.col("country") == c)
            rows.append({"split": "eval" if split == "train" else split, "country": c, "S1": p.height,
                         "linked": fmean(p["len"] > 0), "links/S1": fmean(p["len"]),
                         "top<0.2": fmean(p["top"] < 0.2), "top 0.2-0.8": fmean(p["top"].is_between(0.2, 0.8)),
                         "top>0.95": fmean(p["top"] > 0.95),
                         "link p<0.8": fmean(l["p"] < 0.8), "link p mean": fmean(l["p"]),
                         "t max<0.05": fmean(t["tmax"] < 0.05), "t max>0.5": fmean(t["tmax"] > 0.5)})
        del S, ex, tmax
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=250, float_precision=3):
        print(pl.DataFrame(rows))
    print("train truth: share of targets that belong to no S1 = 0.26; eval truth: 0.944 of S1 linked, 3.46 links/S1")
