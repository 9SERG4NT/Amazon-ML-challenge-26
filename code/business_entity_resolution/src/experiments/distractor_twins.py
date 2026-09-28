"""Are the test's extra distractors lookalikes of S1 records that are present, or orphans?

A "twin" of a target is an S1 record of the same country with the same core-name key (lower case,
legal forms dropped, tokens sorted) whose address shares at least a given Jaccard share of words
(3+ letters). For a hash sample of targets it reports the twin rate of train matched records and
distractors (all S1 present), of train distractors once 60% of the S1 entities are dropped as in the
test-like universe, and of all test targets. With the test's distractor share (~40%, from the S3 - S2
count) the test rate gives the twin rate of test distractors.

Usage: python -m experiments.distractor_twins <dataset_dir> [sample_mod]   (reads the raw TSVs, ~3 GB RAM)
"""
import sys

import polars as pl

ROOT = sys.argv[1]
MOD = int(sys.argv[2]) if len(sys.argv) > 2 else 33
kw = dict(separator="\t", quote_char=None, infer_schema=False)
LEGAL = ("llc inc corp co ltd pvt pllc llp lp plc pa pc sa sas sasu sarl eurl sci snc public the and of "
         "private limited company corporation incorporated l c p").split()


def name_key(c):
    return (pl.col(c).str.to_lowercase().str.replace_all(r"[^a-z0-9 ]+", " ").str.split(" ")
            .list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(LEGAL))).list.sort().list.join(" "))


def addr_words(c):
    return (pl.col(c).fill_null("").str.to_lowercase().str.replace_all(r"[^a-z ]+", " ").str.split(" ")
            .list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.unique())


def prep(lf):
    return lf.select("entity_id", "country", name_key("business_name").alias("k"), addr_words("business_address").alias("a"))


def best_overlap(t: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    x = t.join(s1.rename({"entity_id": "s1", "a": "a1"}), on=["country", "k"], how="left")
    x = x.with_columns(pl.when(pl.col("a1").is_null() | (pl.col("a").list.len() == 0)).then(0.0)
                       .otherwise(pl.col("a").list.set_intersection("a1").list.len()
                                  / pl.col("a").list.set_union("a1").list.len()).alias("j"))
    return x.group_by("entity_id").agg(pl.col("j").max().alias("best"), pl.col("s1").drop_nulls().n_unique().alias("same_name_s1"))


def summary(df, by):
    return df.group_by(by).agg(pl.len().alias("n"), pl.col("same_name_s1").mean().round(2).alias("s1 same name"),
                               *[(pl.col("best") >= th).mean().round(3).alias(f"twin>={th}") for th in (0.3, 0.5, 0.7)]).sort(by)


samp = pl.col("entity_id").hash(17) % MOD == 0
for split in ("train", "test"):
    s1 = prep(pl.scan_csv(f"{ROOT}/{split}/{split}_source1.tsv", **kw)).collect()
    t = pl.concat([prep(pl.scan_csv(f"{ROOT}/{split}/{split}_source{k}.tsv", **kw).filter(samp)).collect() for k in (2, 3)])
    t = t.filter(pl.col("a").list.len() > 0)
    b = best_overlap(t, s1).join(t.select("entity_id", "country"), on="entity_id")
    if split == "train":
        gt = (pl.scan_csv(f"{ROOT}/train/train_ground_truth.tsv", **kw)
              .select(pl.col("source1_entity_id").alias("owner"), pl.col("matched_entity_ids").str.split(",").alias("entity_id"))
              .explode("entity_id").filter(pl.col("entity_id").is_not_null()).collect())
        b = b.join(gt, on="entity_id", how="left").with_columns(pl.col("owner").is_not_null().alias("matched"))
        print("TRAIN, all S1 present"); print(summary(b, ["country", "matched"]))
        keep = s1.filter(pl.col("entity_id").hash(23) % 10 < 4)  # the test-like universe drops 60% of S1 entities
        bt = best_overlap(t, keep).join(t.select("entity_id", "country"), on="entity_id").join(gt, on="entity_id", how="left")
        bt = (bt.filter(pl.col("owner").is_null() | pl.col("owner").is_in(keep["entity_id"].implode()))
              .with_columns(pl.col("owner").is_not_null().alias("matched")))
        print("TRAIN, 40% of S1 kept (test-like universe)"); print(summary(bt, ["country", "matched"]))
    else:
        print("TEST, all targets"); print(summary(b, ["country"]))
