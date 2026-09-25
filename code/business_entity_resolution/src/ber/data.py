"""Loading the challenge TSVs and building development samples."""
from __future__ import annotations

from pathlib import Path

import polars as pl

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str | Path) -> pl.DataFrame:
    # quote_char=None: names contain stray quotes; every column stays a string
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def load_split(data_dir: str | Path, split: str):
    """Return (s1, targets) for ``split`` in {"train", "test"}.

    ``targets`` stacks Source 2 and Source 3 with a ``src`` column (2 or 3).
    """
    d = Path(data_dir) / split
    s1 = read_tsv(d / f"{split}_source1.tsv").select(COLS)
    s2 = read_tsv(d / f"{split}_source2.tsv").select(COLS).with_columns(pl.lit(2, pl.UInt8).alias("src"))
    s3 = read_tsv(d / f"{split}_source3.tsv").select(COLS).with_columns(pl.lit(3, pl.UInt8).alias("src"))
    return s1, pl.concat([s2, s3])


def load_truth(data_dir: str | Path) -> pl.DataFrame:
    """Ground-truth links as long pairs (s1_id, t_id). Singletons have no rows."""
    gt = read_tsv(Path(data_dir) / "train" / "train_ground_truth.tsv")
    return (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
            .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").alias("t_id")))


def dev_sample(s1: pl.DataFrame, targets: pl.DataFrame, truth: pl.DataFrame, frac: float, seed: int = 42):
    """Keep ``frac`` of S1 entities, all their true matches, and ``frac`` of the distractors.

    This keeps the matched/distractor ratio of the full data at a fraction of the size.
    """
    s1s = s1.sample(fraction=frac, seed=seed)
    tr = truth.join(s1s.select(pl.col("entity_id").alias("s1_id")), on="s1_id", how="semi")
    matched_any = truth.select(pl.col("t_id").alias("entity_id")).unique()
    distract = targets.join(matched_any, on="entity_id", how="anti").sample(fraction=frac, seed=seed)
    tgt = pl.concat([targets.join(tr.select(pl.col("t_id").alias("entity_id")), on="entity_id", how="semi"), distract])
    return s1s, tgt, tr
