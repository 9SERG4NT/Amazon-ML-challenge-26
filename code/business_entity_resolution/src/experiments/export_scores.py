"""A finished run's stage-2 probabilities keyed by entity ids: the "ours" input of experiments/combine_team.py.

Train universe: out-of-fold (fit entities) or from fold models that never saw the entity (every other entity);
test: the mean of the fold models, as in the run's own predictions.

  python -m experiments.export_scores <work> NORM-v2__BLK-v5-tlu40__FEAT-v10__MATCH-v6 <out_prefix>
  -> <out_prefix>_train.parquet, <out_prefix>_test.parquet with columns s1_id, cand_id, p
"""
from __future__ import annotations

import sys

import polars as pl


def main() -> None:
    work, key, prefix = sys.argv[1:4]
    norm = key.split("__")[0]
    for split in ("train", "test"):
        Q = pl.read_parquet(f"{work}/prep/{norm}/Q_{split}.parquet", columns=["rid", "entity_id"]).rename({"rid": "q_rid", "entity_id": "s1_id"})
        T = pl.read_parquet(f"{work}/prep/{norm}/T_{split}.parquet", columns=["rid", "entity_id"]).rename({"rid": "t_rid", "entity_id": "cand_id"})
        S = pl.read_parquet(f"{work}/runs/{key}/scores_{split}.parquet", columns=["q_rid", "t_rid", "p2"])
        S = S.join(Q, on="q_rid").join(T, on="t_rid").select("s1_id", "cand_id", pl.col("p2").alias("p"))
        S.write_parquet(f"{prefix}_{split}.parquet")
        print(split, f"{S.height:,} pairs, {S['s1_id'].n_unique():,} S1", flush=True)


if __name__ == "__main__":
    main()
