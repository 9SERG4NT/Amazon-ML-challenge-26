"""A finished run's probabilities keyed by entity ids: the inputs of experiments/combine_team.py.

Train universe: out-of-fold (fit entities) or from fold models that never saw the entity (every other entity);
test: the mean of the fold models, as in the run's own predictions. Cross-encoder scores are cross-fitted likewise.

  python -m experiments.export_scores <work> NORM-v2__BLK-v5-tlu40__FEAT-v10__MATCH-v6 <out_prefix>            # stage 2 (p2)
  python -m experiments.export_scores <work> NORM-v2__BLK-v5-tlu40__FEAT-v10__MATCH-v6 <out_prefix> --col p1  # stage 1
  python -m experiments.export_scores <work> NORM-v2__BLK-v5-tlu40 <out_prefix> --col ce3                     # a cross-encoder
  -> <out_prefix>_train.parquet, <out_prefix>_test.parquet with columns s1_id, cand_id, p
"""
from __future__ import annotations

import argparse

import polars as pl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("key", help="a run's version key, or NORM__BLK for a cross-encoder score")
    ap.add_argument("prefix")
    ap.add_argument("--col", default="p2", help="p2, p1, or a cross-encoder name (ce1, ce2, ce3)")
    a = ap.parse_args()
    norm = a.key.split("__")[0]
    for split in ("train", "test"):
        Q = pl.read_parquet(f"{a.work}/prep/{norm}/Q_{split}.parquet", columns=["rid", "entity_id"]).rename({"rid": "q_rid", "entity_id": "s1_id"})
        T = pl.read_parquet(f"{a.work}/prep/{norm}/T_{split}.parquet", columns=["rid", "entity_id"]).rename({"rid": "t_rid", "entity_id": "cand_id"})
        path = (f"{a.work}/extra/{a.key}/{a.col}_{split}.parquet" if a.col.startswith("ce")
                else f"{a.work}/runs/{a.key}/scores_{split}.parquet")
        S = pl.read_parquet(path, columns=["q_rid", "t_rid", a.col])
        S = S.join(Q, on="q_rid").join(T, on="t_rid").select("s1_id", "cand_id", pl.col(a.col).alias("p"))
        S.write_parquet(f"{a.prefix}_{split}.parquet")
        print(split, f"{S.height:,} pairs, {S['s1_id'].n_unique():,} S1", flush=True)


if __name__ == "__main__":
    main()
