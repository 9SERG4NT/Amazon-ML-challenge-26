"""Learn token aliases (variant spelling -> Source 1 spelling) from training links.

For every true (S1, S2/S3) pair we look at the tokens that appear on only one
side. If a target-side token ``x`` is, in most of the pairs that contain it,
accompanied by the same unaligned S1 token ``y``, then ``x`` is an alias of
``y`` — e.g. Devanagari transliterations ("praivet" -> "pvt"), state codes
("tn" -> "tamilnadu") or street abbreviations. Aliases are learned per
country, so they never leak between countries, and an unseen country (France)
simply gets none. Only the provided training data is used.
"""
from __future__ import annotations

import polars as pl
from rapidfuzz.distance import JaroWinkler


def learn_aliases(long_q: pl.DataFrame, long_t: pl.DataFrame, pairs: pl.DataFrame,
                  min_count: int = 25, min_ratio: float = 0.5, max_unaligned: int = 4) -> pl.DataFrame:
    """Return a (country, tok, to) alias table.

    long_q / long_t: token frames (rid, country, tok) of S1 / target records.
    pairs: true links as (q_rid, t_rid).
    """
    P = pairs.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("pid"), "q_rid", "t_rid")
    Q = long_q.select(pl.col("rid").alias("q_rid"), "country", "tok").unique()
    T = long_t.select(pl.col("rid").alias("t_rid"), "tok").unique()
    Pq = P.join(Q, on="q_rid").select("pid", "country", "tok")
    Pt = P.join(T, on="t_rid").select("pid", "tok")

    ua = Pq.join(Pt, on=["pid", "tok"], how="anti")
    ub = Pt.join(Pq.select("pid", "tok"), on=["pid", "tok"], how="anti")
    # pairs with many unaligned tokens are too noisy to align reliably
    ok_a = ua.group_by("pid").len().filter(pl.col("len") <= max_unaligned).select("pid")
    ok_b = ub.group_by("pid").len().filter(pl.col("len") <= max_unaligned).select("pid")
    ua = ua.join(ok_a, on="pid", how="semi")
    ub = ub.join(ok_b, on="pid", how="semi")

    co = ub.join(ua.rename({"tok": "to"}), on="pid")
    cnt = co.group_by("country", "tok", "to").agg(pl.len().alias("n")).filter(pl.col("n") >= min_count)
    # Two unaligned tokens often co-occur ("pra li" <-> "pvt ltd"); among candidates with
    # similar support, prefer the one spelled most alike.
    sim = [JaroWinkler.normalized_similarity(a, b) for a, b in zip(cnt["tok"].to_list(), cnt["to"].to_list())]
    cnt = cnt.with_columns(pl.Series("sim", sim, dtype=pl.Float32))
    cnt = cnt.with_columns((pl.col("n") * (1 + pl.col("sim"))).alias("rank_score"))
    best = cnt.sort("rank_score", descending=True).group_by("country", "tok").first()

    pair_country = Pq.select("pid", "country").unique()
    occ = Pt.join(pair_country, on="pid").group_by("country", "tok").agg(pl.len().alias("occ"))
    best = best.join(occ, on=["country", "tok"]).with_columns((pl.col("n") / pl.col("occ")).alias("ratio"))
    # Never alias a token that S1 itself uses often (e.g. a real city name).
    n_pairs = pair_country.group_by("country").agg(pl.len().alias("np"))
    q_occ = Pq.group_by("country", "tok").agg(pl.len().alias("q_occ"))
    best = (best.join(q_occ, on=["country", "tok"], how="left").join(n_pairs, on="country")
            .with_columns(pl.col("q_occ").fill_null(0)))
    alias = best.filter((pl.col("ratio") >= min_ratio) & (pl.col("tok") != pl.col("to"))
                        & (pl.col("q_occ") < 0.25 * pl.col("occ")))
    return alias.select("country", "tok", "to", "n", "ratio").sort("country", "n", descending=[False, True])
