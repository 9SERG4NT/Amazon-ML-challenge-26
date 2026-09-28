"""Window features computed one country at a time, rows kept in their original order.

Blocking runs per country, so a candidate never pairs records of two countries and
every window over an S1 record or a target stays inside one country. Computing such
features country by country gives the same values while bounding the memory of the
window intermediates (deep blocking gives ~200M train candidates on the full data).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl


def country_codes(q_rid: np.ndarray, Q: pl.DataFrame) -> np.ndarray:
    """The country of each candidate's S1 record, as a small integer code."""
    _, codes = np.unique(Q["country"].fill_null("").to_numpy(), return_inverse=True)
    lut = np.zeros(int(Q["rid"].max()) + 1, np.uint8)
    lut[Q["rid"].to_numpy()] = codes
    return lut[q_rid]


def by_partition(frame: pl.DataFrame, part: np.ndarray,
                 fn: Callable[[pl.DataFrame], pl.DataFrame]) -> pl.DataFrame:
    """``fn(frame)`` computed on the rows of each partition, returned in the original row order.

    Exact when no window of ``fn`` crosses partitions: rows keep their relative order
    inside a partition, so ordinal ranks break ties as they would on the whole frame.
    """
    keys = np.unique(part)
    if len(keys) < 2:
        return fn(frame)
    cols: dict[str, np.ndarray] = {}
    for k in keys:
        pos = np.flatnonzero(part == k)
        res = fn(frame[pos])
        for c in res.columns:
            v = res.get_column(c).to_numpy()
            if c not in cols:
                cols[c] = np.empty(frame.height, v.dtype)
            cols[c][pos] = v
        del res
    return pl.DataFrame(cols, nan_to_null=True)
