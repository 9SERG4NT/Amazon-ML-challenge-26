"""Candidate generation (blocking).

Each record becomes a sparse vector of hashed features — name tokens, address
tokens, typo-tolerant variants of both (4-char prefix, consonant skeleton) and
house-number prefixes/suffixes. Features are IDF-weighted on the target side,
the name and address blocks are L2-normalised separately, and for every S1
record we keep the top-K records of each target source by
``w_name * cos(name) + w_addr * cos(addr)`` — computed exactly with a
multi-threaded sparse top-n product, within each country.
"""
from __future__ import annotations

import os
import re

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

NBITS = 24
HALF = 1 << (NBITS - 1)  # name features hash into [0, HALF), address into [HALF, 2*HALF)
_VOWELS = re.compile(r"[aeiou]")
_REPEAT = re.compile(r"(.)\1+")


def _skeleton(t: str) -> str:
    return _REPEAT.sub(r"\1", t[0] + _VOWELS.sub("", t[1:]))


def _variants(vocab: pl.Series, prefix: str) -> pl.DataFrame:
    """Typo-tolerant variants for alphabetic tokens of length >= 5."""
    toks = [t for t in vocab.to_list() if len(t) >= 5 and t.isalpha()]
    rows_t, rows_f = [], []
    for t in toks:
        rows_t += [t, t]
        rows_f += [f"{prefix}4:{t[:4]}", f"{prefix}k:{_skeleton(t)}"]
    return pl.DataFrame({"t": rows_t, "f": rows_f}, schema={"t": pl.Utf8, "f": pl.Utf8})


def _number_variants(vocab: pl.Series) -> pl.DataFrame:
    """Leading/trailing 3 digits of long numbers: '7530' ~ '530', '2213' ~ '221'."""
    toks = [t for t in vocab.to_list() if len(t) >= 4]
    return pl.DataFrame({"t": toks * 2, "f": [f"#p:{t[:3]}" for t in toks] + [f"#s:{t[-3:]}" for t in toks]},
                        schema={"t": pl.Utf8, "f": pl.Utf8})


def _char_grams(vocab: pl.Series, n: int = 3) -> pl.DataFrame:
    """Character n-grams of the space-free core name (robust to typos and joined words)."""
    toks, feats = [], []
    for s in vocab.to_list():
        p = f"^{s}$"
        for g in {p[i:i + n] for i in range(len(p) - n + 1)}:
            toks.append(s)
            feats.append(f"g:{g}")
    return pl.DataFrame({"s": toks, "f": feats}, schema={"s": pl.Utf8, "f": pl.Utf8})


def feature_frame(df: pl.DataFrame, char_grams: bool = True, joined_name: bool = True) -> pl.DataFrame:
    """Long frame (rid, fid) of hashed blocking features, one row per distinct feature."""
    nt = (df.select("rid", pl.col("name_n").str.split(" ").alias("t")).explode("t").filter(pl.col("t") != ""))
    at = (df.select("rid", pl.col("addr_n").str.split(" ").alias("t")).explode("t").filter(pl.col("t") != ""))
    nvar = _variants(nt["t"].unique(), "n")
    avocab = at["t"].unique()
    avar = pl.concat([_variants(avocab, "a"), _number_variants(avocab.filter(avocab.str.contains(r"^\d+$")))])
    nosp = df.select("rid", pl.col("name_nosp").alias("s")).filter(pl.col("s") != "")
    parts = [nt.select("rid", ("n:" + pl.col("t")).alias("f")),
             nt.join(nvar, on="t").select("rid", "f")]
    if joined_name:
        parts.append(nosp.select("rid", ("w:" + pl.col("s")).alias("f")))
    if char_grams:
        parts.append(nosp.join(_char_grams(nosp["s"].unique()), on="s").select("rid", "f"))
    name_f = pl.concat(parts)
    addr_f = pl.concat([at.select("rid", ("a:" + pl.col("t")).alias("f")),
                        at.join(avar, on="t").select("rid", "f")])
    name_f = name_f.select("rid", (pl.col("f").hash(seed=17) % HALF).cast(pl.Int64).alias("fid"))
    addr_f = addr_f.select("rid", (pl.col("f").hash(seed=17) % HALF + HALF).cast(pl.Int64).alias("fid"))
    return pl.concat([name_f, addr_f]).unique()


def _matrix(rows: np.ndarray, cols: np.ndarray, n_rows: int, idf: np.ndarray,
            w_name: float = 1.0, w_addr: float = 1.0) -> sp.csr_matrix:
    w = idf[cols]
    keep = w > 0
    rows, cols, w = rows[keep], cols[keep], w[keep]
    block = (cols >= HALF).astype(np.int64)
    key = rows.astype(np.int64) * 2 + block
    ss = np.bincount(key, weights=w * w, minlength=n_rows * 2)
    w = w / np.sqrt(ss[key])
    w = w * np.where(block == 1, w_addr, w_name)
    return sp.csr_matrix((w.astype(np.float32), (rows, cols)), shape=(n_rows, 1 << NBITS))


def _last_token() -> pl.Expr:
    return pl.col("addr_n").str.split(" ").list.last()


def learn_region_merges(Q: pl.DataFrame, T: pl.DataFrame, pairs: pl.DataFrame,
                        min_share: float = 0.002, min_cross: int = 10, min_last_ratio: float = 0.6) -> dict:
    """Learn how to split the search by region, from training data only.

    Region keys are the frequent last tokens of S1 addresses (states). Keys that
    true links cross (e.g. Telangana records filed under Andhra Pradesh) are
    merged. Only countries with training links are split; an unseen country
    keeps a full search, since its keys can't be validated.
    """
    keys = region_keys(Q, min_share, min_last_ratio)
    kq = _record_keys(Q, keys).rename({"rid": "q_rid", "k": "kq"})
    kt = _record_keys(T, keys).rename({"rid": "t_rid", "k": "kt"})
    m = pairs.join(kq, on="q_rid").join(kt, on="t_rid")
    shared = m.filter(pl.col("kq") == pl.col("kt")).select("q_rid", "t_rid").unique()
    cross = (m.join(shared, on=["q_rid", "t_rid"], how="anti").group_by("country", "kq", "kt").len()
             .filter(pl.col("len") >= min_cross))
    return {"countries": sorted(Q.join(pairs.select(pl.col("q_rid").alias("rid")).unique(), on="rid", how="semi")
                                ["country"].unique().to_list()),
            "merges": [(c, a, b) for c, a, b, _ in cross.rows()], "min_share": min_share,
            "min_last_ratio": min_last_ratio}


def region_keys(Q: pl.DataFrame, min_share: float, min_last_ratio: float = 0.6) -> pl.DataFrame:
    """Frequent tokens that sit at the end of S1 addresses most of the times they occur.

    The positional test keeps states and drops common words that merely end
    some addresses ("rd", "nagar").
    """
    last = Q.select("country", _last_token().alias("k")).group_by("country", "k").agg(pl.len().alias("n_last"))
    anywhere = (Q.select("rid", "country", pl.col("addr_n").str.split(" ").alias("k")).explode("k")
                .unique(["rid", "k"]).group_by("country", "k").agg(pl.len().alias("n_any")))
    tot = Q.group_by("country").len().rename({"len": "n"})
    return (last.join(anywhere, on=["country", "k"]).join(tot, on="country")
            .filter((pl.col("n_last") >= min_share * pl.col("n"))
                    & (pl.col("n_last") >= min_last_ratio * pl.col("n_any")))
            .select("country", "k"))


def _record_keys(df: pl.DataFrame, keys: pl.DataFrame) -> pl.DataFrame:
    return (df.select("rid", "country", pl.col("addr_n").str.split(" ").alias("k")).explode("k")
            .join(keys, on=["country", "k"], how="semi").unique(["rid", "k"]))


def _groups(Q: pl.DataFrame, T: pl.DataFrame, spec: dict | None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(rid, country, g) region-group memberships for S1 and targets, after merges."""
    empty = pl.DataFrame(schema={"rid": pl.UInt32, "country": pl.Utf8, "g": pl.Utf8})
    if not spec:
        return empty, empty
    keys = (region_keys(Q, spec["min_share"], spec.get("min_last_ratio", 0.6))
            .filter(pl.col("country").is_in(spec["countries"])))
    parent = {}  # union-find over (country, key)

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    for c, a, b in spec["merges"]:
        ra, rb = find((c, a)), find((c, b))
        if ra != rb:
            parent[ra] = rb
    kmap = keys.with_columns(pl.struct("country", "k").map_elements(
        lambda s: find((s["country"], s["k"]))[1], return_dtype=pl.Utf8).alias("g"))
    gq = _record_keys(Q, keys).join(kmap, on=["country", "k"]).select("rid", "country", "g").unique()
    gt = _record_keys(T, keys).join(kmap, on=["country", "k"]).select("rid", "country", "g").unique()
    return gq, gt


def _block_cosines(Qm, Tm, qi, ti, chunk: int = 2_000_000):
    """Row-wise name-block and address-block cosines for the pairs (qi[j], ti[j])."""
    cos_n = np.zeros(len(qi), np.float32)
    cos_a = np.zeros(len(qi), np.float32)
    for s in range(0, len(qi), chunk):
        P = Qm[qi[s:s + chunk]].multiply(Tm[ti[s:s + chunk]]).tocsr()
        rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
        is_name = P.indices < HALF
        cos_n[s:s + chunk] = np.bincount(rows[is_name], weights=P.data[is_name], minlength=P.shape[0])
        cos_a[s:s + chunk] = np.bincount(rows[~is_name], weights=P.data[~is_name], minlength=P.shape[0])
    return cos_n, cos_a


def _topn_local(Qm, Tm, q_sel, t_sel, k, threads):
    """Top-k of Qm[q_sel] x Tm[t_sel]^T; returns local (q, t) indices into the full matrices."""
    C = sp_matmul_topn(Qm[q_sel], Tm[t_sel].T.tocsr(), top_n=k, threshold=0.05, sort=True,
                       n_threads=threads).tocoo()
    return q_sel[C.row], t_sel[C.col], C.data


def generate_candidates(Q: pl.DataFrame, T: pl.DataFrame, fq: pl.DataFrame, ft: pl.DataFrame,
                        k: int = 30, k_noaddr: int = 5, max_df_frac: float = 0.01, min_df_cap: int = 1000,
                        w_name: float = 1.0, w_addr: float = 1.0, region_spec: dict | None = None,
                        threads: int | None = None) -> pl.DataFrame:
    """Top-``k`` targets per (S1 record, target source), within each country.

    With ``region_spec`` (from ``learn_region_merges``) the search is split by
    region group: an S1 record is compared with targets of its own group(s)
    plus all targets that have no region token, and an S1 record with no
    region token is compared with everything. IDF is computed over the whole
    (country, source), so scores stay comparable across groups.

    A second, name-only pass adds the top ``k_noaddr`` targets that have no
    address: in the main pass they can only score on the name, so same-name
    records with an address would crowd them out.

    Returns (q_rid, t_rid, src, bscore, noaddr_pass).
    """
    threads = threads or os.cpu_count()
    gq_all, gt_all = _groups(Q, T, region_spec)
    out = []
    for country in Q["country"].unique().sort().to_list():
        q_ids = Q.filter(pl.col("country") == country)["rid"].to_numpy()
        fqc = fq.join(pl.DataFrame({"rid": q_ids}), on="rid", how="semi")
        q_local, q_cols = np.searchsorted(q_ids, fqc["rid"].to_numpy()), fqc["fid"].to_numpy()
        gq = gq_all.filter(pl.col("country") == country)
        for src in (2, 3):
            Tc = T.filter((pl.col("country") == country) & (pl.col("src") == src))
            if Tc.height == 0 or len(q_ids) == 0:
                continue
            t_ids = Tc["rid"].to_numpy()
            ftc = ft.join(Tc.select("rid"), on="rid", how="semi")
            t_local, t_cols = np.searchsorted(t_ids, ftc["rid"].to_numpy()), ftc["fid"].to_numpy()
            df = np.bincount(t_cols, minlength=1 << NBITS)
            cap = max(min_df_cap, int(max_df_frac * len(t_ids)))
            idf = np.where((df > 0) & (df <= cap), np.log(len(t_ids) / np.maximum(df, 1)), 0.0)
            Tm = _matrix(t_local, t_cols, len(t_ids), idf)
            Qm = _matrix(q_local, q_cols, len(q_ids), idf, w_name, w_addr)

            res = []
            gt = gt_all.filter(pl.col("rid").is_in(t_ids))
            if gq.height == 0:  # no split for this country
                res.append(_topn_local(Qm, Tm, np.arange(len(q_ids)), np.arange(len(t_ids)), k, threads))
            else:
                t_wild = np.setdiff1d(np.arange(len(t_ids)), np.searchsorted(t_ids, gt["rid"].unique().to_numpy()))
                for g, rows in gq.group_by("g"):
                    qs = np.searchsorted(q_ids, rows["rid"].unique().sort().to_numpy())
                    tg = np.searchsorted(t_ids, gt.filter(pl.col("g") == g[0])["rid"].unique().to_numpy())
                    res.append(_topn_local(Qm, Tm, qs, np.union1d(tg, t_wild), k, threads))
                q_wild = np.setdiff1d(np.arange(len(q_ids)), np.searchsorted(q_ids, gq["rid"].unique().to_numpy()))
                if len(q_wild):
                    res.append(_topn_local(Qm, Tm, q_wild, np.arange(len(t_ids)), k, threads))
            main = pl.DataFrame({"q": np.concatenate([r[0] for r in res]), "t": np.concatenate([r[1] for r in res]),
                                 "bscore": np.concatenate([r[2] for r in res]).astype(np.float32)})
            # an S1 record in several groups: merge its lists, keep the best k
            main = (main.unique(["q", "t"]).sort("bscore", descending=True)
                    .with_columns(pl.int_range(pl.len()).over("q").alias("r")).filter(pl.col("r") < k).drop("r")
                    .with_columns(pl.lit(False).alias("noaddr_pass")))
            parts = [main]
            no_addr = np.flatnonzero(~Tc["has_addr"].to_numpy())
            if len(no_addr) and k_noaddr:
                Qn = _matrix(q_local, q_cols, len(q_ids), idf, w_name, 0.0)
                qn, tn, sn = _topn_local(Qn, Tm, np.arange(len(q_ids)), no_addr, k_noaddr, threads)
                parts.append(pl.DataFrame({"q": qn, "t": tn, "bscore": sn.astype(np.float32),
                                           "noaddr_pass": np.full(len(qn), True)}))
            c = pl.concat(parts).sort("noaddr_pass").unique(["q", "t"], keep="first", maintain_order=True)
            cos_n, cos_a = _block_cosines(Qm, Tm, c["q"].to_numpy(), c["t"].to_numpy())
            out.append(pl.DataFrame({"q_rid": q_ids[c["q"].to_numpy()].astype(np.uint32),
                                     "t_rid": t_ids[c["t"].to_numpy()].astype(np.uint32),
                                     "src": np.full(c.height, src, np.uint8),
                                     "bscore": c["bscore"], "noaddr_pass": c["noaddr_pass"],
                                     "cos_name": cos_n, "cos_addr": cos_a}))
    cand = pl.concat(out)
    # a target found by both passes keeps its main-pass score
    return (cand.sort("noaddr_pass").unique(["q_rid", "t_rid"], keep="first", maintain_order=True)
            .sort("q_rid", "src", "bscore", descending=[False, False, True]))


def recall_at_k(cand: pl.DataFrame, truth_pairs: pl.DataFrame, ks=(5, 10, 20, 30, 40)) -> dict:
    """Pair recall if only the top k main-pass candidates per (S1, source) were kept.

    Name-only-pass candidates always count. One blocking run at the largest k thus shows
    what every smaller cutoff would have found.
    """
    main = (cand.filter(~pl.col("noaddr_pass"))
            .select("q_rid", "t_rid", pl.col("bscore").rank("ordinal", descending=True).over("q_rid", "src").alias("r")))
    extra = cand.filter(pl.col("noaddr_pass")).select("q_rid", "t_rid", pl.lit(0, main["r"].dtype).alias("r"))
    r = truth_pairs.select("q_rid", "t_rid").join(pl.concat([main, extra]), on=["q_rid", "t_rid"], how="left")["r"]
    n = max(truth_pairs.height, 1)
    return {int(k): float((r <= k).sum() / n) for k in ks if k <= int(main["r"].max() or 0)}


def noaddr_recall_at_k(cand: pl.DataFrame, truth_pairs: pl.DataFrame, ks=(5, 10, 15, 20)) -> dict:
    """Pair recall if the name-only pass kept only its top k per (S1, source); the main pass counts whole."""
    extra = (cand.filter(pl.col("noaddr_pass"))
             .select("q_rid", "t_rid", pl.col("bscore").rank("ordinal", descending=True).over("q_rid", "src").alias("r")))
    main = cand.filter(~pl.col("noaddr_pass")).select("q_rid", "t_rid", pl.lit(0, extra["r"].dtype).alias("r"))
    r = truth_pairs.select("q_rid", "t_rid").join(pl.concat([main, extra]), on=["q_rid", "t_rid"], how="left")["r"]
    n = max(truth_pairs.height, 1)
    return {int(k): float((r <= k).sum() / n) for k in ks if k <= int(extra["r"].max() or 0)}


def candidates_at_k(cand: pl.DataFrame, n_s1: int, ks=(5, 10, 15, 20, 30, 40), noaddr: bool = False) -> dict:
    """Candidates per S1 record if one pass kept only its top k (the other pass kept whole): the cost side
    of ``recall_at_k`` (main pass) or ``noaddr_recall_at_k`` (``noaddr=True``)."""
    this = cand.filter(pl.col("noaddr_pass") == noaddr)
    r = this.select(pl.col("bscore").rank("ordinal", descending=True).over("q_rid", "src"))["bscore"]
    other = cand.height - this.height
    return {int(k): float(((r <= k).sum() + other) / max(n_s1, 1)) for k in ks if k <= int(r.max() or 0)}


def recall_report(cand: pl.DataFrame, truth_pairs: pl.DataFrame) -> dict:
    """Share of true (q_rid, t_rid) links present in ``cand``, overall and per source."""
    hit = truth_pairs.join(cand.select("q_rid", "t_rid"), on=["q_rid", "t_rid"], how="semi")
    per_q = truth_pairs.group_by("q_rid").len().join(hit.group_by("q_rid").len(), on="q_rid", how="left", suffix="_hit")
    return {
        "pair_recall": hit.height / max(truth_pairs.height, 1),
        "s1_all_found": float((per_q["len_hit"].fill_null(0) == per_q["len"]).mean()),
        "cand_per_s1": cand.height / max(cand["q_rid"].n_unique(), 1),
    }
