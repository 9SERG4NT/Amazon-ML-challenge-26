"""Country-agnostic normalisation of business names and addresses.

Only generic text rules live here (transliteration, punctuation, common
abbreviations, US state codes, joined multi-word Indian state names, French
street types). Everything data-specific — e.g. which Devanagari spelling means
which English word — is learned from the training pairs in ``aliases.py``.
No external data is used.
"""
from __future__ import annotations

import re

import polars as pl
from anyascii import anyascii

# ---------------------------------------------------------------- vocabularies
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
}
# Multi-word place names: US states become their code, Indian states become one
# token (the per-country alias learner then maps codes like "tn" onto them).
MULTI_WORD = {
    "district of columbia": "dc", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "west virginia": "wv",
    "andhra pradesh": "andhrapradesh", "arunachal pradesh": "arunachalpradesh",
    "himachal pradesh": "himachalpradesh", "madhya pradesh": "madhyapradesh",
    "uttar pradesh": "uttarpradesh", "tamil nadu": "tamilnadu", "west bengal": "westbengal",
    "jammu and kashmir": "jammukashmir", "jammu kashmir": "jammukashmir",
}
ADDR_CANON = {
    # street types / units (English)
    "street": "st", "str": "st", "avenue": "ave", "av": "ave", "road": "rd", "drive": "dr",
    "drv": "dr", "lane": "ln", "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "boul": "blvd",
    "parkway": "pkwy", "pky": "pkwy", "circle": "cir", "court": "ct", "crt": "ct",
    "place": "pl", "terrace": "ter", "terr": "ter", "highway": "hwy", "trail": "trl",
    "square": "sq", "suite": "ste", "apartment": "apt", "floor": "fl", "flr": "fl",
    "number": "no", "num": "no", "nbr": "no", "building": "bldg", "bldng": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "route": "rte", "rt": "rte", "mount": "mt",
    "point": "pt", "heights": "hts", "junction": "jct", "center": "ctr", "centre": "ctr",
    "expressway": "expy", "freeway": "fwy", "crossing": "xing", "alley": "aly", "plaza": "plz",
    "opposite": "opp", "near": "nr", "sector": "sec", "saint": "st", "sainte": "ste",
    # French street types
    "rue": "rue", "r": "rue", "chemin": "chem", "ch": "chem", "impasse": "imp",
    "allee": "allee", "quai": "quai", "cours": "crs", "faubourg": "fbg", "residence": "res",
    # ordinal words -> numbers ("3rd" is reduced to "3" before tokenising)
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
    "twelfth": "12", "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
    "sixteenth": "16", "seventeenth": "17", "eighteenth": "18", "nineteenth": "19",
    "twentieth": "20",
    **US_STATES,
}
NAME_CANON = {
    "corporation": "corp", "incorporated": "inc", "company": "co", "limited": "ltd",
    "private": "pvt", "centre": "center", "ctr": "center", "svcs": "services",
    "service": "services", "svc": "services", "grp": "group", "intl": "international",
    "assoc": "associates", "bros": "brothers", "mgmt": "management", "natl": "national",
    "mfg": "manufacturing", "shree": "sri", "shri": "sri", "sree": "sri",
}
# Legal forms, honorifics and stop words: dropped from the "core" name.
LEGAL = frozenset(
    "llc inc corp co ltd pvt pllc llp lp plc pa pc sa sas sasu sarl eurl sci snc scop selarl "
    "public the and of mr mrs ms dr shri sri smt m s".split()
)
ALIAS_MARKERS = (
    r"\b(?:formerly known as|also known as|doing business as|trading as|formerly|"
    r"f k a|d b a|a k a|fka|dba|aka)\b"
)
WEB_RE = r"^(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|net|org|biz|info|co\.in|in|fr|us|co|io)$"
MISSING_ADDR_RE = r"\b(?:n/a|none|null|nil)\b"
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g",
                       "7": "t", "8": "b", "9": "g"})
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")


# ------------------------------------------------------------------- helpers
def translit(s: pl.Series) -> pl.Series:
    """Transliterate non-ASCII strings (Devanagari, accents) to ASCII."""
    s = s.fill_null("")
    mask = s.str.contains(r"[^\x00-\x7F]")
    if mask.any():
        idx = mask.arg_true()
        s = s.scatter(idx, s.gather(idx).map_elements(anyascii, return_dtype=pl.Utf8))
    return s


def _fix_name_token(t: str) -> str:
    # OCR-style digit-for-letter typos inside words: "c0rporation", "5ummit", "malna6ers".
    # Real alphanumerics ("3m", "24x7", "1st") have few letters or are ordinals.
    n_digits = sum(c.isdigit() for c in t)
    if 1 <= n_digits <= 2 and len(t) - n_digits >= 3 and not _ORDINAL.match(t):
        t = t.translate(_LEET)
    return NAME_CANON.get(t, t)


def _map_tokens(long: pl.DataFrame, fn, alias: pl.DataFrame | None) -> pl.DataFrame:
    """Apply ``fn`` to every distinct token, then an optional learned alias table.

    ``long`` has columns rid, pos, tok[, country]; ``alias`` has country, tok, to.
    """
    vocab = long.select(pl.col("tok").unique())
    if fn is not None:
        vocab = vocab.with_columns(pl.col("tok").map_elements(fn, return_dtype=pl.Utf8).alias("mapped"))
        long = long.join(vocab, on="tok", how="left").with_columns(pl.col("mapped").alias("tok")).drop("mapped")
    if alias is not None and alias.height:
        long = (long.join(alias, on=["country", "tok"], how="left")
                .with_columns(pl.coalesce("to", "tok").alias("tok")).drop("to"))
    return long


def _to_long(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return (df.select("rid", "country", pl.col(col).str.split(" ").alias("tok"))
            .explode("tok").filter(pl.col("tok") != "")
            .with_columns(pl.int_range(pl.len()).over("rid").alias("pos")))


def _join_long(long: pl.DataFrame, name: str) -> pl.DataFrame:
    return long.sort("rid", "pos").group_by("rid", maintain_order=True).agg(pl.col("tok").str.join(" ").alias(name))


# ---------------------------------------------------------------- main entry
def clean_names(df: pl.DataFrame) -> pl.DataFrame:
    """Character-level name cleaning (before token mapping)."""
    raw = translit(df["business_name"]).str.to_lowercase().str.strip_chars()
    out = pl.DataFrame({"rid": df["rid"], "raw": raw})
    out = out.with_columns(
        pl.col("raw").str.contains(WEB_RE).alias("is_web"),
        pl.col("raw").str.extract(WEB_RE, 1).alias("web_label"),
    ).with_columns(
        pl.when(pl.col("is_web")).then(pl.col("web_label").str.replace_all("-", " ")).otherwise(pl.col("raw")).alias("raw")
    )
    clean = (pl.col("raw").str.replace_all(r"\bid\s*[:#]?\s*\d+", " ")  # "(ID: 83039)" tags
             .str.replace_all("&", " and ").str.replace_all(r"[.']", "")
             .str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())
    out = out.with_columns(clean.alias("c"))
    out = out.with_columns(
        pl.col("c").str.contains(ALIAS_MARKERS).alias("has_alias"),
        pl.col("c").str.extract(r"^(.*?)\s*" + ALIAS_MARKERS, 1).str.strip_chars().alias("pre"),
        pl.col("c").str.extract(ALIAS_MARKERS + r"\s*(.*)$", 1).str.strip_chars().alias("post"),
        pl.col("c").str.replace_all(ALIAS_MARKERS, " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("c"),
    )
    return out.select("rid", "c", "pre", "post", "is_web", "has_alias")


def clean_addresses(df: pl.DataFrame) -> pl.DataFrame:
    raw = translit(df["business_address"]).str.to_lowercase()
    e = pl.col("a").str.replace_all(MISSING_ADDR_RE, " ").str.replace_all(r"p\.?\s?o\.?\s?box", " pobox ")
    e = e.str.replace_all(r"(\d+)(?:st|nd|rd|th)\b", "$1")  # "3rd" -> "3", keeps "rd" = road
    e = e.str.replace_all(r"[^a-z0-9]+", " ")
    e = e.str.replace_all(r"(\d)([a-z])", "$1 $2").str.replace_all(r"([a-z])(\d)", "$1 $2")
    out = pl.DataFrame({"rid": df["rid"], "a": raw}).with_columns(e.alias("a"))
    for phrase, code in MULTI_WORD.items():
        out = out.with_columns(pl.col("a").str.replace_all(rf"\b{phrase}\b", code))
    # strip leading zeros from numbers ("017" -> "17")
    out = out.with_columns(pl.col("a").str.replace_all(r"\b0+(\d)", "$1").str.replace_all(r"\s+", " ").str.strip_chars())
    return out.select("rid", "a")


def token_frames(df: pl.DataFrame, name_alias=None, addr_alias=None):
    """Return (name_long, addr_long) token frames after canonical + learned mapping.

    ``df`` needs columns rid, country, business_name, business_address.
    """
    names = clean_names(df).join(df.select("rid", "country"), on="rid")
    addrs = clean_addresses(df).join(df.select("rid", "country"), on="rid")
    name_long = _map_tokens(_to_long(names, "c"), _fix_name_token, name_alias)
    addr_long = _map_tokens(_to_long(addrs, "a"), lambda t: ADDR_CANON.get(t, t), addr_alias)
    return names, name_long, addr_long


def normalize(df: pl.DataFrame, name_alias=None, addr_alias=None) -> pl.DataFrame:
    """Full normalisation. Adds rid (row index) and string/list feature columns."""
    if "rid" not in df.columns:
        df = df.with_row_index("rid")
    names, name_long, addr_long = token_frames(df, name_alias, addr_alias)
    name_n = _join_long(name_long, "name_n")
    core = _join_long(name_long.filter(~pl.col("tok").is_in(list(LEGAL))), "name_core")
    addr_n = _join_long(addr_long, "addr_n")
    nums = (addr_long.filter(pl.col("tok").str.contains(r"^\d+$")).group_by("rid")
            .agg(pl.col("tok").unique().alias("nums")))
    # alias parts get the same token mapping as the main name
    alt = names.select("rid", "country", pl.col("post").fill_null("").alias("c"))
    alt_long = _map_tokens(_to_long(alt, "c"), _fix_name_token, name_alias)
    name_alt = _join_long(alt_long, "name_alt")
    pre = names.select("rid", "country", pl.col("pre").fill_null("").alias("c"))
    name_pre = _join_long(_map_tokens(_to_long(pre, "c"), _fix_name_token, name_alias), "name_pre")

    out = (df.join(names.select("rid", "is_web", "has_alias"), on="rid", how="left")
           .join(name_n, on="rid", how="left").join(core, on="rid", how="left")
           .join(addr_n, on="rid", how="left").join(nums, on="rid", how="left")
           .join(name_alt, on="rid", how="left").join(name_pre, on="rid", how="left"))
    out = out.with_columns(
        [pl.col(c).fill_null("") for c in ("name_n", "name_core", "addr_n", "name_alt", "name_pre")]
        + [pl.col("nums").fill_null([])]
    ).with_columns(
        pl.when(pl.col("name_core") == "").then(pl.col("name_n")).otherwise(pl.col("name_core")).alias("name_core"),
        pl.col("business_name").fill_null("").str.contains(r"[^\x00-\x7F]").alias("non_ascii"),
    ).with_columns(
        pl.col("name_core").str.replace_all(" ", "").alias("name_nosp"),
        (pl.col("addr_n") != "").alias("has_addr"),
    )
    return out.sort("rid")
