"""Named versions of every pipeline component — the IDs used in method_result.md.

A run is described by four component versions:
  NORM   text normalisation and learned aliases     -> stage prep
  BLK    candidate generation (blocking)            -> stage block
  FEAT   pairwise features                          -> stage features
  MATCH  models and the decision rule               -> stages train, predict
An end-to-end version (M-v1, M-v2, ...) is one combination of the four (PRESETS).

Every stage caches its output under the versions it depends on, e.g. block/NORM-v2__BLK-v4b@20,
so switching one version recomputes only what it changes and never reuses stale files.

Rule: never edit a registered version. Add a new one, and log it in method_result.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

STAGE_DEPS = {"prep": ("norm",), "block": ("norm", "block"), "features": ("norm", "block", "feat"),
              "train": ("norm", "block", "feat", "match"), "predict": ("norm", "block", "feat", "match")}


@dataclass(frozen=True)
class Norm:
    id: str
    note: str
    alias_min_count: int = 25       # a target token needs this many links with the same unaligned S1 token ...
    alias_min_ratio: float = 0.5    # ... in at least this share of the links that contain it
    alias_max_unaligned: int = 4    # links with more unaligned tokens are too noisy to learn from


@dataclass(frozen=True)
class Block:
    id: str
    note: str
    k: int = 20                     # candidates per S1 record and target source
    k_noaddr: int = 5               # extra name-only pass over targets without address (0 = off)
    char_grams: bool = True         # character 3-grams of the joined core name
    joined_name: bool = True        # the whole joined core name as one feature
    max_df_frac: float = 0.01       # ignore features found in more than this share of targets ...
    min_df_cap: int = 1000          # ... unless the cap would fall below this document frequency
    region_split: bool = True       # search within region groups (states) instead of the whole country
    region_min_share: float = 0.002     # region key: last token of at least this share of S1 addresses ...
    region_min_last_ratio: float = 0.6  # ... and the last token in at least this share of its occurrences
    region_min_cross: int = 10          # merge two regions when this many training links cross them
    keep_nonevals: float = 1.0      # test-like universe: keep this share of non-eval train S1 (with their true
                                    # targets); the rest leave with their targets, so distractors per S1 rise
    dup_distractors: int = 1        # train universe: every distractor (a target no train S1 links to) appears this
                                    # many times; the copies take top-k slots like the originals would
    prune_keep: float = 0.0         # learned candidate filter, a second blocking stage: a small LightGBM on the
                                    # blocking scores and their competition context (no string similarity) keeps
                                    # the candidates above the threshold that retains this share of the true links
                                    # the search found for fit entities (out-of-fold); 0 = keep the whole search
    search_from: str | None = None  # reuse the cached search output of this blocking version (it must have the
                                    # same search settings) instead of searching again; searches when not cached


@dataclass(frozen=True)
class Feat:
    id: str
    note: str
    align: bool = True              # soft word alignment of core names (8 features)
    number_gap: bool = False        # closest unmatched numbers: edit distance and numeric gap (3 features)
    distinct: bool = False          # similarity of the distinctive parts (each country's frequent tokens removed)
                                    # and how many targets share each core name (10 features)


@dataclass(frozen=True)
class Match:
    id: str
    note: str
    stages: int = 2                 # 1: one model; 2: + a model on stage-1 probability context
    folds: int = 3                  # cross-fitting folds over the fit entities
    support: bool = False           # stage 2 also sees similarity to the S1's confident candidates
    rules: tuple = ("threshold", "expected_f", "gated_ef")  # decision rules tried on the eval slice
    lgb: tuple = ()                 # (name, value) overrides of model.DEFAULT_PARAMS
    pred_margin: float | None = None  # stop summing trees once 2*|raw score| exceeds this (LightGBM pred_early_stop)
    fit_rest: bool = False          # also fit the "rest" entities (75% of train S1 instead of 30%; eval unchanged)
    frozen_from: str | None = None  # score with the fold models of run <NORM>__<frozen_from> instead of training;
                                    # the train split only, to measure an old model in a new universe
    algo: str = "lgb"               # "lgb" (LightGBM) or "xgb" (XGBoost hist; ``lgb`` then holds XGBoost params)
    blend_of: tuple = ()            # MATCH ids of runs on the same blocking and features: no training, their
                                    # stage-1 and stage-2 probabilities are averaged and the rule is chosen again
    distractor_weight: float = 1.0  # training weight of rows whose target is a distractor (no train S1 links to it)
    eval_dup: bool = False          # choose the rule on the doubled-distractor eval (links to distractors count
                                    # twice: the test's lookalike density) instead of the plain eval slice

    def lgb_params(self) -> dict:
        return dict(self.lgb)


def _index(*versions) -> dict:
    return {v.id: v for v in versions}


NORM = _index(
    Norm("NORM-v2", "anyascii transliteration; legal forms, street types, US states and ordinals canonicalised; "
                    "Indian state names joined; French street types; digit-for-letter fixes; fka/dba split; "
                    "website names turned into words; shree/shri/sri merged; per-country aliases learned "
                    "from training links"),
)

BLOCK = _index(
    Block("BLK-v1", "IDF-weighted sparse search on name and address tokens, 4-char prefixes, consonant skeletons "
                    "and house-number affixes; df cap 1%; top-30",
          k=30, k_noaddr=0, char_grams=False, joined_name=False, region_split=False),
    Block("BLK-v2", "BLK-v1 + joined-name token, name 3-grams, name-only pass (top-5) for targets without address",
          k=30, region_split=False),
    Block("BLK-v3", "BLK-v2 with a fixed df cap of 3,000 instead of 1% (rejected: recall -0.8 to -1.6 points)",
          k=30, region_split=False, max_df_frac=0.0, min_df_cap=3000),
    Block("BLK-v4", "BLK-v2 searched within regions: frequent last address tokens, merges learned from links",
          k=30, region_min_last_ratio=0.0),
    Block("BLK-v4b", "BLK-v4 with positional region keys (last token in at least 60% of its occurrences)", k=30),
    Block("BLK-v4b@20", "BLK-v4b with top-20 per source: the candidate set the matcher scores", k=20),
    Block("BLK-v4b@40", "BLK-v4b with top-40 per source: on full data top-20 finds only 98.3% of links "
                        "(99.45% on DEV-10), because regions hold 10x more rival records", k=40),
    Block("BLK-v4b@40n20", "BLK-v4b@40 with a top-20 name-only pass: measures both recall-vs-depth curves in one "
                           "run (targets without address: 88.7% recall at top-5 on full data)", k=40, k_noaddr=20),
    Block("BLK-v4b@20-tlu40", "BLK-v4b@20 on a test-like train universe: every eval S1 plus 40% of the other train "
                              "S1 entities; the others are removed with their true targets. The test has ~2.3 "
                              "distractors per S1 (train: 1.2) and half the US S1 density, and the leaderboard "
                              "(0.96) disagreed with the full-universe eval slice (0.9813)",
          keep_nonevals=0.4),
    Block("BLK-v4b@40-tlu40", "BLK-v4b@20-tlu40 with a top-40 main pass (name-only pass stays at 5): in the full "
                              "train universe a top-40 main pass with a top-5 name-only pass finds 98.82% of eval "
                              "links against 98.32% for BLK-v4b@20, at ~90 instead of 47.5 candidates per S1; a "
                              "deeper name-only pass adds only 0.11 points", k=40, keep_nonevals=0.4),
    Block("BLK-v4b@20-dup2", "BLK-v4b@20 on the full train universe with every distractor twice: the test has "
                             "twice train's distractors per S1 with the same mix. As many test distractors as "
                             "train ones share the name and street of a present S1 (India 9% / US 20%); in the "
                             "tlu40 universe only about half do, since dropping an S1 orphans its lookalikes. "
                             "Duplicating keeps the mix and doubles the lookalikes per S1 (US 2.3, India 2.6 "
                             "distractors per S1; test 2.3)",
          dup_distractors=2),
    Block("BLK-v5-tlu40", "BLK-v4b@20-tlu40 + a learned candidate filter: a small LightGBM on the blocking scores and "
                          "their competition context keeps 99.5% of the true links the search found (threshold set "
                          "out-of-fold on fit entities). The search's 48 candidates per S1 are mostly low-ranked "
                          "rivals, and the final ranking favours a smaller candidate set per S1",
          keep_nonevals=0.4, prune_keep=0.995, search_from="BLK-v4b@20-tlu40"),
)

FEAT = _index(
    Feat("FEAT-v1", "44 features: blocking score and cosines; rapidfuzz similarities of name, core name, joined "
                    "name and address; alias parts; token containment; house-number agreement; lengths and "
                    "flags; rank and competition context; S1 name frequency", align=False),
    Feat("FEAT-v2", "FEAT-v1 + 8 soft word-alignment features (52)"),
    Feat("FEAT-v3", "FEAT-v2 + 3 number-gap features (55)", number_gap=True),
    Feat("FEAT-v4", "FEAT-v3 + 10 distinctive-part features (65): name and address similarity after removing the "
                    "tokens found in over 1% of a country's records (cities, regions, street types, legal forms, "
                    "generic words), and how many targets share the S1's / target's core name. French test S1 are "
                    "often '<city> <generic word> <legal form>' and got ~3x the false links of US/India S1",
         number_gap=True, distinct=True),
)

MATCH = _index(
    Match("MATCH-v1", "one LightGBM (binary log loss), cross-fitted; exclusive assignment; threshold",
          stages=1, rules=("threshold",)),
    Match("MATCH-v2", "MATCH-v1 + stage 2 on stage-1 probability context, cross-fitted; rule chosen on the eval "
                      "slice among threshold, expected F and gated expected F"),
    Match("MATCH-v3", "MATCH-v2 + support features in stage 2", support=True),
    Match("MATCH-v4", "EXPERIMENTAL, not validated: MATCH-v2 with LightGBM prediction early stopping (margin 10) to "
                      "cut scoring time (full-data models grow ~1,400 trees). On tiny data it moved stage-1 "
                      "probabilities by up to 0.31 and stage-2 ones further (their inputs shift): not a free speedup",
          pred_margin=10.0),
    Match("MATCH-v5", "MATCH-v2 with learning rate 0.1 instead of 0.05: about half the trees (MATCH-v2 grew "
                      "~1,400 on full data), so training and scoring take about half the time",
          lgb=(("learning_rate", 0.1),)),
    Match("MATCH-v6", "MATCH-v5 fitted on the rest entities too: 75% of the train S1 entities (fit 30% + rest 45%) "
                      "instead of 30%, cross-fitted like the fit entities. Eval and early-stop entities are unchanged "
                      "(the split draws them independently of the fit share), so it reuses the cached features",
          lgb=(("learning_rate", 0.1),), fit_rest=True),
    Match("MATCH-v2-frozen", "the fold models of the FULL-v1 run (BLK-v4b@20, FEAT-v2, MATCH-v2), applied without "
                             "retraining to another universe's train split: how the leaderboard model scores in a "
                             "test-like universe. Picks its 52 columns by name, so FEAT-v3 parts (a superset) work",
          frozen_from="BLK-v4b@20__FEAT-v2__MATCH-v2"),
    Match("MATCH-v6-frozen-tlu", "the fold models of M-v6 (tlu40 universe, FEAT-v4, MATCH-v6) applied without retraining "
                                 "to another universe's train split: M-v6 on M-v7's doubled-distractor eval slice, to "
                                 "compare the two universes' models on one scale and with M-v6's leaderboard score",
          lgb=(("learning_rate", 0.1),), fit_rest=True, frozen_from="BLK-v4b@20-tlu40__FEAT-v4__MATCH-v6"),
    Match("MATCH-v8", "MATCH-v6 with XGBoost (Apache-2.0) instead of LightGBM: hist trees grown leaf-wise to 255 leaves "
                      "(the LightGBM setting), learning rate 0.1, same folds, early-stop slice, stage 2 and rules. A "
                      "second model family on the same features, for comparison and a later blend",
          lgb=(("eta", 0.1),), fit_rest=True, algo="xgb"),
    Match("MATCH-v9", "blend of MATCH-v6 (LightGBM) and MATCH-v8 (XGBoost) on the same blocking and features: the mean "
                      "of their stage-1 and stage-2 probabilities (both out-of-fold on fit rows, fold means elsewhere), "
                      "rule chosen on the eval slice", fit_rest=True, blend_of=("MATCH-v6", "MATCH-v8")),
    Match("MATCH-v10", "MATCH-v6 trained with distractor rows at weight 2 and the rule chosen on the doubled-distractor "
                       "eval: the test's twice-as-many lookalikes per S1, with no copied records (M-v7's copies let the "
                       "model spot exact twins, so it over-linked the test: 98.5% of S1 linked, 4.2 links each)",
          lgb=(("learning_rate", 0.1),), fit_rest=True, distractor_weight=2.0, eval_dup=True),
)

PRESETS = {
    "M-v1": ("NORM-v2", "BLK-v4b@20", "FEAT-v1", "MATCH-v1"),
    "M-v2": ("NORM-v2", "BLK-v4b@20", "FEAT-v2", "MATCH-v1"),
    "M-v3": ("NORM-v2", "BLK-v4b@20", "FEAT-v2", "MATCH-v2"),
    "M-v4": ("NORM-v2", "BLK-v4b@20", "FEAT-v3", "MATCH-v3"),
    # trained and validated in the test-like universe (the leaderboard disagreed with the full-train one)
    "M-v5": ("NORM-v2", "BLK-v4b@20-tlu40", "FEAT-v3", "MATCH-v6"),
    "M-v6": ("NORM-v2", "BLK-v4b@20-tlu40", "FEAT-v4", "MATCH-v6"),
    # the full train universe with its distractors doubled: as many hard lookalikes per S1 as the test
    "M-v7": ("NORM-v2", "BLK-v4b@20-dup2", "FEAT-v4", "MATCH-v6"),
    "M-v8": ("NORM-v2", "BLK-v4b@20-dup2", "FEAT-v4", "MATCH-v8"),
    "M-v9": ("NORM-v2", "BLK-v4b@20-dup2", "FEAT-v4", "MATCH-v9"),
    # M-v6's universe and features; distractors weighted twice in training, rule chosen on the doubled-distractor eval
    "M-v10": ("NORM-v2", "BLK-v4b@20-tlu40", "FEAT-v4", "MATCH-v10"),
    # M-v6 and M-v10 on the filtered candidates (a smaller candidate set per S1)
    "M-v11": ("NORM-v2", "BLK-v5-tlu40", "FEAT-v4", "MATCH-v6"),
    "M-v12": ("NORM-v2", "BLK-v5-tlu40", "FEAT-v4", "MATCH-v10"),
}
DEFAULT_PRESET = "M-v3"


@dataclass(frozen=True)
class RunVersions:
    norm: Norm
    block: Block
    feat: Feat
    match: Match
    preset: str | None = None

    def key(self, stage: str) -> str:
        """Cache directory name for ``stage``: the IDs of the versions it depends on."""
        return "__".join(getattr(self, part).id for part in STAGE_DEPS[stage])

    def describe(self) -> dict:
        return {"preset": self.preset, **{p: asdict(getattr(self, p)) for p in ("norm", "block", "feat", "match")}}


def resolve(preset: str | None = None, norm: str | None = None, block: str | None = None,
            feat: str | None = None, match: str | None = None) -> RunVersions:
    """A preset, optionally with some components swapped (the result then has no preset name)."""
    base = PRESETS[preset or DEFAULT_PRESET]
    ids = [c or b for c, b in zip((norm, block, feat, match), base)]
    swapped = any(c and c != b for c, b in zip((norm, block, feat, match), base))
    return RunVersions(NORM[ids[0]], BLOCK[ids[1]], FEAT[ids[2]], MATCH[ids[3]],
                       preset=None if swapped else (preset or DEFAULT_PRESET))
