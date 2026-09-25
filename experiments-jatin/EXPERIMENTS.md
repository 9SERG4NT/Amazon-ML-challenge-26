# Experiment Log — Business Entity Resolution

Jatin's experiment record. One entry per run; keep the summary table in sync.
Final `Documentation_template.md` will be written from this file.

**Metric:** F_0.5 = 1.25·P·R / (0.25·P + R), macro-averaged over S1 entities
(singletons included; empty-truth + empty-pred = 1.0). Test set: 1,732,544 S1 entities
(US 663,106 / India 809,986 / France 259,452).

---

## Summary table

| ID | Date | Method (short) | Block recall | Cand/S1 | OOF F0.5 | Val/LB F0.5 | Runtime | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| EXP-001 | 2026-09-25 | LGBM pairs, dev pool 2.4M | 0.1115 | 62.9 | 0.2463 | — | 47 min | PASS validator |
| EXP-002 | 2026-09-25 | Scale-safe rewrite, dev pool 2.4M | 0.4786* | ~60 | 0.1822 | — | 53 min | PASS validator |
| EXP-003 | 2026-09-25 | EXP-002 pipeline, **full 10.3M pool** | — | — | — | — | running | — |

\* EXP-002/003 report blocking recall **within the resolvable pool** (dev pools miss 77%
of true partners); EXP-001's 0.1115 is over all GT links. Comparable figure for EXP-002:
7672/69478 = 0.1104.

---

## EXP-001 — Baseline LightGBM pairwise pipeline (fixed)

- **Date:** 2026-09-25
- **Code:** `experiments-jatin/run_pipeline.py` (first fixed version)
- **Hypothesis:** the original script had crash-level bugs (`eval_X=`/`eval_y=` invalid in
  most LightGBM 4.x builds) and O(n) Python loops everywhere; fixing the API and
  vectorizing should make an end-to-end run feasible in under an hour.

**Methodology**
- Preprocessing: anyascii transliteration → lowercase → strip punctuation; DBA-prefix and
  legal-suffix stripping on names; address stopwords (near/opp/…); TOKEN_MAP
  abbreviation normalization (road→rd, …); house number = first digit run; postal = last
  5–6 digit run.
- Blocking: 12 key schemes per country (first-2-sorted-name, first-token, full-name,
  4/5-char prefix, 10 char 4-grams, house+street, 2-sorted-addr, name+house,
  name+street), buckets pruned at >300, capped at 80 candidates per S1.
- Features: 19 (6 rapidfuzz name/addr scores, 4 token Jaccard/overlap, exact flags,
  length ratios, is_s3, 2 TF-IDF char_wb(2,4) cosines).
- Model: LightGBM 4.7.0 binary, lr 0.05, 63 leaves, 2000 est. w/ early stop,
  GroupKFold(5) grouped by S1 entity, **device=gpu** (pip wheel CUDA; verified
  1M×100 trees: 2.6s GPU vs 5.0s CPU-8T).
- Decision: dual-threshold (p ≥ t_high) OR (p ≥ t_low AND p ≥ rel·pmax), grid 7×5×4
  tuned on OOF for macro F0.5.

**Fixes applied to the original script**
1. `eval_X`/`eval_y` → `eval_set=[(X_val, y_val)]` (the original would crash).
2. Vectorized features (rapidfuzz batch, set ops via numpy), threshold tuning
   (bincount F0.5 — bit-exact vs brute force), and pair building (1.3s vs ~30 min).
3. Vectorized `build_pairs_df` verified identical to the original loop.
4. Dev-mode caps S2/S3 rows so smoke tests fit in time/memory.

**Results** (dev mode: 20k S1 + first 1.2M+1.2M S2/S3 rows; full test inference)

| Metric | Value |
|---|---:|
| Load 12.5M rows | 154s |
| Blocking train / test | 46s / 140s |
| Train pairs | 1,257,033 (avg 62.9/S1) |
| **Blocking recall** | **0.1115** (7,746/69,478) |
| CV (GPU) | 34.4s, mean best_iter 203 |
| **OOF Macro F0.5** | **0.2463** |
| Rule | t_high=0.70, t_low=0.25, rel=0.70 |
| Test candidate pairs | 116,968,657 (~67.5/S1) |
| Predicted matches | 844,328 edges over 609,816 entities |
| Predicted singletons | 64.8% (true rate 5.6%) |
| Validator | **PASS** (incl. `--check-ids`) |
| Total runtime | 2,814s (47 min) |

**Analysis**
- The low F0.5 is a **pool artifact**: with only 23% of S2/S3 loaded, blocking can find
  at most 11% of true matches — even a perfect model scores ≈0.28 here. The model is
  near its restricted ceiling, but the run is over-conservative (64.8% predicted
  singletons vs 5.6% true) because most neighbors simply are not in the pool.
- Conclusion: pipeline mechanically sound; needs full-pool training for a meaningful
  score. Kept as baseline.

---

## EXP-002 — Scale-safe rewrite for the full 10.3M-row pool

- **Date:** 2026-09-25
- **Code:** `experiments-jatin/run_pipeline.py` (rewritten)
- **Hypothesis:** full-pool training was blocked by two scale walls — test TF-IDF hit
  ~40 GB RSS (char_wb (2,4) vocab over 12.5M texts) and giant string-pair columns
  (~7 GB per 117M-pair frame). Chunking + integer pairs removes both, letting us train
  on all 10.3M S2/S3 rows and lift blocking recall toward its true value.

**Changes vs EXP-001**
1. **TF-IDF**: fit on a 300k-row sample, bounded vocab (`max_features=2^20`), chunked
   transform (1M rows) + L2 normalize. Bounded memory at any corpus size.
2. **Pairs as int64 positions** end-to-end (no string pair columns); `explode → dedup →
   isin` label join.
3. **Chunked test scoring**: features + predict in 2M-pair blocks (VRAM-safe); top-3
   per-entity fallback guarantees ≥3 candidates survive thresholding.
4. **Negative subsampling** for CV fit (`--neg-keep 0.20`): ~1.9M fit rows instead of
   9.5M; thresholds still tuned on the full OOF.
5. Streamed output writing (no giant DataFrames).
6. Verified: `rule_mask` ≡ brute force, `f05_from_counts` ≡ per-entity loop
   (diff 0.0), output format invariants (1 row/entity, matches ⊆ candidates).

**Results** (dev mode: 20k S1 + first 2.4M S2/S3 rows; full test inference)

| Metric | Value |
|---|---:|
| Load + blocking | 160s + 50s |
| Blocking recall (resolvable pool / all GT) | 0.4786 / 0.1104 |
| TF-IDF (bounded, chunked) | 141s (vs 40 GB unbounded in EXP-001) |
| Features (1.26M pairs) | 26.4s, pos_rate 0.0061 |
| CV rows (neg_keep=0.2) | 257,369; 29.1s GPU; best_iter 157 |
| OOF Macro F0.5 | 0.1822 (rule: t_high=0.85, t_low=0.25, rel=0.9, topk=3) |
| Test pairs | 116,968,657 in 213s |
| Test TF-IDF + chunked scoring | 246s + 2,137s (59 chunks, ~30s each, RSS flat ~32 GB) |
| Predicted edges | 5,161,012 (~3.0/entity; true avg 3.46) |
| Predicted singletons | 0.40% (true 5.58%) |
| Validator | **PASS** (incl. `--check-ids`) |
| Total runtime | 3,196s (53 min) |

**Analysis**
- Scale fixes all held: TF-IDF memory bounded, chunked scoring flat-RAM, outputs written
  stream-style. Match-rate now in a realistic band (3.0 vs true 3.46), and the
  EXP-001 over-conservatism (64.8% predicted singletons) is gone (0.4%).
- OOF F0.5 (0.182) is **not comparable** to EXP-001 (0.246): (a) the top-3 floor now
  forces ≥3 predictions on every entity, and ~22% of dev-pool entities have zero
  resolvable positives (each a guaranteed 0); (b) different threshold grid point won.
  On the test set the same floor yields 3.0 edges/entity, so the OOF pessimism is a
  dev-pool artifact, not a deployment property.
- Made the top-K floor **tunable by the data** (tuner now evaluates topk∈{0,3}); the
  full-pool run will decide which wins at realistic partner availability.
- Next: EXP-003 = same pipeline, full 10.3M S2/S3 pool, neg-keep 0.2, with pre-feature
  pair subsampling to fit full-pool RAM.
