---
tags: [machine-learning, segmentation, ATTO, experiments, smp]
project: cc-machine-learning
dataset: ATTO
model: mit_b5 (SegFormer encoder) + Unet decoder
status: tuning complete → cross-validation next
---

# ATTO Bark-Microhabitat Segmentation — Experiment Series (A–I)

> [!summary] TL;DR
> Structured one-lever-at-a-time tuning on **ATTO** (8 bark-microhabitat classes, `mit_b5`, 1024² tiles, effective batch 8, single fold `cv0`). Driven by a diagnosed **cyanosbark failure** (IoU started at 0.07). After 9 runs the chosen config for cross-validation is the **"E / B" recipe**: `dice_ce`, `class_weight_power 0.9`, `bg_weight_multiplier 0.1`, with AMP + batch 4 + grad-checkpointing for speed (EMA optional). The fancy levers (label smoothing, aggressive AdamW) caused **degenerate background collapse** and were rejected.

---

## 1. Timeline (rough)

1. **Early phase — weighting study.** Runs **A → B → C**, varying class weighting and loss. Established **B** (`power 0.9`, `bg_mult 0.1`) as the principled baseline; it fixed cyanosbark's background confusion.
2. **Hard-sampling probe.** Run **D** (oversample cyanosbark @ strength 2.0) — regressed.
3. **Major code refactor + bug fixes** (the important inflection point — see §3). Among other things, the **confusion-matrix / prediction code was fixed**, which splits the runs into two non-comparable measurement regimes. The boundary falls **between C and D**.
4. **Infrastructure + recipe phase.** Run **E** (bf16 AMP + real batch 4 + grad-checkpoint — a speed win), then **F** (throw the whole modern recipe at it at once) — **degenerate**.
5. **Isolated tuning set G/H/I** — one lever each over E, to try to genuinely beat B before locking the CV config.
6. **Decision** — adopt the E/B recipe for cross-validation. (**I** still running at decision time; not blocking.)

---

## 2. The metric, and how to read it

- **Primary metric: `iou_corpus_fg`** — mean foreground IoU, computed from the **pooled (corpus) confusion matrix** over the whole validation set, with **background excluded**. This is the **model-selection / early-stopping signal**.
- Per-class IoU = `CM[c,c] / (row_sum[c] + col_sum[c] − CM[c,c])`.

> [!warning] Two measurement regimes — **A/B/C are NOT comparable to D–I**
> A confusion-matrix bug (see §3) **inflated foreground IoU by ~0.05** in the early runs. The fix's commit date is misleading (code was committed late, deployed earlier), so **trust the data, not the commit timestamp**. The clean tell is **offline-CM vs in-loop agreement**: buggy code makes the offline number exceed the in-loop number; fixed code makes them equal.
>
> | Run | offline-CM | in-loop | gap | regime |
> |---|---|---|---|---|
> | A | 0.523 | 0.493 | +0.03 | **old (inflated)** |
> | B | 0.518 | 0.478 | +0.04 | **old (inflated)** |
> | C | 0.520 | 0.470 | +0.05 | **old (inflated)** |
> | D | 0.450 | 0.450 | ~0.00 | **fixed** |
> | E | 0.463 | 0.463 | ~0.00 | **fixed** |
>
> ⇒ **Group 1 (old, inflated) = A, B, C** · **Group 2 (fixed, trustworthy) = D, E, F, G, H, I.** Only compare *within* a group.

> [!tip] Degeneracy detector
> **`all-class-mean IoU < foreground-mean IoU` ⟺ background has collapsed.** In healthy runs background (~0.95) pulls the all-class mean *above* the fg mean; when background collapses the inequality flips. `iou_corpus_fg` excludes background, so it does **not** catch this on its own — always check background IoU / `iou_score`.
>
> ⚠️ **Caveat:** this rule holds only because **background is normally the highest-IoU class here** (~0.95 ≫ fg-mean ~0.46), so its collapse alone flips the inequality. On a dataset where background is *not* the top class, the rule doesn't transfer — fall back to checking background IoU directly.

---

## 3. Code changes that matter

| Change | What it did | Effect on results |
|---|---|---|
| **Fix double-softmax activation** | Model returns **logits**; softmax applied explicitly only where probabilities are needed. | Correctness; ended a double-softmax bug. |
| **Weighted Dice+CE + class-weight calc fix** | Class weights use **1/√frequency** (`class_weight_power` knob), not raw 1/freq. | Enabled the A→B weighting study. |
| **Stratified train/val split** | Iterative stratification on per-image class presence. | Cleaner, balanced validation. |
| **★ CM / prediction fix (the C\|D boundary)** | (1) prediction `softmax→.round()` (threshold-0.5, dumped low-confidence pixels into background) **→ argmax**; (2) `confusion_matrix(labels=true_labels)` (dropped FPs into absent classes, inflated precision) **→ `labels=all_labels`**. | **Removed the ~0.05 inflation.** Splits runs into Group 1 vs Group 2. |
| **Training-recipe flags (opt-in)** | `--optimizer adamw`, `--lr_schedule cosine` + warmup, `--amp` (bf16), `--grad_checkpoint`, `--label_smoothing`, `--ema_decay`, photometric aug. **Defaults reproduce legacy behaviour bit-for-bit.** | Enabled runs E–I. |
| **Gradient-checkpointing fix** | The old MiT checkpointing was a silent **no-op** (patched a forward that smp never calls). Now wraps each block's real forward. | **Required** for batch 4 @1024² — `--amp` alone OOMs a 40 GB A100 (mit_b5 attention softmax stays fp32 under autocast). |
| **Train/val leak fix** | `set_non_unique_paths` made a permutation (was sampling with replacement → leakage). | Removed a train/val leak. |
| **Per-run code snapshot** | Each run now copies code + config into `exp_dir/code_snapshot/` and records aug params in `run_config.json`. | Reproducibility (present from E onward). |

---

## 4. Experiment ledger

Foreground-mean IoU at each run's best epoch. **⚠ compare only within a group** (see §2). **The `bg IoU` column is the degeneracy gate — a high fg-mean with collapsed background (F, H) is *not* a good run; read fg-mean and bg IoU together.**

| Run | Recipe (one lever vs. predecessor) | fg-mean IoU | bg IoU | Group | Verdict |
|---|---|---|---|---|---|
| **A** | `dice_ce`, `power 0.75`, `bg_mult 0.25` | 0.523 | 0.94 | 1 | baseline *(inflated)* |
| **B** | `power 0.9`, `bg_mult 0.1` | 0.518 | 0.94 | 1 | **principled best baseline** *(inflated)* |
| **C** | `tversky_ce` α0.3 β0.7 | 0.520 | 0.96 | 1 | recall↑ precision↓, ≈flat *(inflated)* |
| **D** | B + hard-sample cyanosbark @ 2.0 | 0.450 | 0.95 | 2 | hard sampling **regresses** |
| **E** | B + `--amp` + batch 4 (eff. 8) + `--grad_checkpoint` | **0.463** | 0.96 | 2 | **best healthy; fast; = B recipe under fixed metric** |
| **F** | E + AdamW/cosine/EMA/label-smooth (5 levers, `decoder_lr_mult 10`, lr 6e-5) | 0.488 | **0.002** | 2 | ❌ **degenerate** |
| **G** | E + `--ema_decay 0.999` | 0.464 | 0.95 | 2 | healthy but flat (+0.001) |
| **H** | E + `--label_smoothing 0.05` | 0.464 | **0.026** | 2 | ❌ **degenerate** |
| **I** | E + de-risked AdamW (`decoder_lr_mult 3`, cosine, no EMA/smoothing) | **0.485** | 0.96 | 2 | ✅ **healthy & best in G2** (+0.022 vs E); broad mid-class gain, but **barkdominated −0.10 → lichen** |
| **J** | I + gentle hard-sample barkdominated @ 0.5 | **0.490** | 0.97 | 2 | ❌ **rescue FAILED** — barkdominated fell *further* to 0.256 (recall 0.32); side-effect: cyanosbark 0.09→**0.16** (only run to move it) |

---

## 5. Per-run notes

- **A — baseline.** `power 0.75 / bg 0.25`.
- **B — the baseline to beat.** Raising `power 0.75→0.9` and dropping `bg_mult 0.25→0.1` cut cyanosbark's **background** confusion (≈30%→18%) and lifted **barkdominated**, but caused **rare-class cannibalization** (cyanosliverwort eaten by liverwort). Net ≈ flat vs A — see the callout below on why B is the baseline despite A scoring marginally higher.
- **C — Tversky.** β>α lifted recall across the board at the cost of precision — a roughly fair trade, net flat.
- **D — hard sampling @2.0.** Over-predicted cyanosbark everywhere; precision collapsed, cyanosmoss cratered as collateral. **Lesson: 2.0 is too aggressive — only revisit hard sampling at strength ~0.5.**
- **E — AMP infrastructure win.** bf16 + physical batch 2→4 (effective stays 8) + grad-checkpoint. Much faster; background healthy; cyanosbark still on its ceiling. Same loss/weighting as B → **E is "B under the fixed metric."**
- **F — degenerate (cautionary).** Five levers at once. The (EMA) model predicted **cyanosbark for ~98% of background** (bg recall 0.97→~0.00). It *scored* 0.488 only because `iou_corpus_fg` excludes background and it sacrificed the single hardest class (cyanosbark → 0.0016) to lift the other six. **Five levers at once = unattributable → motivated G/H/I.**
- **G — EMA isolated.** Healthy, but **metric-neutral** (+0.001 over E). Worth keeping only for a more stable final checkpoint.
- **H — label smoothing isolated.** **Refuted the hypothesis** that smoothing was only unsafe alongside `decoder_lr_mult 10`: at `bg_mult 0.1` it is **alone sufficient to collapse background** (bg IoU 0.026, never recovered). **Reject.**
- **I — de-risked AdamW.** The transformer recipe with the poison knobs neutralised (`decoder_lr_mult 10→3`, no smoothing, no EMA). **fg-mean 0.485 (best ep17), bg 0.96 — healthy and the best Group-2 run, +0.022 over E.** offline-CM 0.4853 == in-loop 0.4854 → trustworthy. The lift is broad and *real* across the well-represented mid classes (cyanosliverwort +0.07, cyanosmoss +0.06, liverwort/moss +0.05) — AdamW+cosine sharpened them. **But barkdominated regressed −0.10 (0.42→0.32), driven by a recall collapse 0.73→0.53: its pixels are absorbed by lichen** (lichen recall rose 0.82→0.88). cyanosbark unchanged at its 0.09 ceiling (just traded recall for precision). So I's gain is genuine optimization, *not* a degenerate trade — but it spends barkdominated to buy the mid classes.
- **J — barkdominated rescue, FAILED.** I's recipe + `--hard_sample_target barkdominated --hard_sample_strength 0.5`, meant to recover barkdominated's recall by oversampling tiles where it's missed. It **made barkdominated worse** (0.318→0.256, recall 0.53→0.32): the oversampled tiles are the ambiguous lichen/cyanosbark bark-substrate boundaries, and training harder on them pushed the model to assign them to the dominant neighbors, not barkdominated. fg-mean 0.490 (highest of all, healthy bg 0.97) but only because the leaked signal lifted lichen and — notably — **cyanosbark broke its 0.09 ceiling to 0.16**, the only run in the series to move it. Decisive negative result: barkdominated's loss is a boundary/labeling problem, unfixable by sampling. **Rejected.**

> [!question] Why is B the "baseline to beat" when A actually scores marginally higher?
> Short answer: **B was chosen for principled/diagnostic reasons, not because it won on IoU — on the consistent recompute A and B are a statistical tie.**
> - **The fg-mean gap is noise.** A 0.523 vs B 0.518 = −0.005, single fold / single seed. Treat **A ≈ B**.
> - **A→B is a redistribution, not a lift.** Versus A, B trades **cyanosliverwort −0.09** and **cyanosbark −0.04** for **barkdominated +0.06** and **cyanosmoss +0.03** → nets ~flat (matches the −0.005).
> - **Why B was preferred anyway:** the `0.9 / 0.1` weighting directly targeted the project's diagnosed failure — it cut cyanosbark's *bleed into background* (confusion ≈30%→18%) and structurally fixed barkdominated, a correction judged more generalizable than A's heavier background weighting. B's config then became the **reference point** for every later single-lever experiment.
> - ⚠️ **Measurement caveat (important):** the original A→B write-up recorded different deltas (`barkdominated +0.156`, `cyanosbark 0.07→0.13`, `net +0.025`). Those came from the **old buggy-CM / in-loop** readings and **do not reproduce** under the consistent corpus best-epoch recompute in §6 — which even shows cyanosbark *higher* in A (0.17 vs 0.13). Both A and B are Group 1 (inflated), so don't over-read either set; the safe, consistent statement is **A ≈ B, and "B is the baseline" is a choice, not an empirical win.**

---

## 6. Per-class foreground IoU (best epoch, healthy runs)

> Group 1 (A/B/C) vs Group 2 (D/E/G) are **not** cross-comparable; shown together only for shape.

| Class | A | B | C | D | E | G | **I** | **J** |
|---|---|---|---|---|---|---|---|---|
| liverwort | 0.66 | 0.65 | 0.65 | 0.59 | 0.60 | 0.62 | **0.65** | 0.65 |
| moss | 0.60 | 0.61 | 0.63 | 0.56 | 0.58 | 0.56 | **0.63** | 0.66 |
| cyanosliverwort | 0.56 | 0.47 | 0.53 | 0.46 | 0.46 | 0.47 | **0.53** | 0.53 |
| cyanosmoss | 0.39 | 0.42 | 0.42 | 0.28 | 0.34 | 0.32 | **0.40** | 0.38 |
| lichen | 0.79 | 0.80 | 0.79 | 0.75 | 0.76 | 0.77 | **0.77** | 0.79 |
| barkdominated | 0.49 | 0.55 | 0.48 | 0.43 | 0.42 | 0.43 | **0.32** | 0.26 |
| **cyanosbark** | **0.17** | **0.13** | **0.15** | **0.09** | **0.09** | **0.09** | **0.09** | **0.16** |

> **I vs E (both Group 2), recall/precision shift:** cyanosliverwort recall 0.63→0.75 · cyanosmoss precision 0.45→0.54 · moss precision 0.70→0.82 — the mid-class gains. **barkdominated recall 0.73→0.53 (lost to lichen)** — the cost. cyanosbark recall 0.35→0.17 / precision 0.11→0.17 (ceiling holds).
>
> **J (I + hard-sample barkdominated @0.5):** the rescue **backfired** — oversampling barkdominated's *boundary* tiles trained the model to resolve them toward lichen/cyanosbark, so barkdominated recall fell *further* 0.53→**0.32** (IoU 0.318→0.256). The redirected signal lifted lichen (0.77→0.79) and uniquely knocked **cyanosbark off its ceiling 0.09→0.16** (recall 0.22, precision 0.36) — the only lever in A–J to move it. Lesson: barkdominated's loss under AdamW is a **boundary/labeling problem, not a sampling-starvation problem** — more examples make it worse.

Degenerate runs for contrast: **F** cyanosbark 0.0016 / bg 0.002; **H** cyanosbark 0.0024 / bg 0.026.

---

## 7. Key findings & lessons

- **cyanosbark is the hardest class** and the recurring bottleneck; **cyanosbark↔cyanosmoss confusion sits at ~33–34% regardless of lever** — a **labeling / visual-similarity ceiling**, not an imbalance problem. Audit via the `tile_confusions_*.csv` diagnostics.
- **Class weighting** is the highest-leverage knob (A→B). Pushing it too far cannibalises rare classes.
- **Hard sampling @2.0 hurts** (D); revisit only at ~0.5.
- **AMP does not measurably reduce accuracy.** The earlier scare ("B 0.52 ≫ E 0.46, AMP costs 0.05 IoU") was a **measurement artifact** — B is Group 1 (inflated), E is Group 2. There's *no evidence AMP harms*; bf16's theoretical cost is <0.005 IoU. (Not positively isolated — the only non-AMP Group-2 run, D, is confounded by hard sampling — but safe to use.)
- **Beware metric-rewarded degeneracy.** Because `iou_corpus_fg` excludes background, a model can boost it by collapsing background into a hard class (F, H). **Always check background IoU.** Detector: all-class-mean < fg-mean.
- **Change one lever at a time.** F's five-at-once stack was uninterpretable; the G/H/I bisect immediately localised the culprit (label smoothing).

---

## 8. Decision — cross-validation config

> [!success] LOCKED config (2026-06-17) — script `cv_mixed.sh` (SLURM `--array=0-4`)
> ```
> --loss dice_ce --class_weight_power 0.9 --bg_weight_multiplier 0.1
> --batch_size 4 --accumulation_steps 2 --amp --grad_checkpoint
> --epochs 70 --early_stopping_patience 0
> ```
> Legacy optimizer (adam) + plateau schedule. **Early stopping DISABLED** so all 5 folds train a fixed 70-epoch budget and stay directly comparable (`best_model.pth` still saves the peak-`iou_corpus_fg` epoch). 70 epochs: E plateaus ~ep30 and oscillates to ep49 (peak ep42), so 70 won't lift the peak — it buys equal-length, comparable learning curves, not accuracy.
> **Rejected:** the AdamW path. I (de-risked adamw) lifted fg-mean to 0.485 but cost barkdominated 0.42→0.318; J (I + hard-sample barkdominated) hit 0.490 but the rescue backfired (barkdominated→0.256). **barkdominated is a priority class → both rejected.** Also avoid `--label_smoothing` (H collapsed bg) and aggressive `--optimizer adamw --decoder_lr_mult 10` (F collapsed bg). EMA dropped (G's gain was +0.001, not worth the knob).
> **Recalculation of A–C under the fixed metric was deliberately skipped** — not decision-relevant, since the recipe choice is settled within the comparable Group 2 and E already embodies B's recipe under the corrected metric.

---

## 9. Data & provenance

- **All runs:** `Y:\CryptXChange\03_Processed_data\02_ML\ATTO\v17\04_final_experiments\{A..I}` — each holds the `exp_*` dir (`best_model.pth`, train/valid logs, per-epoch confusion matrices `confusion_matrix[_raw]_ep*.csv`, `results.pdf`), some with `notes.txt`.
- **exp hashes:** A `4fcffb` · B `f26a01` · C `5f9e74` · D `65a0f7` · E `7c0309` · F `1b26f8` · G `899958` · H `441760` · I `50dbd9` · J `0d6660`.
- Numbers in §4/§6 were recomputed from each run's **pooled corpus raw confusion matrix** at its best (max fg-mean) epoch.
- **Common setup:** `mit_b5` encoder + Unet decoder, 1024² tiles, effective batch 8, single fold `cv0`, config `server_config_atto_mixed.yml`, early-stopping patience 30, ~50 epochs.

## 10. Open items

- [x] **I finished: 0.485, healthy, +0.022 over E — but barkdominated −0.10 (→ lichen).** Reopened the CV-config choice; ran J to try to rescue barkdominated.
- [x] **J finished: rescue failed** (barkdominated 0.256). **Decision locked → E/B recipe, `cv_mixed.sh`, 70 epochs, early stopping off.** Series A–J closed.
- [ ] **cyanosbark lead for future work:** J is the *only* run to move cyanosbark off its 0.09 ceiling (→0.16), via oversampling bark-substrate tiles (indirect: target=barkdominated @0.5, adamw). The untested variant is **direct cyanosbark hard-sampling at gentle strength 0.5** — note **D already tried direct cyanosbark but at strength 2.0** (too aggressive: over-predicted everywhere, precision cratered, cyanosmoss collateral, net regression). So D ≠ this lead; they differ on target, strength (2.0 vs 0.5) *and* optimizer (adam vs adamw). Worth isolating post-CV (direct cyanosbark @0.5, ± adamw), or relabel the bark/cyanosbark boundary.
- [ ] (Optional, low priority) one non-AMP run of the plain B recipe under current code would turn "no evidence AMP harms" into a positive proof.
