# Report: Reproducing and Stress-Testing Unlearning Failure under Quantization

We reproduce [*Catastrophic Failure of LLM Unlearning via Quantization*](https://arxiv.org/abs/2410.16454)
(Zhang et al., ICLR 2025) with NPO_KLR and GA_GDR, then test whether GPTQ and AWQ can break SURE, the authors'
proposed fix. The sections follow the four experiments listed in the [README](README.md#the-four-experiments).
- All reproduced values are our own measurements, from [`tables/tables.md`](tables/tables.md); table numbers below
  refer to it. Published values are always labelled *Reported* or "paper".
- Every discrepancy and code issue is analysed in [`FINDINGS.md`](FINDINGS.md) (cited as F§).
- Setup, versions and seeds are in [`Reproducibility.md`](Reproducibility.md).

## Summary

1. **Task 1 (NEWS) largely reproduces.** Quantization brings forgotten knowledge back, e.g. NPO_KLR M1 15.9 → 34.6 under RTN
   (paper 16.6 → 34.1).
2. **Task 2 (BOOKS) reproduces partially.** NPO_KLR's 4-bit row matches Table 3 on M1–M3 (M4 −5.6 under RTN, −1.0 under bnb-FP4); GA_GDR's does not. The unlearned
   full-precision rows do not match, and the paper's own Tables 1 and 3 contradict each other there (F§3.4).
3. **SURE as released never applies its saliency mask** and counts the forget gradient twice (F§5). We report it as
   released (⚠️) and with both bugs fixed (✅). ✅ follows the released code's design (a row-level mask recomputed
   every step), not the paper's text (a module-level mask fixed at the original weights, Eq. 4–6), which we did not
   run (F§6.1).
4. **Main question (Task 3): no recovery detected.** Neither GPTQ nor AWQ recovered forgotten information in any
   NPO_KLR + SURE model we tested. GA_GDR + SURE ⚠️ is collapsed in BF16 and stays collapsed under RTN, GPTQ and AWQ,
   unlike base GA_GDR, which quantization partly revives. Without SURE, NPO_KLR (one seed) recovers under every
   quantizer: strongly under RTN, bnb-FP4 and AWQ, weakly under GPTQ.
5. **Task 4: no tested configuration produced recovery.** Neither calibration set (general / BOOKS retain) nor group
   size (32 / 128) did, but only the released SURE (⚠️, seed 42) was tested on all eight configurations, and it
   applies no mask.

## Setup and labels

- **Code and data:** the official code, unmodified, with MUSE data and checkpoints pinned. Hyperparameters from
  paper App. D. BF16 training and evaluation. Training seed 42; NPO_KLR + SURE also seed 43.
- **INT4:** llm-compressor RTN / GPTQ / AWQ, W4A16, group 128, GPTQ / AWQ calibrated on WikiText-2.
- **bnb-FP4:** the released code's only 4-bit path, which is FP4, not INT4 (F§3.1). Reported next to INT4; the
  conclusions below hold with INT4 alone.
- **INT4 evaluation:** quantized weights are dequantized once to BF16 and evaluated with the authors' BF16 loader
  (Reproducibility §5).
- **M1** is sampled (T = 0.9), as in the authors' code. Greedy M1 is in parentheses where shown (F§3.2).
- **SURE versions:** ⚠️ = as released (mask never applied); ✅ = the released code with both bugs fixed (a
  deviation, F§6).
- **n/a** = not run (outside this project's compute budget).
- **Definitions, fixed after all seed-42 results and before the seed-43 GPTQ / AWQ results were seen, and before
  any failure case was selected:**
  - **Recovery:** ΔM1 ≥ 10 or ΔM2 ≥ 5 against the model's own BF16 checkpoint, with the paired-bootstrap 95 % CI
    above 0, in every evaluated seed.
  - **Useful:** M4 ≥ 0.75 × M4 and Flu ≥ 0.9 × Flu of the original target under the same quantizer, and Fac > 0.
  - In practice the fluency condition decides most verdicts (e.g. NPO_KLR + RTN is "not useful" with M4 at 101 %
    of target(q)). With an M4 factor of 0.6 or 0.9, or a Flu factor of 0.8, some verdicts change, but no conclusion
    about SURE does (Table 2g).

---

## Task 1 — NEWS

| Row | Source | M1 ↓ | M2 ↓ | M3 → 0 | M4 ↑ |
|---|---|---|---|---|---|
| Original target · BF16 | Reported (Table 1) | 58.4 | 63.9 | -99.8 | 55.2 |
|  | Reproduced | **43.5** (58.2) | **64.3** | **-99.8** | **54.6** |
| Original target + RTN · INT4 | Reported (Table 1) | 34.2 | 54.4 | -99.8 | 48.2 |
|  | Reproduced | **35.5** (46.8) | **52.2** | **-99.8** | **46.7** |
| NPO_KLR · BF16 | Reported (Table 1) | 16.6 | 36.6 | -94.0 | 33.3 |
|  | Reproduced | **15.9** (15.0) | **37.1** | **-93.9** | **33.9** |
| NPO_KLR + RTN · INT4 | Reported (Table 1) | 34.1 | 53.7 | -99.8 | 48.8 |
|  | Reproduced | **34.6** (44.4) | **50.1** | **-99.8** | **47.7** |
| GA_GDR · BF16 | Reported (Table 1) | 0.0 | 28.9 | 87.1 | 34.2 |
|  | Reproduced | **0.0** (0.0) | **27.3** | **109.5** | **23.1** |
| GA_GDR + RTN · INT4 | Reported (Table 1) | 25.0 | 50.1 | -99.1 | 47.7 |
|  | Reproduced | **22.9** (24.2) | **50.8** | **-79.6** | **43.3** |

Tables 1a–1d. M3 reference: the retrained NEWS model.

**Result.** Every row is within about 3 points of the paper, except GA_GDR M3 / M4 and the target's M1. The paper's
central claim is reproduced: both unlearned models regain forgotten knowledge after 4-bit quantization.

**Issues** (details in FINDINGS):
- **The paper describes RTN-INT4, but the released code's only 4-bit path is bitsandbytes FP4** (F§3.1). We report
  both paths (Table 1c). The recovery holds with both.
- **The target's M1 (58.4) equals MUSE's greedy value.** Our greedy value is within 0.2 of it (58.2). The 4-bit rows the
  authors computed themselves are closer to sampled decoding, so the paper probably mixes two protocols (F§3.2).
- **GA_GDR is unstable during training:** M4 45.6 → 1.1 → 23.1 at epochs 2, 5 and 10 (Table 1d, F§4.4).

---

## Tasks 2 and 3 — BOOKS (RTN, GPTQ, AWQ)

| Method | Precision / quantizer | M1 ↓ | M2 ↓ | M3 → 0 | M4 ↑ | Gen ↑ | Tru ↑ | Fac ↑ | Flu ↑ |
|---|---|---|---|---|---|---|---|---|---|
| Original target | BF16 | 99.6 | 46.7 | -57.1 | 68.8 | 28.7 | 34.0 | 9.3 | 597.5 |
| Original target | INT4 / RTN | 78.4 | 32.2 | -58.8 | 44.5 | 29.2 | 37.8 | 8.4 | 640.2 |
| Original target | INT4 / GPTQ | 83.7 | 36.5 | -58.2 | 55.3 | 25.7 | 37.2 | 9.1 | 595.5 |
| Original target | INT4 / AWQ | 93.4 | 39.7 | -58.2 | 56.5 | 30.4 | 35.8 | 8.3 | 606.7 |
| NPO_KLR | BF16 | 11.3 | 10.3 | -37.8 | 18.8 | 24.0 | 34.1 | 3.9 | 518.0 |
| NPO_KLR | INT4 / RTN | 72.0 | 37.3 | -58.8 | 44.8 | 28.7 | 37.7 | 8.2 | 499.9 |
| NPO_KLR | INT4 / GPTQ | 17.4 | 21.6 | -48.7 | 29.4 | 22.2 | 39.2 | 7.4 | 618.0 |
| NPO_KLR | INT4 / AWQ | 47.6 | 29.7 | -56.0 | 44.3 | 26.9 | 37.4 | 7.0 | 669.9 |
| NPO_KLR + SURE ⚠️ | BF16 | 1.0 | 21.7 | -22.5 | 48.3 | 20.5 | 34.4 | 4.2 | 538.3 |
| NPO_KLR + SURE ⚠️ | INT4 / RTN | 1.1 | 17.5 | -25.5 | 46.5 | 22.8 | 32.8 | 3.7 | 585.3 |
| NPO_KLR + SURE ⚠️ | INT4 / GPTQ | 1.1 | 17.8 | -23.7 | 47.6 | 21.1 | 35.5 | 4.0 | 600.0 |
| NPO_KLR + SURE ⚠️ | INT4 / AWQ | 1.2 | 18.0 | -23.3 | 47.5 | 22.2 | 34.2 | 4.6 | 557.8 |
| NPO_KLR + SURE ✅ | BF16 | 7.2 | 31.2 | -48.8 | 52.4 | 21.6 | 33.0 | 8.0 | 471.6 |
| NPO_KLR + SURE ✅ | INT4 / RTN | 5.5 | 32.9 | -47.2 | 46.6 | 24.0 | 34.8 | 7.4 | 426.8 |
| NPO_KLR + SURE ✅ | INT4 / GPTQ | 6.0 | 27.0 | -48.9 | 52.1 | 24.0 | 33.9 | 7.6 | 422.6 |
| NPO_KLR + SURE ✅ | INT4 / AWQ | 6.3 | 32.9 | -48.8 | 43.3 | 24.0 | 35.5 | 8.0 | 443.1 |
| GA_GDR | BF16 | 0.0 | 0.0 | -22.6 | 0.0 | 30.4 | nan | 0.0 | 40.3 |
| GA_GDR | INT4 / RTN | 0.5 | 26.9 | -25.1 | 39.9 | 28.1 | 35.8 | 7.8 | 496.0 |
| GA_GDR | INT4 / GPTQ | 0.0 | 0.0 | -23.8 | 0.0 | 29.2 | nan | 0.0 | 144.5 |
| GA_GDR | INT4 / AWQ | 0.0 | 10.4 | -30.4 | 46.1 | 25.7 | nan | 6.0 | 403.5 |
| GA_GDR + SURE ⚠️ | BF16 | 0.0 | 0.0 | -18.9 | 0.0 | 30.4 | nan | 0.0 | 350.2 |
| GA_GDR + SURE ⚠️ | INT4 / RTN | 0.0 | 0.0 | -18.8 | 0.0 | 30.4 | nan | 0.0 | 288.1 |
| GA_GDR + SURE ⚠️ | INT4 / GPTQ | 0.0 | 0.0 | -18.9 | 0.0 | 30.4 | nan | 0.0 | 349.7 |
| GA_GDR + SURE ⚠️ | INT4 / AWQ | 0.0 | 0.0 | -19.1 | 0.0 | 30.4 | nan | 0.0 | 355.3 |
| GA_GDR + SURE ✅ | BF16 | 1.0 | 0.2 | 4.8 | 0.0 | 30.4 | nan | 0.0 | 17.5 |
| GA_GDR + SURE ✅ | INT4 / RTN, GPTQ, AWQ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Retrained (reference) | BF16 | 13.4 | 30.0 | 0.0 | 69.1 | — | — | — | — |

Tables 2a–2b. Training seed 42. Gen = MMLU, Tru = TruthfulQA MC2, Fac = TriviaQA F1, Flu = fluency. Tru = nan:
undefined for a collapsed model (0/0 in the authors' code). GPTQ / AWQ rows are new experiments.
Evaluation sets (the authors' code): 100 examples for each MUSE metric, MMLU 171, TriviaQA 100, TruthfulQA 50,
fluency 50. With these sizes, Gen and Tru differences of a few points are within sampling noise (e.g. about ±3 points
standard error for MMLU at 25 % accuracy).

### Task 2 — comparison with Table 3 (M1–M4, Reported / Reproduced)

| Row | M1 ↓ | M2 ↓ | M3 → 0 | M4 ↑ |
|---|---|---|---|---|
| Original target · BF16 | 99.8 / **99.6** | 59.4 / **46.7** | -57.5 / **-57.1** | 66.9 / **68.8** |
| NPO_KLR · BF16 | 22.6 / **11.3** | 22.7 / **10.3** | -54.9 / **-37.8** | 50.9 / **18.8** |
| NPO_KLR · 4-bit | 70.9 / **72.0** | 34.2 / **37.3** | -60.1 / **-58.8** | 50.4 / **44.8** |
| NPO_KLR + SURE ⚠️ · BF16 | 17.6 / **1.0** | 37.8 / **21.7** | -58.0 / **-22.5** | 49.4 / **48.3** |
| NPO_KLR + SURE ⚠️ · 4-bit | 16.1 / **1.1** | 36.9 / **17.5** | -58.9 / **-25.5** | 34.9 / **46.5** |
| NPO_KLR + SURE ✅ · BF16 | 17.6 / **7.2** | 37.8 / **31.2** | -58.0 / **-48.8** | 49.4 / **52.4** |
| NPO_KLR + SURE ✅ · 4-bit | 16.1 / **5.5** | 36.9 / **32.9** | -58.9 / **-47.2** | 34.9 / **46.6** |
| GA_GDR · BF16 | 0.0 / **0.0** | 2.9 / **0.0** | -56.5 / **-22.6** | 44.2 / **0.0** |
| GA_GDR · 4-bit | 17.9 / **0.5** | 33.7 / **26.9** | -35.2 / **-25.1** | 51.9 / **39.9** |
| GA_GDR + SURE ⚠️ · BF16 | 0.0 / **0.0** | 0.3 / **0.0** | -6.4 / **-18.9** | 49.3 / **0.0** |

4-bit Reproduced = RTN. With the authors' bnb-FP4 path the base methods come even closer, e.g. NPO_KLR M1 73.3,
M4 49.4 (Table 2h). All eight metrics, with Δ: Tables 2c–2d.

**Result.**
- The target matches, except M2.
- NPO_KLR's 4-bit row matches on M1–M3 (M4 −5.6 under RTN, −1.0 under bnb-FP4). GA_GDR's does not (M1 0.5 vs 17.9).
- The unlearned full-precision rows do not match: NPO_KLR, SURE ⚠️ (over-unlearned, M1 1.0) and GA_GDR (collapsed).
- SURE ✅ comes closer to the paper than the released code on M1–M3: the gap shrinks from 16.6 / 16.1 / 35.5 to
  10.4 / 6.6 / 9.2 points. On fluency it moves further away (BF16 Flu 471.6 vs the paper's 588.8).

**Issues** (details in FINDINGS):
- **Our evaluation is not the cause, as far as we can test.** An A/B run against the authors' `eval.py` (first 5
  examples) gave identical outputs, and the three BOOKS-target metrics the authors shipped (M3, Gen, Fac) match
  to the last digit (F§2). On the training side, the effective batch (ours 1, the authors' scripts suggest 4–16)
  was not tested.
- **Target M2 46.7 vs 59.4:** the reported value equals MUSE's, digit for digit, and an FP32 load gives 45.9
  (F§4.1, Table 2i).
- **The released scripts do not contain the paper's App. D settings** (F§3.3).
- **Base NPO_KLR's training diverged numerically:** its raw-logit "KL" term is unbounded, the loss reached about
  1e19 and the gradient norm was infinite at most logged steps (F§4.2). This is the authors' code as released; NEWS
  NPO_KLR still matches the paper within about 1 point.
- **The paper's Tables 1 and 3 give different full-precision values for the same BOOKS methods,** while their 4-bit
  rows are identical. Our NPO_KLR BF16 M1–M3 are close to Table 1 (F§3.4).
- **SURE as released never applies its mask and counts the forget gradient twice** (F§5). On the 7B model, rows
  outside the mask changed 881,089 times in 20 steps with the released code, and 0 times with the fix (F§6.1).
  This plausibly explains the released version's over-unlearning and seed instability.
- **GA_GDR and GA_GDR + SURE collapse in BF16,** in both SURE versions, so SURE is not needed to explain the collapse (F§4.4–4.5).
  GA_GDR + SURE ⚠️ was still quantized for Task 3 and stays collapsed (below); ✅ was not quantized.

### Task 3 — Can GPTQ or AWQ recover forgotten information after SURE while preserving useful performance?

**No recovery was detected in any NPO_KLR + SURE model we tested.** Every quantized model, compared with its own BF16
checkpoint and with the target under the same quantizer:

| Model | Seed | Quantizer | ΔM1 | ΔM2 | Δ\|M3\| | M4 / target(q) | Useful | Recovery |
|---|---|---|---|---|---|---|---|---|
| NPO_KLR | 42 | bnb-FP4 | **+62.0** | **+22.0** | +22.6 | 96 % | yes | **yes** |
| NPO_KLR | 42 | RTN | **+60.7** | **+27.0** | +21.0 | 101 % | no (Flu) | **yes** |
| NPO_KLR | 42 | GPTQ | +6.1 | **+11.2** | +10.8 | 53 % | no | **yes** |
| NPO_KLR | 42 | AWQ | **+36.3** | **+19.4** | +18.1 | 78 % | yes | **yes** |
| NPO_KLR + SURE ⚠️ | 42 | bnb-FP4 / RTN / GPTQ / AWQ | 0.0 to +0.2 | -4.2 to -0.8 | +0.8 to +3.8 | 84–105 % | yes | no |
| NPO_KLR + SURE ⚠️ | 43 | bnb-FP4 / GPTQ / AWQ | -0.1 to +0.1 | -4.7 to -3.2 | -1.4 to +1.2 | 25–35 % | no | no |
| NPO_KLR + SURE ✅ | 42 | bnb-FP4 / RTN / GPTQ / AWQ | -1.7 to -0.8 | -4.2 to +1.7 | -1.6 to +1.4 | 74–105 % | no (Flu; bnb-FP4 also M4) | no |
| NPO_KLR + SURE ✅ | 43 | bnb-FP4 | 0.0 | -11.5 | -3.1 | 94 % | no (Flu) | no |
| GA_GDR | 42 | bnb-FP4 / RTN / AWQ | ≤ +0.5 | **+10.4 to +31.0** | +2.5 to +9.9 | 82–92 % | no (Flu) | yes (M2) |
| GA_GDR | 42 | GPTQ | 0.0 | 0.0 | +1.2 | 0 % | no | no |
| GA_GDR + SURE ⚠️ | 42 | bnb-FP4 / RTN / GPTQ / AWQ | 0.0 | 0.0 | -0.1 to +0.1 | 0 % | no (collapsed) | no |

Tables 2f–2g give per-row 95 % CIs, the recovered share of the gap to target(q), and all utility deltas.
GA_GDR + SURE ✅ (collapsed in BF16) was not quantized. Base NPO_KLR and GA_GDR were trained with one
seed, so their "every evaluated seed" condition rests on that seed. Bootstrap CIs cover example sampling only.

- **No SURE model showed recovery, while base NPO_KLR did.** Base NPO_KLR recovers under all four quantizers (one
  seed; its training diverged numerically, F§4.2). AWQ recovers 44 % of the M1 gap to
  the quantized target and is still useful. No SURE model recovers: 17 quantized ⚠️ models (13 NPO_KLR, including
  Task 4, and 4 GA_GDR) and 5 ✅ models, over both seeds (INT4 for ✅ seed 43 was not run).
- **GA_GDR + SURE ⚠️ stays collapsed under every quantizer, where base GA_GDR is partly revived.** M1, M2, M4 and Fac
  stay at 0.0 and |M3| moves by at most 0.1 under bnb-FP4, RTN, GPTQ and AWQ; base GA_GDR regains M2 up to +31 and M4
  up to 47 under bnb-FP4, RTN and AWQ. This is "no recovery" only in a trivial sense: the model is not useful before
  or after quantization, so it is not evidence of robustness either.
- **✅ is "not useful" mostly because of fluency it already lost in BF16.** Its BF16 Flu is 471.6, 79 % of the
  target's 597.5, so it is below the 0.9 gate before quantization; INT4 lowers it by a further 29–49 points
  (Table 2g). The verdict says little about quantization itself.
- **"Not detected" is weaker for M2 than for M1.** On M1 the CIs are narrow (about ±1) and the headroom is large, so
  the result is strong. On M2, ✅ is already close to the quantized target (31.2 vs 32.2 under RTN) and its CIs reach
  +7.4, so an M2 recovery of 5 points is not excluded.
- **For base NPO_KLR (one seed), GPTQ and AWQ were milder than RTN, not harsher.** One hypothesis for this experiment was that GPTQ / AWQ preserve the unlearning
  changes worse than RTN. For base NPO_KLR the share of the M1 gap recovered is: RTN 90 %, bnb4 80 %, AWQ 44 %,
  GPTQ 8 % (Table 2f). The paper's Table 2 (NEWS) reports GPTQ and AWQ close to RTN; we did not investigate the
  difference. GPTQ is symmetric and AWQ asymmetric here, so the ranking also mixes in symmetry.
- **Why SURE is robust is not isolated.**
  - The released SURE never applies its mask, so its robustness cannot come from saliency. It differs from base
    NPO_KLR in four ways at once: lr (1e-4 vs 1e-5), α (20 vs 2), a proper KL instead of the diverging raw-logit
    term, and a doubled forget gradient.
  - With the mask applied, no recovery was detected either (5 quantized models, Table 2f).
  - This is consistent with the paper's own design rationale (a large learning rate against recovery, the mask for
    utility; App. I). The paper's mask-free ablation (SURE/S, Table 10) was not quantized, so "no recovery without a
    mask" is new evidence; but its utility collapsed (M4 0.0) while our mask-free ⚠️ keeps M4 48.3, which we cannot
    explain (F§6.4).

---

## Task 4 — Sensitivity to calibration set and group size (NPO_KLR + SURE ⚠️, seed 42)

Only the released SURE was run on all eight configurations. It applies no saliency mask (F§5), so this tests the
released training recipe rather than saliency masking.

| Quantizer | Calibration | Group size | M1 ↓ | M2 ↓ | M3 → 0 | M4 ↑ | Gen ↑ | Tru ↑ | Fac ↑ | Flu ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| — (BF16) | — | — | 1.0 | 21.7 | -22.5 | 48.3 | 20.5 | 34.4 | 4.2 | 538.3 |
| GPTQ | General text | 32 | 0.9 | 18.3 | -23.3 | 46.8 | 21.1 | 33.9 | 4.0 | 577.5 |
| GPTQ | General text | 128 | 1.1 | 17.8 | -23.7 | 47.6 | 21.1 | 35.5 | 4.0 | 600.0 |
| GPTQ | BOOKS retain-only | 32 | 1.0 | 17.6 | -23.6 | 49.6 | 19.3 | 35.1 | 4.8 | 581.2 |
| GPTQ | BOOKS retain-only | 128 | 1.2 | 18.5 | -23.5 | 47.5 | 21.6 | 34.7 | 4.7 | 588.0 |
| AWQ | General text | 32 | 1.0 | 20.2 | -21.2 | 48.7 | 22.2 | 35.8 | 4.0 | 546.9 |
| AWQ | General text | 128 | 1.2 | 18.0 | -23.3 | 47.5 | 22.2 | 34.2 | 4.6 | 557.8 |
| AWQ | BOOKS retain-only | 32 | 1.0 | 21.0 | -21.7 | 52.0 | 22.2 | 35.2 | 3.3 | 563.4 |
| AWQ | BOOKS retain-only | 128 | 1.1 | 17.1 | -23.4 | 48.7 | 22.2 | 34.3 | 3.9 | 536.2 |

Table 4a; the dev / held-out halves are in Table 4b.

- **Calibration** data excludes every paragraph that shares a 13-gram with forget or evaluation text.
- **Matched across GPTQ and AWQ:** 128 × 2048 tokens, calibration seed 0.
- **Settings:** W4A16. GPTQ is symmetric and AWQ asymmetric (llm-compressor 0.14.0 presets; the full recipe of every run is in its `quant_report`).

**Result.** No tested configuration produced recovery.
- M1 stays at 0.9–1.2, against 1.0 in BF16.
- M2 is below the BF16 value in all eight configurations.
- Utility is essentially unchanged (M4 46.8–52.0 vs 48.3).
- AWQ with group size 32 comes closest to recovery (M2 20.2 / 21.0; no CIs for these six configurations), but
  stays below BF16.
- Dev / held-out check: the configuration selected on dev examples (AWQ g32 BOOKS-retain, highest dev M2 21.7) has
  M1 0.9 and M2 20.3 on the held-out examples. That is the joint highest held-out M2, and still below the full-set
  BF16 value (21.7, not directly comparable).

**Not tested:**
- GA_GDR + SURE: none of the eight configurations. It is collapsed in BF16 and stays collapsed under RTN, GPTQ and
  AWQ with the standard settings (M4 0.0, Fac 0.0), so there is no useful model to stress-test, and the remaining compute
  went to the NPO_KLR runs.
- SURE ✅ beyond g128 / general, so saliency masking itself was not stress-tested across configurations.
- The target under 5 of the 8 configurations, so recovery shares relative to the target are missing for those.

---

## Analysis

- **Compare against the model's own BF16 checkpoint and against the target under the same quantizer:** done for the
  standard configurations (Tables 2f–2g) and 3 of the 8 Task 4 configurations; the other 5 have no target control.
- **Recovery (increase in M1, M2):** strong for base NPO_KLR (up to +62 M1). M2-only recovery for GA_GDR. None for
  any SURE model.
- **Privacy (change in |M3|):** worse for base NPO_KLR (+11 to +23). Small but significant for SURE ⚠️ seed 42
  (+0.8 to +3.8, CI above 0), even though M1 and M2 did not recover.
- **Utility (M4, Gen, Tru, Fac, Flu):**
  - SURE ⚠️ seed 42: quantization barely changes utility (ΔM4 −3.3 to −0.7).
  - SURE ✅: loses M4 under bnb4 (−14), similar to the paper's 4-bit drop (−14.5). Its fluency is low already in
    BF16.
  - GA_GDR: quantization restores utility that the collapsed BF16 model had lost. GA_GDR + SURE ⚠️: it does not
    (M4 0.0 under every quantizer).
- **Does SURE reduce recovery?** No SURE model showed recovery where base NPO_KLR did (Task 3 table). Which
  ingredient is responsible is not isolated (F§6.4).
- **Utility threshold:** defined before any failure case was selected, after the seed-42 results (Setup).
- **Seed repeats:**
  - SURE ⚠️ and ✅, seed 43 (Table 2e): neither recovers in either seed.
  - SURE ⚠️ utility is seed-unstable: BF16 M4 is 48.3 with seed 42 and 21.4 with seed 43.
  - SURE ✅ differs less on M4 (52.4 / 63.3) and M1, M3 are nearly identical, but M2 differs more (31.2 / 44.4).
    Two seeds cannot establish stability.
  - No SURE failure case was found, so the calibration-seed repeats were not needed.
- **Robust cases:** every NPO_KLR + SURE model and quantizer we tested, and all eight Task 4 configurations of SURE ⚠️.
- **Is there a convincing failure of SURE?** **None found.** No tested SURE run met the recovery criterion. The only models that recover and stay useful are base NPO_KLR under bnb-FP4 and AWQ, without
  SURE (with a Flu factor of 0.8, also base GA_GDR under bnb-FP4, on M2 only).
- **New failure, or worse forgetting that was already incomplete?**
  - For NPO_KLR, probably the second, but the evidence does not separate the two. Under RTN and bnb-FP4, quantization closes 80–90 % of the M1 gap to the quantized
    target (AWQ 44 %, GPTQ 8 %). This is consistent with the paper's explanation, that forgetting which looks
    complete in BF16 rests on weight changes smaller than one quantization step. We did not run the weight-level
    test that would show it directly. Measured against the retrained reference instead, BF16 NPO_KLR is
    over-forgotten (M1 11.3, M2 10.3 vs 13.4, 30.0; Table 2a), so "incomplete" refers to the weights, not the
    metrics.
  - For GA_GDR, quantization does something different: it undoes a collapse, bringing back QA ability (M2, M4) but
    not verbatim text (F§4.4).

## Which quantization configurations most strongly undermine SURE, and how do they affect utility?

**No tested configuration undermined SURE** among the 17 quantized SURE ⚠️ models and the 5 SURE ✅ models we
tested. The configuration space described in Task 3–4 was not fully covered (see Not run).
- **Closest to recovery:** AWQ with group size 32, which gives the highest M2. M2 still stays below the unquantized
  model's (Task 4).
- **Utility:** these configurations leave SURE ⚠️ (seed 42) about as useful as before, with M4 46.8–52.0 against
  48.3 in BF16.
- **For contrast, base NPO_KLR:** RTN and bnb-FP4 undermine unlearning most (80–90 % of the M1 gap recovered).
  Under INT4 only AWQ keeps the model useful (RTN fails the Flu gate); under bnb-FP4 it stays useful too.

## Not run (outside this project's compute budget)

- **GA_GDR + SURE:** ✅ under RTN / GPTQ / AWQ, and both versions under the Task 4 configurations (collapsed in
  BF16; ⚠️ also stays collapsed under the standard quantizers).
- **SURE ✅:** the 6 non-standard Task 4 configurations, and seed 43 under RTN / GPTQ / AWQ.
- **Task 4:** target controls for 5 configurations.
- **Base methods:** only one training seed.
- **The paper's SURE as written** (module-level mask fixed at the original weights).
- **Presets defined in our code but not run:** a proper-KL base NPO_KLR, a single-fix and a fixed-mask SURE, and a
  test of the authors' earliest commit (FINDINGS §6.5).
- **A larger effective training batch** (gradient accumulation), a candidate explanation for the BF16 gaps.
- **The weight-level test** of the "incomplete forgetting" reading (`weight_index_diff`).

Checkpoints and outputs of every run: Manifest in [`tables/tables.md`](tables/tables.md). Commands:
[`COMMANDS.md`](COMMANDS.md).