# Detailed Findings: Discrepancies, Code Issues and the SURE Investigation

This file gives the full evidence for every discrepancy with the paper and every code problem we found. The other
documents refer to its sections as F§N:
- [`REPORT.md`](REPORT.md): the main report.
- [`tables/tables.md`](tables/tables.md): all measured values, cited below as "Table 1a", "Table 2c", ….
- [`Reproducibility.md`](Reproducibility.md): setup, versions and seeds.

- **Paper:** Zhang et al., [*Catastrophic Failure of LLM Unlearning via Quantization*](https://arxiv.org/abs/2410.16454), ICLR 2025.
- **Authors' code:** [zzwjames/FailureLLMUnlearning](https://github.com/zzwjames/FailureLLMUnlearning), used unmodified at commit [`10131ae`](https://github.com/zzwjames/FailureLLMUnlearning/tree/10131ae25f55f1d8feb744eabb235ffc3f094b1b). Every `file.py:N` reference to the authors' code below is to this commit.

## 1. Conventions

- **Metrics:** M1 = VerbMem, M2 = KnowMem (forget), M3 = PrivLeak (→ 0), M4 = KnowMem (retain); Gen = MMLU,
  Tru = TruthfulQA MC2, Fac = TriviaQA F1, Flu = fluency. All values are ×100.
- **M1** is sampled (T = 0.9, as in the authors' code) with a fixed per-example seed. Greedy M1 is shown in
  parentheses.
- **4-bit paths:** "bnb4" = the authors' only 4-bit path (bitsandbytes FP4). "RTN" = llm-compressor INT4 W4A16,
  symmetric, group 128.
- **SURE versions:** ⚠️ = SURE as released; ✅ = the released code with its two bugs fixed (§6; a deviation, and not
  the paper's module-level mask).
- **Row labels:** Reported (Table N) = paper; Reproduced = ours; Δ = Reproduced − Reported.

## 2. Is our pipeline the cause?

We checked our own pipeline before interpreting any discrepancy:
- **Evaluation:** an A/B test ran the authors' `eval.py` and our `extra/eval_run.py` on the same model and examples.
  The outputs were identical. On the BOOKS target, our M3, Gen and Fac equal the authors' shipped `output.csv` to
  the last digit.
- **Training:** our wrapper's default path is bitwise identical to calling the authors' trainer directly (CPU test
  on a small model; GPU training is not bitwise deterministic, so two identical GPU runs already differ).
- **Data:** the data in the authors' repository equals the pinned MUSE datasets on Hugging Face, in every split.

So our evaluation does not cause the discrepancies below. Training settings the paper does not give, notably the
effective batch (§3.3), were not tested and remain a candidate cause.

## 3. Issues in the paper's description of its setup

### 3.1 The paper's "RTN 4-bit" is bitsandbytes FP4 in the code

- **What the paper says:** quantization is uniform integer quantization (Eq. 2), and App. E says the 4-bit models
  use RTN.
- **What the code does:** its only 4-bit path is in
  [`utils.py:94-101`](https://github.com/zzwjames/FailureLLMUnlearning/blob/10131ae25f55f1d8feb744eabb235ffc3f094b1b/utils.py#L92-L101):

```python
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
return AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto',
                                            quantization_config=bnb_config, torch_dtype=torch.bfloat16, **kwargs)
```

- **What that means:** with transformers 4.40.0 defaults
  ([`quantization_config.py#L243-L264`](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/utils/quantization_config.py#L243-L264)),
  this is **FP4**: 16 non-uniform levels, one scale per block of 64, float32 compute. It is neither uniform nor
  integer.
- **Our runs confirm it:** every bnb4 `metrics.json` records the configuration actually loaded,
  `"bnb_4bit_quant_type": "fp4"`.
- **What we report:** both paths, side by side (Tables 1c, 2h).
- **Which one did the paper use?** The numbers cannot decide: on M1 of the published 4-bit rows, neither path is
  consistently closer.
- **Effect on the main result:** none. Recovery under 4-bit holds with both paths.

### 3.2 M1 mixes greedy and sampled decoding

- **Neither paper states the decoding.** This paper and MUSE (Shi et al., [*MUSE: Machine Unlearning Six-Way Evaluation for Language Models*](https://arxiv.org/abs/2407.06460)) are both silent on it.
- **MUSE's code is greedy:** `do_sample=False`
  ([`muse_bench/metrics/verbmem.py:25-29`](https://github.com/jaechan-repo/muse_bench/blob/main/metrics/verbmem.py#L25-L29)).
- **The authors' code samples:** `do_sample=True, temperature=0.9`, with no seed
  ([`metrics/verbmem.py:25-33`](https://github.com/zzwjames/FailureLLMUnlearning/blob/10131ae25f55f1d8feb744eabb235ffc3f094b1b/metrics/verbmem.py#L25-L33)).
- **The full-precision Target and Retrain rows of the paper's Table 1 are identical to MUSE Table 3,** digit for
  digit, on both corpora and all four metrics, which suggests they were taken from MUSE (FP32, greedy decoding). The
  4-bit rows of the same models, which the authors computed themselves, are closer to the sampled protocol:

| Row | Reported | Reproduced, sampled | Reproduced, greedy | Closer |
|---|---|---|---|---|
| NEWS target · BF16 | 58.4 | 43.5 | **58.2** | greedy |
| NEWS target · 4-bit | 34.2 | **36.4** | 46.0 | sampled |
| NEWS NPO_KLR · 4-bit | 34.1 | **33.0** | 42.9 | sampled |
| BOOKS target · 4-bit | 85.3 | **88.8** | 94.1 | sampled |
| BOOKS NPO_KLR · 4-bit | 70.9 | **73.3** | 84.1 | sampled |

(4-bit = bnb4.)

- **Consequence:** the paper's NEWS target drop under 4-bit (58.4 → 34.2, −24) probably compares two protocols. With one
  protocol the drop is −12 (greedy) or −7 (sampled).
- **Effect on the unlearned rows:** none. They all use the sampled protocol.

### 3.3 The released scripts do not contain the paper's settings

- **The paper:** gives BOOKS settings in App. D (Table 4 for base methods, Table 5 for SURE).
- **The code:** has only argument defaults and two scripts. In the final commit the BOOKS script runs only `rmu`.
  No version of it in the git history uses 5 epochs, α ∈ {2, 100} or the SURE settings, and none runs
  `npo_klr_sure` or `ga_gdr_sure` on BOOKS.
- **What we did:** used App. D. Only values the paper omits and every script version shares were taken from the
  scripts (`max_len` 2048, `retain1.txt`).
- **Settings the paper does not give:**
  - Batch: ours 1 GPU × 1 (BOOKS), 1 × 2 (NEWS). The scripts suggest 4–8 GPUs, i.e. an effective batch of about 4–16.
  - Training seed: ours 42. The authors set none.
  - Evaluated checkpoint: ours is the final epoch. The paper does not say which one it used.
- **Effect of the batch (untested):** with batch 1, our BOOKS runs take 2,765 optimizer steps. A batch of 4–16 at the
  same lr would take 4–16× fewer, which could explain stronger forgetting in BF16, GA_GDR's collapse, and (by the
  paper's own argument) less recovery.

### 3.4 The paper's Tables 1 and 3 contradict each other (BOOKS)

App. D.2 says Table 3's base methods use the same setup as Table 1. Yet three rows differ in full precision while
their 4-bit rows are identical:

| BOOKS (M1 / M2 / M3 / M4) | Table 1 · BF16 | Table 3 · BF16 | 4-bit (both tables) |
|---|---|---|---|
| NPO_KLR | 12.4 / 13.7 / −40.7 / 35.1 | 22.6 / 22.7 / −54.9 / 50.9 | 70.9 / 34.2 / −60.1 / 50.4 |
| GA_KLR | 13.0 / 15.1 / −40.8 / 33.7 | 23.8 / 25.1 / −54.5 / 51.9 | 75.6 / 34.6 / −60.0 / 51.3 |
| NPO_GDR | 0.4 / 13.4 / −42.6 / 58.6 | 3.2 / 27.4 / −51.2 / 57.0 | 66.0 / 31.9 / −60.8 / 53.2 |

A 4-bit model is made from one full-precision model. So for each of these rows, at least one of the two
full-precision values cannot come from the run that produced the shared 4-bit row.

### 3.5 Other inconsistencies in the text

| Item | Paper text | Code / data |
|---|---|---|
| PrivLeak formula (§4.1) | divides by AUC(f_unlearn) | divides by AUC(f_retrain), as MUSE does. Reported numbers follow the code. |
| Min-K% | K not stated | K = 40 % |
| TruthfulQA | MC1 | published values match MC2 (we report MC2; MC1 in Table 2j, 18–24 for every model, n = 50) |
| Tru, GA_GDR + SURE | 0.2 / 0.18 | every other row is ×100 (likely the wrong scale) |

## 4. Discrepancies in the BOOKS results (Tables 2c, 2d)

### 4.1 Target: M2 46.7 vs 59.4

- **Everything else matches:** M1, M3, M4 and all utility metrics.
- **Not caused by evaluation precision:** an FP32 load gives 45.9 (Table 2i).
- **The authors' own data:** their shipped `output.csv` has 46.9.
- **Conclusion:** the reported 59.4 equals MUSE's value (§3.2). The cause of the gap cannot be determined.

### 4.2 NPO_KLR: BF16 far from Table 3, 4-bit matches

- **BF16:** M1 11.3 vs 22.6, and M4 18.8 vs 50.9. M1–M3 are close to the paper's **Table 1** row instead (§3.4); M4
  is not (18.8 vs 35.1).
- **4-bit:** matches Table 3 (bnb4 M1 73.3 vs 70.9).
- **Training diverged numerically** (authors' code as released). The base trainer's "KL" is computed on raw logits
  and is unbounded (§5.3):
  - BOOKS: loss about −1.4e19 at step 500; gradient norm infinite at steps 1000, 1500, 2000 and 2500 (4 of 5 logged
    steps).
  - NEWS: loss about −2.5e20; gradient norm infinite at every logged step from 500.
  - What the optimizer applied on steps with a non-finite norm is not logged. SURE and GA runs stay finite.
  - NEWS NPO_KLR still matches the paper within about 1 point, so the authors' runs may have behaved the same way
    (not verifiable).
  - This matters because base NPO_KLR is the only BOOKS model that shows recovery. A proper-KL variant is defined in
    our code but was not run.

### 4.3 NPO_KLR + SURE ⚠️: over-unlearning and seed instability

- **Seed 42:** forgets far more than the paper (M1 1.0 vs 17.6, M3 −22.5 vs −58.0), while M4 matches (48.3 vs 49.4).
- **Seed 43,** same code and settings: M4 halves (21.4), and M2 and M3 also move a lot (Table 2e).
- §5 explains both.

### 4.4 GA_GDR: collapse in BF16, utility returns under 4-bit

- **BF16 collapses:** M4 0.0, Flu 40.3, Tru undefined (0/0 in the authors' MC2 code). The paper reports a usable
  model (M4 44.2).
- **4-bit:** M2–M4 come back close to the paper (bnb4 M2 31.0, M4 47.3), but M1 stays ≈ 0 (paper 17.9), so this
  4-bit row does not reproduce. Under GPTQ the model stays collapsed (M4 0.0).
- **Interpretation:** no verbatim memorization, but most forgotten knowledge is back. This is neither clean
  forgetting nor clean recovery.
- **Cause:** no code defect we found affects GA_GDR. A candidate is the effective batch (ours 1, theirs probably
  4–16); it was not tested.
- **NEWS:** GA_GDR is also unstable during training. M4 goes 45.6 → 1.1 → 23.1 at epochs 2 → 5 → 10 (Table 1d), and
  the loss runs away near step 427.

### 4.5 GA_GDR + SURE: collapse in both versions

- **Collapse:** M4 = 0 with SURE ⚠️ and with SURE ✅. Base GA_GDR collapses the same way, so SURE is not needed to
  explain the collapse; a matched-hyperparameter ablation would be needed to rule out a contribution from it.
- **The paper's row is not collapsed by its M4 (49.3) or Flu (544.9).** Its Tru (0.2) and Fac (0.0) are near zero,
  and the Tru value may be on the wrong scale (§3.5).
- **Quantization (⚠️ only) does not revive it.** Under bnb-FP4, RTN, GPTQ and AWQ, M1, M2, M4 and Fac stay 0.0,
  |M3| moves by at most 0.1 and Flu stays at 288–355 (Tables 2b, 2f). Base GA_GDR, in contrast, regains M2 and M4
  under bnb-FP4, RTN and AWQ (§4.4). The SURE recipe (lr 1e-4, α 400, P99) leaves a collapse that 4-bit rounding does
  not undo; why is not investigated. Gen is 30.4 in every collapsed row, the same value as base GA_GDR in BF16.
- **Not quantized:** ✅, and both versions under the Task 4 configurations (no useful model to stress-test).

## 5. SURE as released: two bugs ([`baselines/baselines/iterative.py`](https://github.com/zzwjames/FailureLLMUnlearning/blob/10131ae25f55f1d8feb744eabb235ffc3f094b1b/baselines/baselines/iterative.py))

### 5.1 The saliency mask is never applied

- **What SURE should do:** update only the parameters most salient for forgetting. The paper (Eq. 4–6) computes a
  **module-level** mask **once**, from the forget-loss gradient at the original weights θo, and keeps it fixed. The
  released code instead computes a **row-level** mask at **every step** from the current forget batch (top 10 % of
  rows for P90).
- **How the code tries:**
  - `SURE.compute_loss` builds the mask `m_S` (lines 278–305).
  - The mask is applied only inside a method named `optimizer_step` (lines 339–357).
- **Why it fails:** the Hugging Face `Trainer` has no `optimizer_step` hook. Its loop calls `self.optimizer.step()`
  directly ([`trainer.py#L2266`](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py#L2266),
  v4.40.0), so every parameter is updated at every step:

```
SURE.train() → training_step → compute_loss (builds m_S, backward #1) → backward #2
             → clip_grad_norm_ → self.optimizer.step()   # all parameters
SURE.optimizer_step(...)                                  # never reached
```

**Evidence:**
1. A call counter in our wrapper: **0 calls in 2,765 steps**, in every SURE run (`unlearn_run.json → counters`).
2. A 20-step GPU test on the 7B model, auditing five weight matrices (layers 0 and 31 `q_proj` / `down_proj`,
   and `lm_head`): rows outside that step's mask changed **881,089 times** in total (per-step counts summed). With a
   working mask this would be 0. The same audit on the fixed version gives 0 (§6.1).

The bug has been present since the first commit that contains SURE (`1882021`, 2024-10-20). There the saliency
trainer is still named `IterativeUnlearner`, and `optimizer_step` is likewise the only place the mask is applied.

### 5.2 The forget gradient is counted twice

- **First backward:** line 282 runs `loss_f.backward(retain_graph=True)` to build the mask. The gradients stay in
  `param.grad`.
- **Second backward:** the Trainer then calls `backward` on the total loss (`loss_f + α · KL_r`).
- **Result:** the update uses **2 × ∇forget + α · ∇retain**, i.e. α is effectively halved.

### 5.3 Not changed by us

- **Base trainer's "KL"** (used by NPO_KLR without SURE) is computed on raw logits with `log_target=True`
  (lines 173–178). It is not a true KL.
- **NPO term** uses raw logits instead of target-token log-probabilities, in both trainers (lines 160 and 271).
- **SURE's own KL** is a proper KL (lines 330–334).

### 5.4 What actually ran as "SURE"

NPO_KLR with:
- lr 10× larger (1e-4 vs 1e-5),
- α 10× larger (20 vs 2),
- a proper KL on the retain set instead of the base trainer's diverging raw-logit term (§4.2, §5.3),
- **no mask**,
- **double weight on the forget gradient**.

This plausibly explains the over-unlearning and the seed instability (§4.3).

## 6. SURE ✅: the released code with its two bugs fixed (deviation)

### 6.1 The fix and its verification

[`extra/fixed_unlearn.py`](extra/fixed_unlearn.py) subclasses the authors' `SURE` and leaves their file untouched.
Presets `books_*_sure_masked` keep every Table 5 hyperparameter and add two fixes. They follow the released code's
design (a row-level mask recomputed every step), **not** the paper's text (a module-level mask fixed at θo,
§5.1), which we did not run:
- `sure_mask="step"`: the mask is recomputed every step. Gradients of rows outside it are zeroed before clipping,
  and those rows are saved before `optimizer.step()` and written back bitwise afterwards. The restore is needed
  because AdamW momentum would otherwise still move rows that were salient in earlier steps. The optimizer state
  itself is not reset.
- `sure_single_grad=True`: the forget gradient is used once.

Verification:
- **CPU unit test:** rows outside the mask stay unchanged.
- **Full 7B runs:** the mask size (`stats.fix_mask_stats.salient_frac`, logged for the first 50 steps) is 0.100 of
  all rows for NPO_KLR (P90) and 0.010 for GA_GDR (P99). How many distinct rows were updated over the whole run is
  not recorded.
- **7B audit, like for like with the released code.** Same 20 steps, seed 42, batches, five audited matrices and
  per-step definition; the step-1 forget cross-entropy, which depends on the batch, is identical (0.0439632).

| 7B, 20 steps | SURE ⚠️ released | SURE ✅ fixed |
|---|---|---|
| Rows changed outside that step's mask (summed over steps) | 881,089 | **0** |
| Rows outside the mask with a non-zero gradient after masking | all (no mask applied) | **0** |

AdamW momentum still moved rows that had been salient in earlier steps (78,161 such moves in 20 steps). The restore
reverted every one of them, and rows never inside the mask never moved.

Raw outputs (run records of both audits, field `verdict.D2_rows_changed_outside_mask_total`):
[released](evidence/20261003-191402_test_sure_equivalence_books_npo_klr_sure.json), [fixed](evidence/20261007-175151_test_sure_equivalence_books_npo_klr_sure_masked.json).

### 6.2 Results vs the paper (Tables 2a, 2c, 2e)

- **Closer to the paper on M1–M3 and Fac:** the BF16 distance drops from 16.6 / 16.1 / 35.5 / 3.2 (⚠️, s42) to
  10.4 / 6.6 / 9.2 / 0.6 (✅, s42). **Further on fluency:** Flu gap −117.2 (✅) vs −50.5 (⚠️) in BF16, and −165.8 vs
  −7.3 under 4-bit.
- **Not a match:** M1 ≈ 7 (vs 17.6) and M3 ≈ −48 (vs −58) in both seeds, so the gap is systematic. Flu is lower
  (≈ 475 vs 589).
- **Seeds:** the M4 gap between seeds shrinks from 27 points (⚠️) to 11 (✅), and M1 and M3 are nearly identical; but
  the M2 gap grows (13.2 vs 9.6). Two seeds cannot establish stability.

### 6.3 No recovery detected under any quantizer (Table 2f)

- **All quantized ✅ models:** ΔM1 between −1.7 and 0.0, and no ΔM2 has a CI above 0. Tested: bnb4 (s42, s43), RTN,
  GPTQ and AWQ (s42).
- **M2 has little room to show recovery:** ✅'s BF16 M2 (31.2) is already close to the quantized target's (32.2
  under RTN), and the ΔM2 CIs reach +7.4 (RTN) and +5.8 (AWQ), so an M2 recovery of 5 points is not excluded. On M1
  the CIs are narrow and the result is strong.
- **Utility:** the bnb4 M4 drop (−14.3 / −14.9) is similar to the paper's 4-bit vs full-precision drop (−14.5).

### 6.4 Learning rate vs mask

The paper separates the two in its own ablation (App. I, Table 10): "SURE/S" removes the saliency mask and keeps
lr 1e-4, with hyperparameters re-tuned per method. Its NPO_KLR + SURE/S row is M1 3.6, M2 0.0, M3 −31.4,
**M4 0.0**, Flu 368.2: the mask-free version forgets more and loses its utility. The paper concludes that the mask
maintains utility and avoids bias toward the retain set; SURE/S was not quantized.

Our three NPO_KLR variants:

| | Mask | lr | α | Retain term | Forget gradient | Recovery under 4-bit? | BF16 M4 (s42 / s43) |
|---|---|---|---|---|---|---|---|
| NPO_KLR | — | 1e-5 | 2 | raw-logit "KL" (diverged, §4.2) | 1× | **yes** (M1 11.3 → 73.3) | 18.8 / — |
| NPO_KLR + SURE ⚠️ | none | 1e-4 | 20 | proper KL | 2× | no (13 of 13 quantized models) | 48.3 / 21.4 |
| NPO_KLR + SURE ✅ | yes | 1e-4 | 20 | proper KL | 1× | no (5 of 5) | 52.4 / 63.3 |

- **What this shows:** a mask-free model trained with SURE's recipe (⚠️) does not recover under quantization. The
  paper did not test that, so it is new evidence, and it is consistent with the paper's design rationale (a large
  learning rate against recovery, the mask for utility).
- **What it does not show:** which ingredient prevents recovery. Base NPO_KLR and ⚠️ differ in lr, α, the retain term
  and the forget-gradient weight at once.
- **An unexplained contrast:** the paper's mask-free SURE/S collapses (M4 0.0), our mask-free ⚠️ does not (M4 48.3
  with seed 42, 21.4 with seed 43). Candidates: the doubled forget gradient, the paper's re-tuned hyperparameters, the
  batch. Not tested.

### 6.5 Limitations

1. The two fixes were applied together, so their effects cannot be separated.
2. The paper's SURE as written (module-level mask fixed at θo) was not run. Presets defined in
   `extra/fixed_unlearn.py` but not run: proper-KL base NPO_KLR (`*_properkl`), sequence-level NPO (`*_seqnpo`),
   single-gradient-only SURE (`*_sure_single`), fixed-mask SURE (`*_sure_maskfixed`), and a reconstruction of the
   authors' earliest commit (`books_npo_klr_as1882021`).
3. Only two seeds were run. RTN, GPTQ and AWQ were run on seed 42 only.
4. A gap to the paper remains (M1, M3, Flu). Untested candidates: the NPO term on logits (§5.3), the effective
   batch, the number of epochs.
5. Task 4 used the released SURE only. With ✅, only the standard configuration (g128, general) was run.

## 7. Summary

1. **NEWS reproduces** within about 3 points, except GA_GDR (M3, M4). Recovery under 4-bit is confirmed with bnb4
   and RTN.
2. **The paper's 4-bit is FP4 in the code** (§3.1), its M1 mixes two decoding protocols (§3.2), and its BOOKS
   Tables 1 and 3 contradict each other (§3.4).
3. **BOOKS:** NPO_KLR's 4-bit row reproduces; GA_GDR's does not. BF16 rows do not, GA_GDR collapses, and base
   NPO_KLR's training diverges numerically in the authors' code (§4).
4. **SURE as released never applies its mask and double-counts the forget gradient** (§5). It is NPO_KLR with a
   10× learning rate and α, a proper KL, and no mask.
5. **SURE ✅** (the released code with both bugs fixed) is closer to the paper on M1–M3 but not on fluency. No
   quantizer showed recovery. Which ingredient prevents recovery is not isolated; the paper's own ablation already
   ties the mask to utility (§6).

## 8. Code references

- **Authors' code** (commit [`10131ae`](https://github.com/zzwjames/FailureLLMUnlearning/tree/10131ae25f55f1d8feb744eabb235ffc3f094b1b)):
  - [`iterative.py`](https://github.com/zzwjames/FailureLLMUnlearning/blob/10131ae25f55f1d8feb744eabb235ffc3f094b1b/baselines/baselines/iterative.py):
    base NPO 160–161, base KL 172–179, SURE `compute_loss` 240–337, `optimizer_step` 339–357.
  - [`utils.py:92-115`](https://github.com/zzwjames/FailureLLMUnlearning/blob/10131ae25f55f1d8feb744eabb235ffc3f094b1b/utils.py#L92-L115):
    model loader, 4-bit path.
  - [`metrics/verbmem.py:25-33`](https://github.com/zzwjames/FailureLLMUnlearning/blob/10131ae25f55f1d8feb744eabb235ffc3f094b1b/metrics/verbmem.py#L25-L33):
    VerbMem decoding.

- **Our code:**
  - [`extra/common.py`](extra/common.py): presets.
  - [`extra/unlearn_run.py`](extra/unlearn_run.py): training, seed, `optimizer_step` counter.
  - [`extra/fixed_unlearn.py`](extra/fixed_unlearn.py): fixed SURE.
  - [`extra/eval_run.py`](extra/eval_run.py): evaluation.
  - [`extra/quantize_run.py`](extra/quantize_run.py): RTN / GPTQ / AWQ.
  - Tests: [`extra/tests/`](extra/tests/).