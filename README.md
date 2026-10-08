# failure-llm-unlearning-reproduction

Independent reproduction of *Catastrophic Failure of LLM Unlearning via Quantization* (ICLR 2025) on MUSE, with
GPTQ/AWQ stress tests of SURE and an audit of its released implementation.

This repository reproduces Zhang et al., [*Catastrophic Failure of LLM Unlearning via Quantization*](https://arxiv.org/abs/2410.16454)
(ICLR 2025). It uses the MUSE NEWS and BOOKS benchmarks with their Llama-2-7B checkpoints, and runs the authors' code
unmodified ([zzwjames/FailureLLMUnlearning](https://github.com/zzwjames/FailureLLMUnlearning) at commit
[`10131ae`](https://github.com/zzwjames/FailureLLMUnlearning/tree/10131ae25f55f1d8feb744eabb235ffc3f094b1b), linked,
not included). It then tests whether GPTQ or AWQ quantization can break SURE, the defence the paper proposes, and
audits the released SURE code. It was done as a course project and is not affiliated with the paper's authors.

## The four experiments

- **Task 1 — NEWS.** Reproduce the paper's NEWS results for NPO_KLR and GA_GDR: the target model, both unlearned
  models in BF16, and all three after RTN-INT4 quantization.
- **Task 2 — BOOKS.** Reproduce the selected BOOKS rows of the paper's Table 3: the target, NPO_KLR and GA_GDR, both
  with and without SURE, before and after 4-bit quantization, on all eight metrics.
- **Task 3 — GPTQ / AWQ.** Quantize the same BOOKS checkpoints with GPTQ-INT4 and AWQ-INT4 (target as control) and
  ask: *can GPTQ or AWQ recover forgotten information after SURE while keeping the model useful?*
- **Task 4 — calibration and group size.** Vary the calibration set (general text vs BOOKS retain data) and the
  group size (32 / 128) for GPTQ and AWQ.

Unlearning and evaluation are in BF16 throughout; every quantized model is made directly from its BF16 checkpoint.
Metrics: M1 = VerbMem, M2 = KnowMem (forget), M3 = PrivLeak (→ 0), M4 = KnowMem (retain); utility on BOOKS: MMLU,
TruthfulQA, TriviaQA, fluency.

## Headline findings

All numbers are our measurements from [`tables/tables.md`](tables/tables.md); the table ID is given for each.

- **NEWS reproduces.** Quantization brings forgotten knowledge back: NPO_KLR M1 15.9 → 34.6 under RTN-INT4
  (paper 16.6 → 34.1) — Table 1b.
- **BOOKS, base NPO_KLR (one training seed): quantization recovers forgotten text.** ΔM1 against its own BF16
  checkpoint: +60.7 (RTN), +62.0 (bitsandbytes FP4), +36.3 (AWQ), +6.1 (GPTQ) — Table 2f.
- **NPO_KLR + SURE as released: no recovery detected.** Under bnb-FP4, RTN, GPTQ and AWQ, with training seeds 42 and
  43, ΔM1 is between −0.1 and +0.2 and no configuration meets the recovery rule — Table 2f. Across all eight
  calibration × group-size configurations, M1 stays at 0.9–1.2, against 1.0 in BF16 — Table 4a.
- **The released SURE never applies its saliency mask and counts the forget gradient twice** (next section). With
  both issues fixed, still no recovery was detected (ΔM1 −1.7 to −0.8 at seed 42; −0.0 for the seed-43 bnb-FP4 run)
  — Table 2f.
- **Inconsistencies in the paper's setup:** the paper's "RTN 4-bit" is bitsandbytes FP4 in the code; its M1 values
  mix greedy and sampled decoding; and its BOOKS Tables 1 and 3 give different full-precision values for the same
  methods. See [`FINDINGS.md`](FINDINGS.md) §3.

The full analysis, including what does *not* reproduce (BOOKS full-precision rows, GA_GDR's collapse), is in
[`REPORT.md`](REPORT.md).

## Findings about the official SURE implementation

At commit `10131ae`, `SURE` (`baselines/baselines/iterative.py`) builds a saliency mask in `compute_loss` but applies
it in a method named `optimizer_step`. That hook doesn't exist in Hugging Face `Trainer` (v4.40.0 calls
`self.optimizer.step()` directly), so it is never called. We counted 0 calls in 2,765 steps. Every parameter is
updated. `compute_loss` also calls `loss_f.backward(retain_graph=True)` before the Trainer's own backward, so the
forget gradient is applied twice. In practice, "SURE as released" is NPO_KLR with a 10× learning rate, a 10× α, a
proper KL retain term, no mask and a doubled forget gradient. We report it as released (⚠️) and with both issues
fixed (✅, [`extra/fixed_unlearn.py`](extra/fixed_unlearn.py); a row-level per-step mask, not the paper's Eq. 4–6).
Neither version showed recovery under the quantizers we tested. Details: [`FINDINGS.md`](FINDINGS.md) §5–6; raw
mask-audit records in [`evidence/`](evidence/).

## Reproduce

**Prerequisites.** A [Modal](https://modal.com) account (all GPU jobs run there), a Hugging Face token with access to
`meta-llama/Llama-2-7b-hf`, and the authors' repository cloned at the pinned commit into the repository root.
Measured cost per job is in [`COMMANDS.md`](COMMANDS.md) (e.g. ≈ $3.5–5 per BOOKS unlearning run on an H200,
≈ $1 per BOOKS evaluation). Then follow `COMMANDS.md` in order: setup (§0), local CPU tests (§1), downloads (§2), the
four tasks (§3–§7).

**Rebuild the tables without a GPU.** The evaluation outputs of every run (`metrics.json` plus per-example JSON) are
the asset `outputs.zip` of the GitHub Release `v1.0`:

```bash
# from the repository root, after creating .venvs/paper (COMMANDS.md §0)
gh release download v1.0 -p outputs.zip      # or download it from the Releases page of this repository
echo "af34caf33ca8c170f1dd55421fa00da5169f81d32d25e909ce31a55c367c1fe9  outputs.zip" | sha256sum -c
unzip -q outputs.zip                         # creates ./dl_acc1 … ./dl_acc4 next to compare_results.py
PYTHONDONTWRITEBYTECODE=1 .venvs/paper/bin/python compare_results.py \
    --results dl_acc1/results dl_acc2/results dl_acc3/results dl_acc4/results dl_acc4/results_quant --out tables
git diff --stat tables/                      # empty: the rebuilt tables are byte-identical
```

`dl_acc1` … `dl_acc4` are the result folders of the four Modal accounts the runs used (`acc1`–`acc4` in the
Manifest). Run `compare_results.py` from the repository root. Checkpoints and quantized models (13.5 GB each) are not
distributed; the Manifest at the end of `tables/tables.md` records where each one was stored.

## Layout

| File / folder | Content |
|---|---|
| [`REPORT.md`](REPORT.md) | Main report: results of the four tasks, the recovery analysis and what was not run. |
| [`FINDINGS.md`](FINDINGS.md) | Full evidence for every discrepancy with the paper and every code issue (cited as F§N). |
| [`tables/`](tables/) | `tables.md` with all result tables and the Manifest, one CSV per table, `tables.json`. Generated by `compare_results.py`. |
| [`Reproducibility.md`](Reproducibility.md) | Environment, code revision, data versions, checkpoints, seeds, configurations, deviations. |
| [`COMMANDS.md`](COMMANDS.md) | Exact commands for every training, quantization and evaluation job, and for rebuilding the tables. |
| `modal_app.py`, `modal_quant.py` | Modal entry points: downloads, unlearning, evaluation, RTN / GPTQ / AWQ. |
| [`extra/`](extra/) | Wrappers around the authors' code, the fixed SURE, calibration, bootstrap CIs, CPU tests. |
| `compare_results.py` | Builds `tables/` from the saved evaluation outputs. |
| [`train_configs/`](train_configs/) | `unlearn_run.json` of every trained model (full configuration and seed). |
| [`evidence/`](evidence/) | Run records of the 7B mask audit (released vs fixed SURE) and the fixed run's per-step loss terms. |

## Labels

⚠️ = SURE as released (mask never applied). ✅ = the released SURE code with its two bugs fixed (a deviation; not
the paper's Eq. 4–6 mask). `n/a` = not run (outside this project's compute budget).

## Limitations

- The base methods (NPO_KLR, GA_GDR) were trained with one seed; NPO_KLR + SURE with two (42, 43).
- RTN / GPTQ / AWQ for the fixed SURE ✅ were run on seed 42 only.
- Task 4 used the released SURE only (plus target controls for 3 of the 8 configurations).
- The paper's SURE as written (a module-level mask fixed at the original weights) was not run.
- A larger effective training batch, a candidate explanation for the BOOKS full-precision gaps, was not tested.
- Bootstrap CIs cover evaluation sampling only, not training randomness. Evaluation sets are small (e.g. MMLU 171,
  TruthfulQA 50 examples).

The complete list is in [`REPORT.md`](REPORT.md#not-run-outside-this-projects-compute-budget) and
[`FINDINGS.md`](FINDINGS.md) §6.5.

## Citation

If you use this work, please cite the original paper and the MUSE benchmark:

```bibtex
@inproceedings{zhang2025catastrophic,
  title     = {Catastrophic Failure of {LLM} Unlearning via Quantization},
  author    = {Zhang, Zhiwei and Wang, Fali and Li, Xiaomin and Wu, Zongyu and Tang, Xianfeng and Liu, Hui and
               He, Qi and Yin, Wenpeng and Wang, Suhang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  note      = {arXiv:2410.16454}
}

@article{shi2024muse,
  title   = {{MUSE}: Machine Unlearning Six-Way Evaluation for Language Models},
  author  = {Shi, Weijia and Lee, Jaechan and Huang, Yangsibo and Malladi, Sadhika and Zhao, Jieyu and
             Holtzman, Ari and Liu, Daogao and Zettlemoyer, Luke and Smith, Noah A. and Zhang, Chiyuan},
  journal = {arXiv preprint arXiv:2407.06460},
  year    = {2024}
}
```
