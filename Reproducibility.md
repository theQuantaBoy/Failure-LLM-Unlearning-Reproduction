# Reproducibility

Reproduction and stress test of Zhang et al., *Catastrophic Failure of LLM Unlearning via Quantization*
(ICLR 2025, arXiv 2410.16454v3). This file covers the scripts, environment, code revision, data, checkpoints, seeds
and configurations. Where every result and checkpoint lives is the **Manifest** table at the end of
`tables/tables.md` (`tables/manifest.csv`). Checkpoints and quantized models (13.5 GB each) are not distributed; the
Manifest records where they were stored. The evaluation outputs are distributed as `outputs.zip` (GitHub Release, §8).

Every measured value in the tables comes from saved evaluation outputs (`results/<corpus>/<model>/<tag>/metrics.json`
plus per-example JSON files), read by `compare_results.py`. Published values appear only in rows labelled
"Reported".

---

## 1. Code

| Component | Revision | Role |
|---|---|---|
| Authors' code `zzwjames/FailureLLMUnlearning` | `10131ae25f55f1d8feb744eabb235ffc3f094b1b` | Unlearning (`baselines/`), metrics (`metrics/`, `LLama_factory/…/eval_*`), model loader (`utils.py`). Used **unmodified**; `git status` clean. |
| Our wrappers (`modal_app.py`, `modal_quant.py`, `extra/`) | the version in this folder | Call the authors' functions with fixed seeds and save every output. |
| Tables | `compare_results.py` | Builds `tables/tables.md`, one CSV per table and `tables.json`. |

Run logs record package versions but not a hash of our own source files, so the exact wrapper source used by each
training run cannot be verified retroactively. Later changes to the training files (diagnostics only) were checked
on CPU to leave training bitwise unchanged. GPU training is not bitwise deterministic: re-running a configuration
with the same seed gives a slightly different checkpoint, so differences of that size between runs are noise.

Our wrappers, one line each:

| File | What it does |
|---|---|
| `modal_app.py` | Modal entry points: `download`, `env_info`, `train`, `evaluate`, `push_ckpt` / `pull_ckpt`, smoke tests. |
| `modal_quant.py` | Modal entry points: `quantize` (RTN / GPTQ / AWQ) and `weight_index_diff`. |
| `extra/unlearn_run.py` | Runs the authors' unlearning code with a fixed training seed. Writes `unlearn_run.json` (full config, effective `TrainingArguments`, counters, RNG fingerprint). |
| `extra/fixed_unlearn.py` | **Deviation**: SURE with the saliency mask applied and the forget gradient counted once (presets `*_sure_masked`). |
| `extra/eval_run.py` | Reproduces the authors' `eval.py:eval_model` step by step with the same functions and arguments. Adds seeds, deterministic kernels and per-example outputs. |
| `extra/quantize_run.py` | llm-compressor RTN / GPTQ / AWQ (W4A16). Then dequantizes to a BF16 copy that the authors' loader can read. |
| `extra/calib.py` | Calibration sets (general text / BOOKS retain), with a 13-gram exclusion of forget and evaluation text. |
| `extra/ckpt_transfer.py` | Moves a final checkpoint between accounts through a private HF repo, with sha256 checks on push and on pull. |
| `extra/bootstrap_ci.py`, `extra/splits.py` + `extra/splits.json` | Paired bootstrap CIs; dev / held-out halves for Task 4. |
| `extra/tests/` | CPU tests. `run_cpu_e2e.py` runs the whole pipeline on a tiny model. Other tests: equivalence of `--sure-fast`, eval A/B against the authors' `eval.py`, checkpoint transfer. |

## 2. Environment

Four container images, all defined in `modal_app.py` / `modal_quant.py` and pinned:

| Image | Used for | Pins |
|---|---|---|
| paper | training, evaluation (authors' code) | Python 3.10.17, `torch==2.2.0+cu121` (CUDA 12.1), and `extra/requirements-paper.txt`: the pip section of the authors' `environment.yml` (commit `631fcec`), installed with `uv --exclude-newer 2024-10-25`. Key versions: transformers 4.40.0, accelerate 0.29.0, bitsandbytes 0.43.0, datasets 2.19.0, scikit-learn 1.4.0, numpy 1.26.0, nltk 3.9.1 (+ `punkt`, `punkt_tab`), rouge-score 0.1. `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `HF_HUB_OFFLINE=1`. The full freeze (`logs/environment_paper.txt`, 132 packages, written by `modal run modal_app.py::env_info`) is **identical in all four Modal accounts used**. |
| quant | RTN / GPTQ / AWQ | Python 3.12. `torch==2.14.0` (+cu130), `llmcompressor==0.14.0`, `compressed-tensors==0.19.0`, `transformers==5.17.0`, installed with `uv --exclude-newer 2026-10-01` (`modal_quant.py`; versions also recorded in every `quant_report`). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (allocator only). |
| download | model / data downloads | Python 3.12, `datasets==5.0.1`, `huggingface_hub`, `hf_xet` (`--exclude-newer 2026-10-01`). |
| transfer | `push_ckpt` / `pull_ckpt` | Python 3.12, `huggingface_hub==1.33.0`, `hf_xet==1.6.0`. |

GPUs:
- Training: H200.
- Evaluation: Modal "A10", served by an A10 or an A10G.
- RTN / GPTQ quantization: L40S.
- AWQ quantization: A100-80GB.
- FP32 diagnostic: L40S.

Each run's RunLog (`logs/<stamp>_<kind>_<name>.json`) records package versions and the GPU it ran on.

Local machine:
- Launching jobs: `modal==1.5.5` (any Python ≥ 3.10).
- Building the tables: `compare_results.py` needs `numpy` and `scikit-learn`. Use the local CPU copy of the paper
  environment, `.venvs/paper` (Python 3.10, `torch==2.2.0` CPU, `extra/requirements-paper.txt`; `COMMANDS.md` §0).

## 3. Data and base models

All Hugging Face artifacts are pinned to a commit (`extra/common.py`). The `download` job checks that the data
committed in the authors' repo equals the pinned MUSE revisions. `logs/download_report.json` shows
`verify_data.news.all_equal = verify_data.books.all_equal = true` in every Modal account used, with the same revisions everywhere.

| Artifact | Revision |
|---|---|
| `muse-bench/MUSE-News` (dataset) | `506bd5b150b92814d45e4404a82f120ab2d748bf` |
| `muse-bench/MUSE-Books` (dataset) | `051ba90319e920d410d87cfdbd61f25843c1b892` |
| `muse-bench/MUSE-news_target` | `a2f39769e9a0b98ec1cdd12f65e9962502208935` |
| `muse-bench/MUSE-news_retrain` | `324ef49ee0a038078aba7d8de831edf57235c9b3` |
| `muse-bench/MUSE-books_target` | `c8dd3fb23a726762ec66d277933c7cff6767f3c2` |
| `muse-bench/MUSE-books_retrain` | `1d67430e4e8bdf2a65823740e909792519175ac2` |
| Tokenizer `meta-llama/Llama-2-7b-hf` | `01c7f73d771dfac7d292323805ebc428287df4f9` |
| `Salesforce/wikitext`, `wikitext-2-raw-v1`, train, 36,718 rows (GPTQ/AWQ "general" calibration) | `b08601e04326c79dfdd32d625aee71d232d685c3` |

Training data comes from the authors' repo:
- forget set: `data/<corpus>/raw/forget.txt`
- retain set: `data/<corpus>/raw/retain1.txt` (as in the authors' scripts)

## 4. Unlearning runs (configuration and seeds)

Hyperparameters follow paper App. D, Table 4 (base methods) and Table 5 (SURE, BOOKS); `max_len` 2048 and `retain1`
come from the authors' scripts. Settings shared by all runs:
- `optim=adamw_torch`, constant LR, no warmup, `bf16=True`
- 1 GPU, no gradient accumulation (global batch = per-device batch)
- final checkpoint = last epoch

Values are copied from each checkpoint's `unlearn_run.json`; copies are in `train_configs/`.

| Run | Algo | Epochs | LR | α | Threshold | Batch | Seed | Steps | SURE path |
|---|---|---|---|---|---|---|---|---|---|
| `news_npo_klr_s42` | npo_klr | 10 | 1e-5 | 1 | 90 | 2 | 42 | 2040 | — |
| `news_ga_gdr_s42` | ga_gdr | 10 | 1e-5 | 1 | 90 | 2 | 42 | 2040 | — |
| `books_npo_klr_s42` | npo_klr | 5 | 1e-5 | 2 | 90 | 1 | 42 | 2765 | — |
| `books_ga_gdr_s42` | ga_gdr | 5 | 1e-5 | 100 | 90 | 1 | 42 | 2765 | — |
| `books_npo_klr_sure_s42` | npo_klr_sure | 5 | 1e-4 | 20 | 90 | 1 | 42 | 2765 | released ⚠️ (`--sure-fast`) |
| `books_npo_klr_sure_s43` | npo_klr_sure | 5 | 1e-4 | 20 | 90 | 1 | 43 | 2765 | released ⚠️ (`--sure-fast`) |
| `books_ga_gdr_sure_s42` | ga_gdr_sure | 5 | 1e-4 | 400 | 99 | 1 | 42 | 2765 | released ⚠️ (`--sure-fast`) |
| `books_npo_klr_sure_masked_s42` | npo_klr_sure | 5 | 1e-4 | 20 | 90 | 1 | 42 | 2765 | fixed ✅ |
| `books_npo_klr_sure_masked_s43` | npo_klr_sure | 5 | 1e-4 | 20 | 90 | 1 | 43 | 2765 | fixed ✅ |
| `books_ga_gdr_sure_masked_s42` | ga_gdr_sure | 5 | 1e-4 | 400 | 99 | 1 | 42 | 2765 | fixed ✅ |

- **Training seed**: the authors never set one, so we use 42, the `TrainingArguments` default (`extra/common.py`). The
  seed-43 repeats differ only in `seed`. `unlearn_run.json` records an RNG fingerprint at train start: identical for
  every seed-42 run (`6c93ce70…`) and different for seed 43 (`55515633…`).
- **`--sure-fast`** (released SURE only): a faster execution path for the released code. On CPU it is bitwise
  identical to the released path; on the 7B model (GPU) the difference is indistinguishable from run-to-run noise
  (`extra/tests/test_sure_equivalence.py`, `test_fast_sure_verbatim.py`).
- **Released SURE ⚠️**: the authors' code as is.
  - Its `SURE.optimizer_step` is never called: counter `sure_optimizer_step_calls = 0` in every run.
  - So the saliency mask is never applied, and the forget gradient is counted twice.
- **Fixed SURE ✅** (`extra/fixed_unlearn.py`, a **deviation**): the released code with its two bugs fixed,
  `fixes = {sure_single_grad: true, sure_mask: "step"}`.
  - It follows the released code's design: a row-level mask recomputed at every step. It is not the paper's
    Eq. 4–6 (a module-level mask computed once at θo), which was not run.
  - Gradients of rows outside the mask are zeroed; those rows are also restored bitwise after `optimizer.step()`,
    because AdamW momentum would otherwise still move them. The forget gradient is counted once.
  - Unchanged from the released code: KL on raw logits (`kl: raw`) and NPO on logits (`npo: logits`).
  - Mask size (`stats.fix_mask_stats.salient_frac`, first 50 steps): 0.100 of rows for NPO_KLR (P90), 0.010 for
    GA_GDR (P99). Mask audit: `FINDINGS.md` §6.1, raw outputs in `evidence/` (two audit RunLogs, and `loss_components.jsonl`: the per-step loss terms of
    the fixed run's audit).
- **Presets defined but not run** (`extra/fixed_unlearn.py`): `*_properkl`, `*_seqnpo`, `*_sure_single`,
  `*_sure_maskfixed`, `books_npo_klr_as1882021`. No table row comes from them.
- **Checkpoint transfer**: `books_npo_klr_s42`, `books_ga_gdr_s42` and `books_npo_klr_sure_s42` were trained in one
  account and copied to a second one for quantization and evaluation; `books_ga_gdr_sure_s42` was copied the same
  way to a third account.
  - The copy went through a private HF repo, pinned to one commit.
  - sha256 of every file was checked on push and again on pull.
  - Manifest and pull report: `dl_acc2/manifest_<run>.json` and `dl_acc3/pull_<run>.json` or
    `dl_acc4/pull_<run>.json` (on the Volume the pull report is `ckpt/<corpus>/<run>/failunl_pull_report.json`).
  - The bytes downstream jobs read are identical (Manifest, column "Checkpoint transfer").

## 5. Quantization

| Path | Library | Settings |
|---|---|---|
| **bnb4** (authors' only 4-bit path) | bitsandbytes via `utils.py:94-101` | `BitsAndBytesConfig(load_in_4bit=True)` with transformers defaults: FP4 (not INT4), block 64, compute dtype float32, no double quantization. Quantizes at load time; recorded in `meta.quantization_config`. |
| **RTN** | llm-compressor 0.14.0 | W4A16, INT4, symmetric, group 128, `lm_head` not quantized, no calibration. |
| **GPTQ** | llm-compressor 0.14.0 | W4A16, symmetric, group 128 or 32, `lm_head` not quantized, block size 128, dampening 0.01, static act-order (from the recorded recipe). |
| **AWQ** | llm-compressor 0.14.0 | W4A16, asymmetric, group 128 or 32, `lm_head` not quantized. |

Shared by RTN, GPTQ and AWQ (recipe fixed in `extra/quantize_run.py`; recorded in each `quant_report`, values below read from
the GA_GDR + SURE ⚠️ reports):
- weights INT4 per group (`strategy: group`), observer `memoryless_minmax`: each group's range is its own min and
  max, so **no clipping**. No recorded recipe contains a clipping or clip-search parameter.
- activations not quantized (`input_activations: null`): A16, the BF16 load dtype, the same in every run.
- GPTQ act-order `static`; RTN and AWQ none. AWQ stores an INT8 zero point (asymmetric); RTN and GPTQ have none.

Calibration (GPTQ and AWQ):
- 128 samples × 2048 tokens, `calib_seed = 0`.
- **general** = WikiText-2 train. **books_retain** = MUSE BOOKS `retain1.txt` + `retain2.txt`.
- Before tokenization, every paragraph that shares a word 13-gram with the corpus's forget set or evaluation sets is
  removed (`extra/calib.py`). BOOKS retain keeps 2,204 of 3,023 paragraphs; general text keeps all 23,767.
- Windows are drawn with `random.Random(seed).sample`, and every sample's sha1 is recorded.

INT4 models are evaluated as **dequantized BF16 copies** (`quant/<model>/<tag>/dq`):
- the compressed checkpoint is reloaded with `CompressedTensorsConfig(dequantize=True)` in FP32;
- the values are rounded once to BF16;
- this keeps the authors' BF16 loader unchanged.

The BF16 rounding is not exact: a dequantized INT4 value (scale × integer) need not be representable in BF16, so the
copy differs from the INT4 model by at most one BF16 rounding (relative error ≤ 2⁻⁸ per weight). The `quant_report`
records this as `dequant_bf16_inexact`.

Each `quant_report` (copied into `metrics.json`) records the recipe, calibration statistics, versions, device and peak
memory.

## 6. Evaluation

`extra/eval_run.py` makes the same calls as the authors' `eval.py:eval_model`, in the same order:
1. Load the model with `utils.load_model` (BF16, or bnb4).
2. Utility metrics: MMLU (batch 1), TruthfulQA (batch 4), TriviaQA (batch 16), Fluency (batch 8).
3. MUSE metrics: VerbMem, PrivLeak (Min-40%), KnowMem-forget, KnowMem-retain.

The tokenizer is the authors' (`meta-llama/Llama-2-7b-hf`).

Additions (none changes what is computed for a given model):
- **E1** `transformers.set_seed(0)` before every metric (`seed_mode = per_metric`).
- **E2** VerbMem `sample` (primary, the authors' call: `do_sample=True`, T = 0.9). Example *i* is generated right after
  `set_seed(0 + i)`, so a sample does not depend on which other examples are evaluated. VerbMem `greedy` (extra column)
  is the same call with `do_sample=False`, as in upstream MUSE.
- **E3** `torch.use_deterministic_algorithms(True, warn_only=True)`, `cudnn.benchmark = False`.
- **E4** Every per-example record is saved with its index in the source file and a sha1 of its text.

Evaluation set sizes (the authors' sets, all examples used): VerbMem, KnowMem-forget and KnowMem-retain 100 each;
PrivLeak 100 forget, 100 retain, 100 holdout; MMLU 171; TriviaQA 100; TruthfulQA 50; fluency 50. NEWS runs use only
the four MUSE metrics.

PrivLeak: (AUC_unl − AUC_retrain) / AUC_retrain × 100. The retrained AUC constants are 0.4772 (NEWS) and 0.5393
(BOOKS) (authors' `constants.py:64,165`). As a check, the NEWS retrained model was also evaluated: its own M3 is −0.2 (Table 1a).
**Deviation (one row):** `fp32diag_knowmem` evaluates the BOOKS target's KnowMem-forget with an FP32 load.

Known evaluation non-determinism:
- **A10 vs A10G.** The same bnb4 model evaluated on both gave identical greedy VerbMem, M2, PrivLeak AUC, Gen and Fac.
  Sampled M1, M4, Tru and Flu differed slightly (e.g. M4 47.6 vs 47.3).
- **Bootstrap CIs** (Tables 2f, 2g) cover example sampling only, not training randomness (see the seed-43 repeats
  for that).

## 7. Seeds (summary)

| Seed | Value |
|---|---|
| Training | 42 (all runs); 43 for the NPO_KLR + SURE repeats (released and fixed) |
| Evaluation | 0, reset before every metric; VerbMem example *i*: 0 + *i* |
| Calibration (GPTQ / AWQ) | 0 |
| Bootstrap | 0 (`numpy.random.default_rng(0)`, B = 10,000 paired resamples, `extra/bootstrap_ci.py`) |
| Dev / held-out split | 0 (`extra/splits.json`, 50 / 50) |

## 8. How to reproduce each table

Exact commands, in order: `COMMANDS.md`. All jobs use `modal run --detach`; each one writes a RunLog under `logs/`.

| Tables | Steps (COMMANDS.md) |
|---|---|
| 1a–1d (NEWS) | §0 setup → §2 download → §3 Task 1 (train 2 runs, then evaluate BF16 / bnb4 / RTN, plus per-epoch checkpoints). |
| 2a–2i (BOOKS, Tasks 2 + 3) | §4 Task 2 (train 4 runs; BF16 / bnb4 / RTN) → §5 Task 3 (GPTQ / AWQ g128 general) → §7 seed 43. Fixed SURE and its mask audit: §4. FP32 diagnostic: §4. |
| 4a–4b (Task 4) | §6 (GPTQ / AWQ × {g32, g128} × {general, books_retain}). |
| Manifest | built from the `meta` of every `metrics.json` and the transfer reports. |

Build the tables locally (no GPU, about 1 min) from the evaluation outputs of the GitHub Release (`outputs.zip`,
unzipped in the repository root; see `README.md` or `COMMANDS.md` §9):

```bash
PYTHONDONTWRITEBYTECODE=1 .venvs/paper/bin/python compare_results.py --results dl_acc1/results dl_acc2/results dl_acc3/results dl_acc4/results \
    dl_acc4/results_quant --out tables
.venvs/paper/bin/python compare_results.py --show t23_main task4_main --from tables      # print saved tables; --list-tables
```

Here `dl_accN/results` is a local copy of `failunl-runs:/results` from the N-th Modal account we used (`acc1`–`acc4`
in the Manifest). If you run everything in one account, a single `dl_acc1/results` is enough.

Cost per job, measured: training ≈ $3.5–5.5 (H200), BOOKS evaluation ≈ $0.9–1.2 (A10), GPTQ ≈ $0.4, AWQ ≈ $1.1,
RTN ≈ $0.1 (`COMMANDS.md`, "Measured costs").

## 9. Deviations from the paper's setup (each labelled in the tables; reasons in `FINDINGS.md`)

1. BF16 for training and evaluation (the authors' loader).
2. INT4 RTN via llm-compressor next to the authors' bitsandbytes-FP4 path. The paper calls its 4-bit models RTN;
   the code's only 4-bit path is FP4.
3. INT4 models evaluated as dequantized BF16 copies.
4. VerbMem sampled with a fixed per-example seed; greedy reported in parentheses.
5. Fixed SURE ✅ (released code, two bugs fixed; not the paper's fixed module-level mask) next to the released SURE ⚠️.
6. Training seed 42 set explicitly (the authors set none); one repeat with seed 43.
7. FP32 KnowMem diagnostic (one row, Table 2i).
8. Runs not done (outside this project's compute budget) are marked `n/a` in the tables.

## 10. Verifying the authors' code

```bash
git -C FailureLLMUnlearning rev-parse HEAD                # 10131ae25f55f1d8feb744eabb235ffc3f094b1b
git -C FailureLLMUnlearning status --short                # empty
```