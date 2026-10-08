# Commands

How to run every step on [Modal](https://modal.com), in order. Run all commands from the project root.

- **Detach every job.** `modal run --detach …` spawns the job and returns. The job keeps running if the connection
  drops.
- **Logs.** Each job writes a RunLog `logs/<stamp>_<kind>_<name>.json` (status, versions, GPU, minutes) to the Volume
  `failunl-runs`.
- **One Modal account is enough.** If you split jobs across several accounts (we used four, `acc1`–`acc4` in the
  Manifest), prefix commands with `MODAL_PROFILE=<profile>`. Each account has its own Volumes; §8 explains how
  checkpoints move between accounts.

Measured cost per job (Modal prices of 2026-10):

| Job | GPU | Time | Cost |
|---|---|---|---|
| BOOKS unlearning (5 epochs) | H200 | 0.6–1 h | ≈ $3.5–5 |
| NEWS unlearning (10 epochs) | H200 | ≈ 1 h | ≈ $5 |
| BOOKS evaluation, 8 metrics | A10 | 35–50 min | ≈ $1 |
| RTN / GPTQ / AWQ (128 × 2048 calibration) | L40S / L40S / A100-80GB | 3 / 6 / 19 min | ≈ $0.1 / 0.4 / 1.1 |

These commands describe the full design. Not every combination was run (e.g. RTN / GPTQ / AWQ only for seed 42;
Task 4 only for the released SURE and the target). Which runs exist, and where, is in the Manifest
(`tables/tables.md`); missing cells are `n/a` in the tables.

Naming used by `compare_results.py`:
- checkpoints: `ckpt/<corpus>/<preset>_s<seed>`
- quantized models: `quant/<corpus>_<model>/<method>_g<gs>_<calib|nocalib>/dq`
- results: `results/<corpus>/<model>/<tag>`

## 0. Setup (once)

```bash
pip install modal==1.5.5 && modal setup
modal secret create huggingface HF_TOKEN=hf_...      # an HF account with access to meta-llama/Llama-2-7b-hf
git clone https://github.com/zzwjames/FailureLLMUnlearning && git -C FailureLLMUnlearning checkout 10131ae
export PYTHONDONTWRITEBYTECODE=1                     # the authors' repo tracks .pyc files: never rewrite them

# local CPU environments (tests, tables)
pip install --user uv
uv venv --python 3.10 .venvs/paper
uv pip install --python .venvs/paper/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.2.0
uv pip install --python .venvs/paper/bin/python --exclude-newer 2024-10-25 -r extra/requirements-paper.txt
uv venv --python 3.12 .venvs/quant
uv pip install --python .venvs/quant/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
uv pip install --python .venvs/quant/bin/python --exclude-newer 2026-10-01 llmcompressor==0.14.0 \
    compressed-tensors==0.19.0 transformers==5.17.0
```

## 1. Local tests (CPU, free)

```bash
python3 extra/tests/run_cpu_e2e.py                               # whole pipeline on a tiny model (~20–40 min)
.venvs/quant/bin/python -m extra.tests.test_ckpt_transfer        # checkpoint transfer against a fake Hub (~1 s)
git -C FailureLLMUnlearning status                               # must be clean
```

## 2. Downloads and environment (once per Modal account)

```bash
# pinned models, tokenizer, WikiText-2, and a check that the repo's data equals the pinned MUSE revisions
modal run --detach modal_app.py::download --what models,tokenizer,wikitext,verify_data
modal run modal_app.py::env_info        # writes logs/environment_paper.txt (pip freeze + GPU)
```

Success means `logs/download_report.json` shows `verify_data.news.all_equal` and `verify_data.books.all_equal`
`true`.

## 3. Task 1 — NEWS

```bash
M14='--extra-args=--metrics verbmem_f privleak knowmem_f knowmem_r'
# target: BF16, the authors' bnb-FP4 path, INT4 RTN
modal run --detach modal_app.py::evaluate --model hf:news_target --corpus news "$M14"
modal run --detach modal_app.py::evaluate --model hf:news_target --corpus news --quant bnb4 "$M14"
modal run --detach modal_quant.py::quantize --source hf:news_target --corpus news --method rtn
modal run --detach modal_app.py::evaluate --model quant/news_target/rtn_g128_nocalib/dq --corpus news "$M14"
# retrained reference (PrivLeak)
modal run --detach modal_app.py::evaluate --model hf:news_retrain --corpus news --extra-args="--metrics privleak"
# unlearning, then BF16 / bnb4 / RTN for each
for m in npo_klr ga_gdr; do modal run --detach modal_app.py::train --preset news_${m} --seed 42; done
for m in npo_klr ga_gdr; do                                  # after training has finished
  modal run --detach modal_app.py::evaluate --model ckpt/news/news_${m}_s42 --corpus news "$M14"
  modal run --detach modal_app.py::evaluate --model ckpt/news/news_${m}_s42 --corpus news --quant bnb4 "$M14"
  modal run --detach modal_quant.py::quantize --source ckpt/news/news_${m}_s42 --corpus news --method rtn
done
for m in npo_klr ga_gdr; do                                  # after the RTN jobs
  modal run --detach modal_app.py::evaluate --model quant/news_${m}_s42/rtn_g128_nocalib/dq --corpus news "$M14"
done
# optional: an intermediate epoch (Table 1d)
modal run --detach modal_app.py::evaluate --model ckpt/news/news_ga_gdr_s42/checkpoint-1020 --corpus news "$M14"
```

## 4. Task 2 — BOOKS (Table 3 rows)

```bash
modal run --detach modal_app.py::evaluate --model hf:books_target --corpus books
modal run --detach modal_app.py::evaluate --model hf:books_target --corpus books --quant bnb4
modal run --detach modal_quant.py::quantize --source hf:books_target --corpus books --method rtn
modal run --detach modal_app.py::evaluate --model quant/books_target/rtn_g128_nocalib/dq --corpus books
modal run --detach modal_app.py::evaluate --model hf:books_retrain --corpus books --extra-args="--metrics verbmem_f privleak knowmem_f knowmem_r"

# unlearning. --sure-fast = verified-equivalent fast path of the released SURE (extra/tests/test_sure_equivalence.py)
modal run --detach modal_app.py::train --preset books_npo_klr --seed 42
modal run --detach modal_app.py::train --preset books_ga_gdr --seed 42
modal run --detach modal_app.py::train --preset books_npo_klr_sure --seed 42 --sure-fast
modal run --detach modal_app.py::train --preset books_ga_gdr_sure --seed 42 --sure-fast
# fixed SURE (deviation; extra/fixed_unlearn.py)
modal run --detach modal_app.py::train --preset books_npo_klr_sure_masked --seed 42
modal run --detach modal_app.py::train --preset books_ga_gdr_sure_masked --seed 42
# mask audit, 20 steps on the 7B model (released: --preset books_npo_klr_sure; fixed: run A only)
modal run --detach modal_app.py::smoke_sure --preset books_npo_klr_sure_masked --steps 20 --runs A
# result: logs/<stamp>_test_sure_equivalence_<preset>.json → verdict.D2_rows_changed_outside_mask_total

# evaluate each checkpoint C = ckpt/books/books_<preset>_s42: BF16, bnb4, RTN
modal run --detach modal_app.py::evaluate --model $C --corpus books
modal run --detach modal_app.py::evaluate --model $C --corpus books --quant bnb4
modal run --detach modal_quant.py::quantize --source $C --corpus books --method rtn
modal run --detach modal_app.py::evaluate --model quant/books_<preset>_s42/rtn_g128_nocalib/dq --corpus books

# diagnostic, deviation (Table 2i): target KnowMem-forget with an FP32 load
modal run --detach modal_app.py::evaluate --model hf:books_target --corpus books --extra-args="--diag_fp32_knowmem --metrics knowmem_f"
```

## 5. Task 3 — GPTQ / AWQ (g128, general text, 128 × 2048, calibration seed 0)

```bash
for src in hf:books_target ckpt/books/books_npo_klr_s42 ckpt/books/books_ga_gdr_s42 \
           ckpt/books/books_npo_klr_sure_s42 ckpt/books/books_npo_klr_sure_masked_s42 \
           ckpt/books/books_ga_gdr_sure_s42; do
  modal run --detach modal_quant.py::quantize --source $src --corpus books --method gptq
  modal run --detach modal_quant.py::quantize --source $src --corpus books --method awq     # A100-80GB
done
# then evaluate each quant/<model>/{gptq,awq}_g128_general/dq:
modal run --detach modal_app.py::evaluate --model quant/books_target/gptq_g128_general/dq --corpus books
```

## 6. Task 4 — calibration set × group size

g128 / general is reused from Task 3. Run the same three extra configurations for each SURE checkpoint and for the
target:

```bash
for src in ckpt/books/books_npo_klr_sure_s42 hf:books_target; do
  for q in gptq awq; do
    modal run --detach modal_quant.py::quantize --source $src --corpus books --method $q --group-size 32
    modal run --detach modal_quant.py::quantize --source $src --corpus books --method $q --group-size 32 --calib books_retain
    modal run --detach modal_quant.py::quantize --source $src --corpus books --method $q --group-size 128 --calib books_retain
  done
done
# evaluate each quant/<model>/<q>_g<gs>_<calib>/dq as in §5
```

Every evaluation runs on the full sets. `compare_results.py` also reports the dev and held-out halves
(`extra/splits.json`, Table 4b).

## 7. Seed repeats

```bash
modal run --detach modal_app.py::train --preset books_npo_klr_sure --seed 43 --sure-fast
modal run --detach modal_app.py::train --preset books_npo_klr_sure_masked --seed 43
# then evaluate / quantize ckpt/books/books_npo_klr_sure{,_masked}_s43 as in §4–§5
```

## 8. Optional: moving a checkpoint to another Modal account

Only needed if you split jobs across accounts. A checkpoint goes through a **private** HF repo, pinned to one commit and checked with sha256 on both sides
(`extra/ckpt_transfer.py`).
- **Tokens:** the source account's Secret needs a token that can write to your own namespace. The target account
  needs only a read token.
- **What moves:** the final weights, config, tokenizer and `unlearn_run.json`.

```bash
MODAL_PROFILE=<source> modal run --detach modal_app.py::push_ckpt --run books_npo_klr_sure_s42
mkdir -p transfer/
MODAL_PROFILE=<source> modal volume get failunl-runs ckpt/books/books_npo_klr_sure_s42/failunl_transfer_manifest.json transfer/
HF_COMMIT=$(python3 -c "import json; m=json.load(open('transfer/failunl_transfer_manifest.json')); assert m['status']=='ok'; print(m['hf_commit'])")
MODAL_PROFILE=<target> modal run --detach modal_app.py::pull_ckpt --run books_npo_klr_sure_s42 --revision "$HF_COMMIT"
# check ckpt/books/<run>/failunl_pull_report.json on the target: "status": "ok", same hf_commit
```

Run each (corpus, model, tag) in **one** account only.

## 9. Build the tables (local, free)

**From the published outputs** (no Modal account needed). The evaluation outputs (`metrics.json` and per-example JSON
of every run, 549 files) are the asset `outputs.zip` of the GitHub Release `v1.0`. Unzipped in the repository root,
it gives `dl_acc1/` … `dl_acc4/` next to `compare_results.py` (one folder per Modal account we used; `acc1`–`acc4` in
the Manifest).

```bash
# from the repository root
gh release download v1.0 -p outputs.zip      # or download it from the Releases page of this repository
echo "af34caf33ca8c170f1dd55421fa00da5169f81d32d25e909ce31a55c367c1fe9  outputs.zip" | sha256sum -c
unzip -q outputs.zip                         # creates ./dl_acc1 … ./dl_acc4
PYTHONDONTWRITEBYTECODE=1 .venvs/paper/bin/python compare_results.py \
    --results dl_acc1/results dl_acc2/results dl_acc3/results dl_acc4/results dl_acc4/results_quant --out tables
git diff --stat tables/                      # empty: the rebuilt tables are byte-identical to the committed ones
.venvs/paper/bin/python compare_results.py --list-tables
.venvs/paper/bin/python compare_results.py --show t23_main manifest --from tables     # print saved tables, no rebuild
```

Run `compare_results.py` from the repository root, with the `dl_accN/` folders directly there: the account label
(`accN`) and the checkpoint-transfer reports are found from the first component of each `--results` path.

**From your own runs.** Download `results/` (JSON only) from each Volume you used and pass every copy:

```bash
mkdir -p dl_acc1/
modal volume get failunl-runs results dl_acc1/              # prefix MODAL_PROFILE=<profile> per account if several
PYTHONDONTWRITEBYTECODE=1 .venvs/paper/bin/python compare_results.py --results dl_acc1/results --out tables
```

Never download `ckpt/` or `quant/` (13.5 GB per model). Checkpoints and quantized models are not distributed with
this repository; their Volume paths are listed in the Manifest (`tables/tables.md`).
