"""
modal_quant.py — INT4 RTN / GPTQ / AWQ with llm-compressor 0.14.0 in its own image (it needs torch 2.10–2.14 and
transformers 5.15–5.17, which conflict with the paper image). Uses the same Volumes as modal_app.py.
The work is done by extra/quantize_run.py (settings, calibration, dequantize-to-BF16 numerics are documented there).

    modal run --detach modal_quant.py::quantize --source hf:books_target --corpus books --method gptq
    modal run --detach modal_quant.py::quantize --source ckpt/books/books_npo_klr_sure_s42 --corpus books \
        --method awq --group-size 32 --calib books_retain          # AWQ defaults to A100-80GB (see AWQ_GPU)
    modal run --detach modal_quant.py::quantize --source hf:books_target --corpus books --method gptq --gpu H100
    modal run --detach modal_quant.py::quantize --source ckpt/news/news_ga_gdr_s42 --corpus news --method rtn

Output: /vol/runs/quant/<source-tag>/<method>_g<gs>_<calib>[_n<n>_l<len>][_s<seed>]/{compressed/, dq/,
quant_report.json}. Evaluate the dq/ copy with modal_app.py::evaluate --model quant/.../dq --quant none
(eval_job picks up quant_report.json automatically).
"""

import shlex
import sys
from pathlib import Path

import modal

from modal_app import (COMMON_ENV, HF_DIR, HF_SECRET, HERE, LOCAL_REPO, REMOTE_PROJ, REMOTE_REPO, REPO_IGNORE,
                       RUNS_DIR, TOKENIZER_KEY, VOLUMES, runs_vol)

QUANT_GPU = "L40S"  # 48 GB; llm-compressor calibrates layer by layer (RTN, GPTQ)
# AWQ at 128 x 2048 OOMs on L40S (42.4 GiB allocated, 2026-10-05): AWQModifier._run_samples (awq/base.py:596-605)
# keeps every sample's full (attn_output, attn_weights) tuple until the list is built, and the eager attention that
# llm-compressor forces during calibration (utils/helpers.py:147-162) returns 32x2048x2048 BF16 weights = 256 MiB per
# sample -> ~32 GiB for 128 samples, on top of ~12.5 GiB of AWQ input caches (offload_device=None). Peak ~50 GiB.
AWQ_GPU = "A100-80GB"
# Allocator setting only (fewer fragmentation OOMs); does not change any computed value.
ALLOC_CONF = "expandable_segments:True"
LLMC_VERSION = "0.14.0"
RESOLVE_AS_OF = "2026-10-01"

app = modal.App("failunl-quant")

quant_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .run_commands(
        f"uv pip install --system --exclude-newer {RESOLVE_AS_OF} torch==2.14.0 llmcompressor=={LLMC_VERSION} "
        "compressed-tensors==0.19.0 transformers==5.17.0"
    )
    .env({**COMMON_ENV, "HF_HUB_OFFLINE": "1"})
    .add_local_dir(LOCAL_REPO, REMOTE_REPO, ignore=REPO_IGNORE)
    .add_local_dir(HERE / "extra", f"{REMOTE_PROJ}/extra", ignore=["**/__pycache__", "tests/_out*"])
    .add_local_python_source("modal_app")  # shared helpers (snapshot, resolve_model, run_logged)
)


def _tag(source: str) -> str:
    return source[3:] if source.startswith("hf:") else Path(source).name


@app.function(image=quant_image, gpu=QUANT_GPU, cpu=4.0, memory=65536, timeout=4 * 3600, volumes=VOLUMES)
def quantize_job(source: str, corpus: str, method: str, group_size: int = 128, calib: str = "general",
                 n_samples: int = 128, seq_len: int = 2048, calib_seed: int = 0, symmetric: str = "auto",
                 out_rel: str = "") -> int:
    import os

    import modal_app  # helpers only (snapshot / resolve_model / run_logged)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ALLOC_CONF  # inherited by the run_logged subprocess
    src = modal_app.resolve_model(source)
    calib_part = "nocalib" if method == "rtn" else calib
    name = f"{method}_g{group_size}_{calib_part}"
    if method != "rtn" and (n_samples, seq_len) != (128, 2048):
        name += f"_n{n_samples}_l{seq_len}"
    if method != "rtn" and calib_seed != 0:
        name += f"_s{calib_seed}"
    if symmetric != "auto":
        name += f"_sym{symmetric}"
    out_rel = out_rel or f"quant/{_tag(source)}/{name}"
    argv = [sys.executable, "-m", "extra.quantize_run", "--src", src, "--tokenizer_dir",
            modal_app.snapshot(TOKENIZER_KEY), "--corpus", corpus, "--method", method, "--group_size",
            str(group_size), "--symmetric", symmetric, "--calib", calib, "--n_samples", str(n_samples),
            "--seq_len", str(seq_len), "--calib_seed", str(calib_seed), "--wikitext_dir",
            f"{HF_DIR}/calib/wikitext2_train", "--out_dir", f"{RUNS_DIR}/{out_rel}", "--logs_dir",
            f"{RUNS_DIR}/logs"]
    rc = modal_app.run_logged(argv, f"quant_{_tag(source)}_{name}")
    runs_vol.commit()
    if rc != 0:
        raise RuntimeError(f"quantization failed with exit code {rc}")
    return rc


@app.function(image=quant_image, cpu=8.0, memory=32768, timeout=75 * 60, volumes=VOLUMES)
def weight_index_diff_job(target: str, unlearned: list, group_size: int = 128, out_rel: str = "",
                          verify_src: str = "", verify_compressed: str = "", verify_only: bool = False,
                          verify_layers: str = "") -> int:
    """CPU only. INT4 RTN index differences target vs each unlearned checkpoint (extra/weight_index_diff.py).
    First (if given) compares the re-implemented RTN (quotient modes bf16 and fp32) with a real llm-compressor RTN
    output on this Volume; the per-tensor report goes to <out_rel>/verify.json and the run log BEFORE any decision.
    The diff runs only if the pre-registered acceptance rule passed, with the verified quotient mode.
    verify_only=True stops after the report; verify_layers="0,15,31" limits it to those layers."""
    import modal_app

    out_rel = out_rel or f"analysis/index_diff_rtn_g{group_size}"
    vjson = f"{RUNS_DIR}/{out_rel}/verify.json"
    if verify_src and verify_compressed:
        argv = [sys.executable, "-m", "extra.weight_index_diff", "--verify", modal_app.resolve_model(verify_src),
                f"{RUNS_DIR}/{verify_compressed}", "--group_size", str(group_size), "--verify_out", vjson,
                "--logs_dir", f"{RUNS_DIR}/logs"]
        if verify_layers:
            argv += ["--verify_layers", verify_layers]
        rc = modal_app.run_logged(argv, f"index_diff_verify_g{group_size}")
        runs_vol.commit()
        if verify_only:
            return rc
        if rc != 0:
            raise RuntimeError(f"verification did not pass the acceptance rule (report: {vjson}): not running the diff")
    elif verify_only:
        raise ValueError("verify_only needs verify_src and verify_compressed")
    argv = [sys.executable, "-m", "extra.weight_index_diff", "--target", modal_app.resolve_model(target),
            "--unlearned", *[f"{Path(u).name}={modal_app.resolve_model(u)}" for u in unlearned],
            "--group_size", str(group_size), "--out", f"{RUNS_DIR}/{out_rel}", "--logs_dir", f"{RUNS_DIR}/logs"]
    argv += ["--verification", vjson] if (verify_src and verify_compressed) else ["--quotient", "bf16"]
    rc = modal_app.run_logged(argv, f"index_diff_rtn_g{group_size}")
    runs_vol.commit()
    if rc != 0:
        raise RuntimeError(f"weight_index_diff failed with exit code {rc}")
    return rc


@app.local_entrypoint()
def weight_index_diff(unlearned: str = "", target: str = "hf:books_target", group_size: int = 128,
                      verify_src: str = "ckpt/books/books_npo_klr_s42",
                      verify_compressed: str = "quant/books_npo_klr_s42/rtn_g128_nocalib/compressed",
                      verify_only: bool = False, verify_layers: str = ""):
    """--unlearned: comma-separated runs-Volume paths, e.g. ckpt/books/books_npo_klr_s42,ckpt/books/books_ga_gdr_s42.
    --verify-src/--verify-compressed: a checkpoint and its RTN compressed/ output on the same Volume ('' = skip; the
    diff is then labelled unverified). --verify-only: write analysis/index_diff_rtn_g<gs>/verify.json and stop.
    --verify-layers 0,15,31: verify only those layers."""
    if not verify_only and not unlearned:
        raise SystemExit("--unlearned is required unless --verify-only")
    call = weight_index_diff_job.spawn(target, [u for u in unlearned.split(",") if u], group_size, "",
                                       verify_src, verify_compressed, verify_only, verify_layers)
    from modal_app import _spawned

    _spawned(call, f"weight_index_diff {target} vs {unlearned or '(verify only)'} (CPU)")


@app.local_entrypoint()
def quantize(source: str, corpus: str, method: str, group_size: int = 128, calib: str = "general",
             n_samples: int = 128, seq_len: int = 2048, calib_seed: int = 0, symmetric: str = "auto",
             gpu: str = ""):
    gpu = gpu or (AWQ_GPU if method == "awq" else QUANT_GPU)
    call = quantize_job.with_options(gpu=gpu).spawn(source, corpus, method, group_size, calib, n_samples, seq_len,
                                                    calib_seed, symmetric)
    from modal_app import _spawned

    _spawned(call, f"quantize {method} g{group_size} {calib} of {source} on {gpu}")
