"""
extra/quantize_run.py — INT4 weight-only quantization of one BF16/FP32 checkpoint with llm-compressor
(runs in the quantization image: llmcompressor 0.14.0, compressed-tensors 0.19.0, transformers 5.x).

Methods (official llm-compressor modifiers; recipes follow the 0.14.0 examples
examples/quantization_w4a16/llama3_example.py and examples/awq/llama_example.py, with config_groups instead of a
preset name so that group size / symmetry can be set):
  rtn   QuantizationModifier, no calibration data (round-to-nearest with the weight observer).
  gptq  GPTQModifier (module defaults, recorded in the report).
  awq   [AWQModifier(duo_scaling="both"), QuantizationModifier] as in the AWQ example.
All: weights INT4, strategy "group", group size 32 or 128, lm_head ignored (as in both examples and as in the
authors' bitsandbytes path), activations not quantized (W4A16). Symmetry default: rtn/gptq symmetric (= preset
W4A16), awq asymmetric (= preset W4A16_ASYM); override with --symmetric.

Input precision: the source is loaded in BF16 (FP32 MUSE targets are cast exactly as the paper image does for its
BF16 evaluation and for bitsandbytes), then quantized.

Outputs in --out_dir:
  compressed/          compressed-tensors checkpoint written by llm-compressor (save_compressed=True)
  dq/                  the same INT4 weights dequantized for the paper image: compressed/ is reloaded with
                       CompressedTensorsConfig(dequantize=True) in FP32 (scale*(q-zero_point) computed in FP32,
                       exact), each tensor is rounded once to BF16 and written as model.safetensors next to the
                       SOURCE config.json (transformers 4.40 must be able to read it). Elements whose dequantized
                       value is not representable in BF16 are counted in the report (dequant_bf16_inexact).
  quant_report.json    method, versions, recipe repr, full quantization_config (observer, symmetry, group size,
                       ...), calibration report (sources, filter stats, sample hashes), sanity checks, timings.
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from extra import calib
from extra.common import REPO_DIR, RunLog, utc_now, write_json


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="HF checkpoint directory (local)")
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--corpus", required=True, choices=["news", "books"], help="for the calibration exclusion set")
    p.add_argument("--method", required=True, choices=["rtn", "gptq", "awq"])
    p.add_argument("--group_size", type=int, default=128, choices=[32, 128])
    p.add_argument("--symmetric", choices=["auto", "true", "false"], default="auto")
    p.add_argument("--calib", choices=["general", "books_retain"], default="general")
    p.add_argument("--wikitext_dir", help="local dir from datasets save_to_disk (modal_app.py::download)")
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--calib_seed", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--logs_dir", required=True)
    p.add_argument("--keep_compressed", action="store_true", default=True)
    return p


def make_recipe(method, group_size, symmetric):
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    groups = {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "strategy": "group",
                "group_size": group_size,
                "symmetric": symmetric,
                "dynamic": False,
            },
        }
    }
    if method == "rtn":
        return QuantizationModifier(config_groups=groups, ignore=["lm_head"])
    if method == "gptq":
        return GPTQModifier(config_groups=groups, ignore=["lm_head"])
    if method == "awq":
        return [AWQModifier(duo_scaling="both"), QuantizationModifier(config_groups=groups, ignore=["lm_head"])]
    raise ValueError(method)


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    sym = {"auto": a.method != "awq", "true": True, "false": False}[a.symmetric]
    cfg = vars(a).copy()
    cfg["symmetric_effective"] = sym
    name = Path(a.out_dir).name
    runlog = RunLog(a.logs_dir, "quant", name, cfg)
    try:
        report = run(a, sym)
        runlog.finish(0, report=report)
        return 0
    except BaseException:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        runlog.finish(1, error=tb)
        return 1


def _calibration(a, tok):
    from datasets import Dataset, load_from_disk

    rows = None
    if a.calib == "general":
        if not a.wikitext_dir:
            raise SystemExit("--wikitext_dir is required for --calib general")
        rows = load_from_disk(a.wikitext_dir)["text"]
    samples, rep = calib.build(a.calib, tok, REPO_DIR, a.corpus, a.n_samples, a.seq_len, a.calib_seed, rows)
    ds = Dataset.from_dict({"input_ids": samples, "attention_mask": [[1] * len(s) for s in samples]})
    return ds, rep


def run(a, sym) -> dict:
    import compressed_tensors
    import llmcompressor
    import torch
    import transformers
    from llmcompressor import oneshot
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig

    out = Path(a.out_dir)
    if out.exists():
        shutil.rmtree(out)
    out_c, out_dq = out / "compressed", out / "dq"
    src = Path(a.src)
    report = {"method": a.method, "src": str(src), "group_size": a.group_size, "symmetric": sym,
              "weights_bits": 4, "activations": "not quantized (W4A16)", "ignore": ["lm_head"],
              "load_dtype": "bfloat16", "start": utc_now(),
              "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                           "llmcompressor": llmcompressor.__version__,
                           "compressed_tensors": compressed_tensors.__version__},
              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}

    torch.manual_seed(a.calib_seed)
    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    recipe = make_recipe(a.method, a.group_size, sym)
    report["recipe"] = repr(recipe)

    kwargs = {}
    if a.method != "rtn":
        ds, crep = _calibration(a, tok)
        report["calibration"] = crep
        kwargs = dict(dataset=ds, max_seq_length=a.seq_len, num_calibration_samples=a.n_samples,
                      shuffle_calibration_samples=False)
    else:
        report["calibration"] = None
    report["oneshot_kwargs"] = {k: (v if k != "dataset" else "Dataset(see calibration)") for k, v in kwargs.items()}

    if not torch.cuda.is_available():
        # CPU tests only: llm-compressor's AWQ cache calls Tensor.pin_memory(), which raises without an accelerator
        # (llmcompressor/pipelines/cache.py::_pin_intermediate). Pinning only speeds up host->GPU copies.
        from llmcompressor.pipelines.cache import IntermediatesCache

        IntermediatesCache._pin_intermediate = classmethod(lambda cls, intermediate: None)
        report["cpu_test_shim"] = "IntermediatesCache._pin_intermediate disabled (no accelerator)"

    t0 = time.time()
    oneshot(model=model, processor=tok, recipe=recipe, **kwargs)
    report["quantize_minutes"] = round((time.time() - t0) / 60, 2)
    if torch.cuda.is_available():
        report["peak_gpu_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    model.save_pretrained(out_c, save_compressed=True)
    tok.save_pretrained(out_c)
    report["quantization_config"] = json.loads((out_c / "config.json").read_text()).get("quantization_config")
    del model
    gc.collect()

    # ── dequantize in FP32, round once to BF16 ──
    t0 = time.time()
    dq = AutoModelForCausalLM.from_pretrained(out_c, dtype=torch.float32,
                                              quantization_config=CompressedTensorsConfig(dequantize=True))
    ref = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    ref_sd = ref.state_dict()
    dq_sd = dq.state_dict()
    missing = [k for k in ref_sd if k not in dq_sd]
    if missing:
        raise RuntimeError(f"dequantized model lacks {len(missing)} tensors, e.g. {missing[:5]}")
    out_sd, inexact, total, changed, groups_checked, groups_bad = {}, 0, 0, 0, 0, 0
    max_rel_err = 0.0  # one BF16 rounding of an exact FP32 dequantized value: bounded by 2^-8 (7 stored bits)
    other_changed = []  # AWQ folds its scales into the RMSNorm weights, so these may change by design
    for k, v_ref in ref_sd.items():
        v32 = dq_sd[k].float()
        v16 = v32.to(torch.bfloat16)
        if v_ref.dim() == 2 and k.endswith(".weight") and "lm_head" not in k and "embed_tokens" not in k:
            inexact += int((v16.float() != v32).sum())
            nz = v32 != 0
            if nz.any():
                max_rel_err = max(max_rel_err, float(((v16.float() - v32).abs()[nz] / v32.abs()[nz]).max()))
            total += v32.numel()
            changed += int(not torch.equal(v16, v_ref))
            if groups_checked < 20000:  # sanity: at most 16 distinct values per group
                g = v32.reshape(v32.shape[0], -1, a.group_size)[:8]
                for row in g.reshape(-1, a.group_size):
                    groups_checked += 1
                    groups_bad += int(row.unique().numel() > 16)
        elif not torch.equal(v16, v_ref):
            if "lm_head" in k or "embed_tokens" in k:
                raise RuntimeError(f"{k} must not change")
            other_changed.append(k)
        out_sd[k] = v16.contiguous()
    del dq, dq_sd, ref
    gc.collect()
    out_dq.mkdir(parents=True)
    save_file(out_sd, str(out_dq / "model.safetensors"), metadata={"format": "pt"})
    for f in ("config.json", "generation_config.json"):
        if (src / f).exists():
            shutil.copy(src / f, out_dq / f)
    report["dequant"] = {
        "path": str(out_dq),
        "how": "CompressedTensorsConfig(dequantize=True) in FP32, rounded once to BF16; source config.json",
        "dequant_bf16_inexact": inexact,
        "quantized_elements": total,
        "dequant_bf16_max_rel_err": max_rel_err,
        "dequant_bf16_note": "BF16 stores 7 mantissa bits, so round-to-nearest has relative error <= 2^-8; for "
                             "INT4 values |q - zp| <= 15 the absolute error is <= 15*2^-8 < 1/17 of one quantization "
                             "step (scale)",
        "quantized_tensors_changed_vs_source_bf16": changed,
        "non_quantized_tensors_changed": other_changed,
        "groups_checked": groups_checked,
        "groups_with_more_than_16_values": groups_bad,
        "minutes": round((time.time() - t0) / 60, 2),
    }
    if groups_bad:
        raise RuntimeError(f"{groups_bad} groups have more than 16 distinct values: not INT4")
    report["end"] = utc_now()
    write_json(report, out / "quant_report.json")
    print(json.dumps({k: v for k, v in report.items() if k != "calibration"}, indent=2, default=str))
    return report


if __name__ == "__main__":
    sys.exit(main())
