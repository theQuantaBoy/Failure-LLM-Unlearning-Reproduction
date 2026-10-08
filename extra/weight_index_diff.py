"""
extra/weight_index_diff.py — the paper's Section 5 mechanism, measured directly: after INT4 RTN (W4A16, symmetric,
group 128, lm_head excluded — the deterministic configuration of quantize_run.py), how many INT4 indices of an
unlearned checkpoint differ from those of the target it was trained from? No GPU, no calibration data.

Per Linear weight (q, k, v, o, gate, up, down of every layer) and per (target, unlearned) pair:
  w_changed      fraction of BF16 weights that differ (what unlearning changed at all)
  idx_diff       fraction of INT4 indices that differ
  scale_diff     fraction of group scales that differ
  dq_equal       fraction of dequantized values (scale · q) that are identical
  absorbed       fraction of CHANGED weights whose dequantized value is unchanged (the update was "rounded away")
Aggregated per module type, per layer and overall (element-weighted). Output: <out>/index_diff.csv (per tensor),
<out>/index_diff_summary.json, <out>/index_diff.md.

Exactness: indices and scales come from compressed-tensors' own calculate_qparams / quantize (the functions
llm-compressor's QuantizationModifier uses with the memoryless_minmax observer: per-group amin/amax of the BF16
weight). `--verify SRC COMPRESSED` checks that claim against a real llm-compressor RTN output (packed INT4 values and
scales must be bit-identical); extra/tests/test_weight_index_diff.py runs it on the tiny CPU model.

    # in the quantization image / venv (needs torch, compressed-tensors 0.19.0, safetensors):
    python -m extra.weight_index_diff --target <target dir> --unlearned <dir> [<dir> ...] --out <dir>
    python -m extra.weight_index_diff --verify <src dir> <compressed dir>
    # on Modal (CPU only, runs Volume of the account that has the checkpoints; see modal_quant.py::weight_index_diff):
    modal run --detach modal_quant.py::weight_index_diff --unlearned ckpt/books/books_npo_klr_s42,...

Quotient mode (2026-10-06): compressed-tensors 0.19.0 quantizes with two different kernels.
  bf16  the torch fallback `_quantize` (forward_helpers.py:517-530): scaled = x / scale in the weight dtype, i.e. the
        quotient is rounded to BF16 before torch.round. Used for CPU tensors (the tiny CPU test).
  fp32  the Triton backend `_quantize_triton` (forward_helpers.py:393, registered for CUDA/XPU tensors via
        utils/triton.py:98-101): output = div_rn(x.fp32, scale.fp32), clamp, rint (round half to even) with no BF16
        rounding of the quotient (kernel body forward_helpers.py:210-300). Used when llm-compressor runs on a GPU.
Both share the scale (calculate_qparams in the weight dtype, helpers.py:71-88; observer min_max.py:70-74) and clamp
before rounding (quant_args.py:449-458). The two modes can differ by exactly +-1 index where the quotient lies within
half a BF16 ulp of a .5 boundary (emulated on N(0, 0.02) BF16 weights: 6.5e-3 of indices, all +-1, scales identical).

--verify writes a per-tensor report for BOTH modes and applies the acceptance rule fixed in ACCEPTANCE before any diff.

Loading: each tensor is read lazily from safetensors (sharded or not) and cast to BF16 exactly as quantize_run loads
the model (from_pretrained(torch_dtype=bfloat16): round-to-nearest-even from FP32), one tensor at a time.
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

LINEAR = re.compile(r"^model\.layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
                     r"\.weight$")


class Ckpt:
    """Lazy tensor access to a HF safetensors checkpoint directory (single file or sharded with an index)."""

    def __init__(self, d):
        from safetensors import safe_open

        d = Path(d)
        idx = d / "model.safetensors.index.json"
        if idx.exists():
            files = json.loads(idx.read_text())["weight_map"]
        else:
            with safe_open(d / "model.safetensors", "pt") as f:
                files = {k: "model.safetensors" for k in f.keys()}
        self.dir, self.files, self._open = d, files, {}

    def names(self):
        return sorted(self.files)

    def get(self, name):
        from safetensors import safe_open

        fn = self.files[name]
        if fn not in self._open:
            self._open[fn] = safe_open(self.dir / fn, "pt")
        return self._open[fn].get_tensor(name)


def rtn_args(group_size=128, symmetric=True):
    from compressed_tensors.quantization import QuantizationArgs

    return QuantizationArgs(num_bits=4, type="int", strategy="group", group_size=group_size, symmetric=symmetric,
                            dynamic=False, observer="memoryless_minmax")


QUOTIENT_MODES = ("bf16", "fp32")


def rtn(w, args, quotient="bf16"):
    """(INT4 indices as int8 [out, in], scales [out, in/gs]) of one BF16 weight, via compressed-tensors.

    quotient="bf16": compressed-tensors' torch `quantize` (CPU path). quotient="fp32": the arithmetic of its CUDA Triton
    kernel (FP32 division, clamp, round half to even), emulated in torch on CPU (see the module docstring)."""
    import torch
    from compressed_tensors.quantization.lifecycle.forward import quantize
    from compressed_tensors.quantization.utils import calculate_qparams, calculate_range

    w = w.to(torch.bfloat16)
    out, inp = w.shape
    g = w.reshape(out, inp // args.group_size, args.group_size)
    scale, zp = calculate_qparams(torch.amin(g, dim=-1), torch.amax(g, dim=-1), args)
    if quotient == "bf16":
        q = quantize(w, scale, zp, args, dtype=torch.int8)
    elif quotient == "fp32":
        q_min, q_max = calculate_range(args, w.device)
        quo = g.float() / scale.float().unsqueeze(-1)
        q = torch.round(torch.clamp(quo, float(q_min), float(q_max))).to(torch.int8).reshape(out, inp)
    else:
        raise ValueError(f"quotient must be one of {QUOTIENT_MODES}")
    return q, scale


# Acceptance rule for --verify, fixed BEFORE the first informative run (2026-10-06).
# Evaluated per quotient mode over every verified tensor:
#   EXACT     shapes equal, scales bit-identical (same dtype), INT4 indices bit-identical in every tensor.
#   TOLERANT  shapes equal; scales equal after rounding both to BF16 (max relative difference <= 2^-8); in every tensor
#             the fraction of differing indices <= MAX_IDX_MISMATCH and every difference is exactly +-1.
#   FAIL      otherwise.
# The diff then uses the best passing mode (EXACT before TOLERANT, then the smaller worst-tensor mismatch). With
# TOLERANT, idx_diff / dq_equal / absorbed carry an absolute uncertainty of +- eps (eps = worst per-tensor mismatch
# fraction), stated in index_diff.md; w_changed (a BF16 comparison) never depends on the quantizer.
MAX_IDX_MISMATCH = 1e-4
MAX_SCALE_REL = 2.0 ** -8
ACCEPTANCE = {"exact": "shapes equal, scales bit-identical, indices bit-identical in every tensor",
              "tolerant": f"scales equal in BF16 (max rel diff <= 2^-8), per-tensor index mismatch <= {MAX_IDX_MISMATCH:g}, "
                          "all differences +-1",
              "max_idx_mismatch": MAX_IDX_MISMATCH, "max_scale_rel": MAX_SCALE_REL}


def _layer_ok(name, layers):
    return layers is None or int(LINEAR.match(name).group(1)) in layers


def verify_report(src, compressed, group_size=128, layers=None, log=print) -> dict:
    """Per-tensor comparison of rtn() (both quotient modes) with an llm-compressor RTN pack-quantized checkpoint."""
    import torch
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

    a, c = Ckpt(src), Ckpt(compressed)
    args = rtn_args(group_size)
    cnames = set(c.names())
    tensors = []
    t0 = time.time()
    names = [n for n in a.names() if LINEAR.match(n) and _layer_ok(n, layers)]
    for i, name in enumerate(names):
        base = name[: -len(".weight")]
        w = a.get(name)
        shape = tuple(int(x) for x in c.get(f"{base}.weight_shape").tolist())
        qc = unpack_from_int32(c.get(f"{base}.weight_packed"), 4, torch.Size(shape)).to(torch.int16)
        sc = c.get(f"{base}.weight_scale")
        rec = {"tensor": name, "n": w.numel(), "src_shape": list(w.shape), "src_dtype": str(w.dtype),
               "compressed_shape": list(shape), "shape_equal": list(w.shape) == list(shape),
               "scale_dtype_compressed": str(sc.dtype), "scale_shape_compressed": list(sc.shape),
               "zero_point_present": f"{base}.weight_zero_point" in cnames,
               "packed_dtype": str(c.get(f"{base}.weight_packed").dtype), "modes": {}}
        for mode in QUOTIENT_MODES:
            q, s = rtn(w, args, quotient=mode)
            m = {"scale_shape": list(s.shape), "scale_dtype": str(s.dtype)}
            if list(s.shape) != list(sc.shape) or not rec["shape_equal"]:
                m.update(comparable=False)
                rec["modes"][mode] = m
                continue
            rel = ((s.float() - sc.float()).abs() / sc.float().abs().clamp_min(1e-30))
            d = q.to(torch.int16) - qc
            nz = d != 0
            m.update(comparable=True, scale_bit_identical=bool(s.dtype == sc.dtype and torch.equal(s, sc)),
                     scale_equal_in_bf16=bool(torch.equal(s.to(torch.bfloat16), sc.to(torch.bfloat16))),
                     scale_max_rel_diff=float(rel.max()), scales_differing=int((s.float() != sc.float()).sum()),
                     idx_mismatch=int(nz.sum()), idx_mismatch_frac=float(nz.float().mean()),
                     max_abs_didx=int(d.abs().max()),
                     didx_hist={str(k): int(v) for k, v in zip(*torch.unique(d[nz], return_counts=True))} if nz.any() else {})
            if nz.any():  # where do mismatches sit? group extremes (|w| = group absmax) vs interior
                g = w.to(torch.bfloat16).reshape(w.shape[0], -1, group_size)
                ext = (g.abs() == g.abs().amax(-1, keepdim=True)).reshape(w.shape)
                m["mismatch_at_group_absmax"] = int((nz & ext).sum())
                idx = nz.nonzero()[:3].tolist()
                m["examples"] = [{"row": r, "col": k, "w": float(w[r, k]), "scale": float(sc[r, k // group_size]),
                                  "quotient_fp32": float(w[r, k].float() / sc[r, k // group_size].float()),
                                  "ours": int(q[r, k]), "compressed": int(qc[r, k])} for r, k in idx]
            rec["modes"][mode] = m
        tensors.append(rec)
        if (i + 1) % 8 == 0 or i + 1 == len(names):
            log(f"verify {i + 1}/{len(names)} tensors, {time.time() - t0:.0f} s")
    return decide({"src": str(src), "compressed": str(compressed), "group_size": group_size,
                   "layers": sorted(layers) if layers is not None else "all", "n_tensors": len(tensors),
                   "acceptance": ACCEPTANCE, "tensors": tensors})


def decide(rep: dict) -> dict:
    """Apply ACCEPTANCE to a verify report (adds per-mode summaries, 'decision' and 'quotient')."""
    summ = {}
    for mode in QUOTIENT_MODES:
        ms = [t["modes"].get(mode, {}) for t in rep["tensors"]]
        comparable = bool(ms) and all(m.get("comparable") for m in ms)
        s = {"comparable": comparable}
        if comparable:
            s.update(
                tensors_exact=sum(m["scale_bit_identical"] and m["idx_mismatch"] == 0 for m in ms),
                tensors_scale_bit_identical=sum(m["scale_bit_identical"] for m in ms),
                worst_idx_mismatch_frac=max(m["idx_mismatch_frac"] for m in ms),
                total_idx_mismatch_frac=sum(m["idx_mismatch"] for m in ms) / sum(t["n"] for t in rep["tensors"]),
                max_abs_didx=max(m["max_abs_didx"] for m in ms),
                worst_scale_rel_diff=max(m["scale_max_rel_diff"] for m in ms))
            n = len(ms)
            if s["tensors_exact"] == n:
                s["verdict"] = "exact"
            elif (all(m["scale_equal_in_bf16"] and m["scale_max_rel_diff"] <= MAX_SCALE_REL for m in ms)
                  and s["worst_idx_mismatch_frac"] <= MAX_IDX_MISMATCH and s["max_abs_didx"] <= 1):
                s["verdict"] = "tolerant"
            else:
                s["verdict"] = "fail"
        else:
            s["verdict"] = "fail"
        summ[mode] = s
    rank = {"exact": 0, "tolerant": 1, "fail": 2}
    best = min(QUOTIENT_MODES, key=lambda m: (rank[summ[m]["verdict"]], summ[m].get("worst_idx_mismatch_frac", 1.0)))
    rep["summary"] = summ
    rep["decision"] = summ[best]["verdict"]
    rep["quotient"] = best if rep["decision"] != "fail" else None
    rep["eps"] = summ[best].get("worst_idx_mismatch_frac") if rep["decision"] == "tolerant" else 0.0
    return rep


def verify(src, compressed, group_size=128, quotient="bf16"):
    """Backward-compatible check: (n tensors, names that are not bit-identical in `quotient` mode)."""
    rep = verify_report(src, compressed, group_size, log=lambda *_: None)
    bad = [t["tensor"] for t in rep["tensors"]
           if not (t["modes"][quotient].get("comparable") and t["modes"][quotient]["scale_bit_identical"]
                   and t["modes"][quotient]["idx_mismatch"] == 0)]
    return rep["n_tensors"], bad


def diff(target, unlearned: dict, group_size=128, log=print, quotient="bf16"):
    """Per-tensor rows for every (target, unlearned) pair; the target is read and quantized once per tensor."""
    import torch

    t = Ckpt(target)
    us = {k: Ckpt(v) for k, v in unlearned.items()}
    args = rtn_args(group_size)
    rows = []
    names = [n for n in t.names() if LINEAR.match(n)]
    t0 = time.time()
    for i, name in enumerate(names):
        layer, _, mod = LINEAR.match(name).groups()
        wt = t.get(name).to(torch.bfloat16)
        qt, st = rtn(wt, args, quotient)
        dqt = st.repeat_interleave(group_size, dim=1).float() * qt.float()
        for key, u in us.items():
            wu = u.get(name).to(torch.bfloat16)
            qu, su = rtn(wu, args, quotient)
            dqu = su.repeat_interleave(group_size, dim=1).float() * qu.float()
            changed = wt != wu
            same_dq = dqt == dqu
            n_ch = int(changed.sum())
            rows.append({"pair": key, "tensor": name, "layer": int(layer), "module": mod, "n": wt.numel(),
                         "n_groups": st.numel(), "w_changed": n_ch, "idx_diff": int((qt != qu).sum()),
                         "scale_diff": int((st != su).sum()), "dq_equal": int(same_dq.sum()),
                         "absorbed": int((changed & same_dq).sum())})
        if (i + 1) % 16 == 0 or i + 1 == len(names):
            log(f"{i + 1}/{len(names)} tensors, {time.time() - t0:.0f} s")
    return rows


def summarize(rows):
    """Element-weighted fractions per pair x {overall, module, layer}."""
    out = {}
    for r in rows:
        for key in ("overall", f"module:{r['module']}", f"layer:{r['layer']:02d}"):
            s = out.setdefault(r["pair"], {}).setdefault(key, {k: 0 for k in
                                                                ("n", "n_groups", "w_changed", "idx_diff",
                                                                 "scale_diff", "dq_equal", "absorbed")})
            for k in s:
                s[k] += r[k]
    frac = {}
    for pair, groups in out.items():
        for key, s in groups.items():
            frac.setdefault(pair, {})[key] = {
                "w_changed": s["w_changed"] / s["n"], "idx_diff": s["idx_diff"] / s["n"],
                "scale_diff": s["scale_diff"] / s["n_groups"], "dq_equal": s["dq_equal"] / s["n"],
                "absorbed": s["absorbed"] / s["w_changed"] if s["w_changed"] else None, "n": s["n"]}
    return frac


def write_outputs(rows, frac, out: Path, meta: dict):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "index_diff.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "index_diff_summary.json").write_text(json.dumps({"meta": meta, "fractions": frac}, indent=1))
    md = ["# INT4 RTN index differences, target vs unlearned (generated by extra/weight_index_diff.py)\n",
          "| Pair | Group | BF16 weights changed % | INT4 indices differ % | group scales differ % | "
          "dequantized equal % | changed weights absorbed % |", "|---|---|---|---|---|---|---|"]
    for pair, groups in frac.items():
        for key in sorted(groups, key=lambda k: (k != "overall", k)):
            if key.startswith("layer:"):
                continue
            f = groups[key]
            ab = "—" if f["absorbed"] is None else f"{100 * f['absorbed']:.2f}"
            md.append(f"| {pair} | {key} | {100 * f['w_changed']:.2f} | {100 * f['idx_diff']:.2f} | "
                      f"{100 * f['scale_diff']:.2f} | {100 * f['dq_equal']:.2f} | {ab} |")
    v = meta.get("verification")
    if v:
        md.insert(1, f"INT4 indices recomputed with quotient={meta.get('quotient')}. Verification against llm-compressor's "
                     f"RTN output ({v.get('n_tensors')} tensors, layers {v.get('layers')}): **{v.get('decision')}**"
                     + (f"; idx_diff / dq_equal / absorbed carry an absolute uncertainty of +-{v.get('eps'):.1e} per tensor"
                        if v.get("decision") == "tolerant" else "") + ". BF16 weights changed % does not depend on it.\n")
    else:
        md.insert(1, f"INT4 indices recomputed with quotient={meta.get('quotient')}; NOT verified against llm-compressor "
                     "(INT4 columns unconfirmed). BF16 weights changed % does not depend on it.\n")
    md.append("\nPer-layer values: index_diff_summary.json (keys layer:NN) and index_diff.csv (per tensor).")
    (out / "index_diff.md").write_text("\n".join(md) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target")
    ap.add_argument("--unlearned", nargs="*", default=[], help="checkpoint dirs; name = dir name, or name=dir")
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--out")
    ap.add_argument("--logs_dir", default=None, help="write a RunLog record there (Modal job)")
    ap.add_argument("--verify", nargs=2, metavar=("SRC", "COMPRESSED"))
    ap.add_argument("--verify_layers", default="", help="comma-separated layer indices to verify (default: all)")
    ap.add_argument("--verify_out", help="write the full verify report (JSON) here")
    ap.add_argument("--verification", help="diff mode: a verify report; the diff refuses to run unless it passed, "
                                           "and uses its quotient mode")
    ap.add_argument("--quotient", choices=list(QUOTIENT_MODES) + ["auto"], default="auto",
                    help="diff mode: quotient arithmetic; auto = from --verification (required then)")
    a = ap.parse_args(argv)
    if a.verify:
        layers = {int(x) for x in a.verify_layers.split(",") if x.strip()} or None
        runlog = None
        if a.logs_dir:
            from extra.common import RunLog

            runlog = RunLog(a.logs_dir, "index_diff_verify", f"g{a.group_size}",
                            {"src": a.verify[0], "compressed": a.verify[1], "group_size": a.group_size,
                             "layers": sorted(layers) if layers else "all", "acceptance": ACCEPTANCE})
        rep = verify_report(*a.verify, group_size=a.group_size, layers=layers)
        if a.verify_out:
            Path(a.verify_out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.verify_out).write_text(json.dumps(rep, indent=1))
        print(f"verify: {rep['n_tensors']} Linear tensors; decision {rep['decision']} (quotient {rep['quotient']})")
        for mode, s in rep["summary"].items():
            print(f"  {mode}: " + ", ".join(f"{k}={v}" for k, v in s.items()))
        for t in rep["tensors"][:3]:
            print("  e.g.", t["tensor"], {m: {k: v for k, v in d.items() if k not in ("examples", "didx_hist")}
                                          for m, d in t["modes"].items()})
        ok = rep["decision"] != "fail" and rep["n_tensors"] > 0
        if runlog:
            runlog.finish(0 if ok else 1, results={k: rep[k] for k in ("decision", "quotient", "eps", "summary",
                                                                         "n_tensors", "layers")})
        return 0 if ok else 1
    verification = None
    quotient = a.quotient
    if a.verification:
        verification = json.loads(Path(a.verification).read_text())
        if verification.get("decision") not in ("exact", "tolerant"):
            print(f"verification {a.verification} did not pass ({verification.get('decision')}): not running the diff")
            return 1
        if quotient == "auto":
            quotient = verification["quotient"]
        elif quotient != verification["quotient"]:
            print(f"--quotient {quotient} contradicts the verified mode {verification['quotient']}")
            return 1
    if quotient == "auto":
        print("--quotient auto needs --verification (or pass --quotient bf16|fp32 explicitly: unverified)")
        return 1
    unl = dict(x.split("=", 1) if "=" in x else (Path(x).name, x) for x in a.unlearned)
    runlog = None
    if a.logs_dir:
        from extra.common import RunLog

        runlog = RunLog(a.logs_dir, "index_diff", f"rtn_g{a.group_size}", {**vars(a), "pairs": unl})
    try:
        rows = diff(a.target, unl, a.group_size, quotient=quotient)
        frac = summarize(rows)
        meta = {"target": a.target, "unlearned": unl, "group_size": a.group_size, "symmetric": True,
                "quotient": quotient,
                "verification": ({k: verification[k] for k in ("decision", "quotient", "eps", "n_tensors", "layers",
                                                                 "src", "compressed")} if verification else None),
                "quantizer": "compressed-tensors calculate_qparams + quantize, memoryless_minmax (= llm-compressor "
                             "QuantizationModifier RTN as in extra/quantize_run.py)"}
        write_outputs(rows, frac, Path(a.out), meta)
        for pair, g in frac.items():
            o = g["overall"]
            print(f"{pair}: w_changed {100 * o['w_changed']:.2f}%  idx_diff {100 * o['idx_diff']:.2f}%  "
                  f"absorbed {100 * (o['absorbed'] or 0):.2f}%")
        if runlog:
            runlog.finish(0, results={p: g["overall"] for p, g in frac.items()})
        return 0
    except BaseException:
        import traceback

        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        if runlog:
            runlog.finish(1, error=tb)
        return 1


if __name__ == "__main__":
    sys.exit(main())
