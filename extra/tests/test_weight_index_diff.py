"""
Offline test of extra/weight_index_diff.py on the tiny CPU model of run_cpu_e2e.py (needs its outputs in
extra/tests/_out; run `python3 extra/tests/run_cpu_e2e.py` first).

Checks: (1) the INT4 indices and scales it computes are bit-identical to llm-compressor's RTN output
(_out/quant/books_npo_klr_sure_s42/rtn_g128_nocalib/compressed, made from _out/ckpt/books/books_npo_klr_sure_s42);
(2) verify() reports mismatches for the wrong source (tiny target) — the check is not vacuous; (3) target vs itself:
0 changed weights, 0 differing indices, 100 % dequantized-equal; (4) target vs the 4 tiny unlearned checkpoints:
counts are consistent (absorbed <= w_changed <= n, idx_diff <= n) and the outputs exist; (5) the informative verify:
the CPU-made checkpoint verifies "exact" in quotient mode bf16; a GPU-style (FP32-quotient) checkpoint verifies "exact"
in mode fp32 and differs from bf16 by +-1 only with identical scales; the acceptance rule gives tolerant / fail as
designed; --verify_layers filters; the diff refuses a failed or absent verification and states the mode it used.

    .venvs/quant/bin/python -m extra.tests.test_weight_index_diff
"""

import json
import tempfile
from pathlib import Path

from extra import weight_index_diff as wid

OUT = Path(__file__).parent / "_out"


def _fake_compressed(src: Path, out: Path, quotient: str, perturb_scale=False, flip=0):
    """A pack-quantized checkpoint made from rtn(quotient) (what llm-compressor writes, CPU or GPU kernel)."""
    import torch
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    from safetensors.torch import save_file

    a = wid.Ckpt(src)
    sd = {}
    for name in a.names():
        if not wid.LINEAR.match(name):
            continue
        base = name[: -len(".weight")]
        q, sc = wid.rtn(a.get(name), wid.rtn_args(128), quotient=quotient)
        if flip:  # move `flip` interior indices by +-1 (never out of range)
            flat = q.view(-1)
            for i in range(flip):
                flat[i * 997] = flat[i * 997] + (1 if flat[i * 997] < 7 else -1)
        if perturb_scale:
            sc = sc * 1.01
        sd[f"{base}.weight_packed"] = pack_to_int32(q, 4)
        sd[f"{base}.weight_scale"] = sc.contiguous()
        sd[f"{base}.weight_shape"] = torch.tensor(q.shape)
    out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out / "model.safetensors"))
    return out


def verify_modes(src: Path, comp: Path) -> dict:
    """The informative verify (2026-10-06): mode discrimination, acceptance outcomes, diff gating."""
    res = {}
    rep = wid.verify_report(src, comp, log=lambda *_: None)
    assert rep["decision"] == "exact" and rep["quotient"] == "bf16", rep["summary"]  # CPU-made -> torch path
    res["cpu_compressed"] = {m: rep["summary"][m]["verdict"] for m in wid.QUOTIENT_MODES}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # GPU-style output (Triton arithmetic): fp32 must be exact, bf16 must differ only by +-1 with equal scales
        rep = wid.verify_report(src, _fake_compressed(src, tmp / "fp32", "fp32"), log=lambda *_: None)
        sb = rep["summary"]["bf16"]
        assert rep["decision"] == "exact" and rep["quotient"] == "fp32", rep["summary"]
        assert sb["tensors_scale_bit_identical"] == rep["n_tensors"] and sb["max_abs_didx"] <= 1, sb
        res["gpu_style_compressed"] = {"decision": rep["decision"], "quotient": rep["quotient"],
                                       "bf16_total_idx_mismatch_frac": sb["total_idx_mismatch_frac"]}
        # a few +-1 flips -> tolerant only if <= 1e-4 per tensor: tiny tensors (65k-131k elements) -> 1 flip = 7.6e-6..
        rep = wid.verify_report(src, _fake_compressed(src, tmp / "flip1", "bf16", flip=1), log=lambda *_: None)
        assert rep["decision"] == "tolerant" and rep["eps"] > 0, rep["summary"]
        rep = wid.verify_report(src, _fake_compressed(src, tmp / "flip50", "bf16", flip=50), log=lambda *_: None)
        assert rep["decision"] == "fail", rep["summary"]  # 50 flips > 1e-4 of a 65,536-element tensor
        rep = wid.verify_report(src, _fake_compressed(src, tmp / "scale", "bf16", perturb_scale=True),
                                log=lambda *_: None)
        assert rep["decision"] == "fail", rep["summary"]  # scales 1 % off -> fails the scale criterion
        res["acceptance_outcomes"] = "exact / tolerant / fail(index) / fail(scale) as designed"
        # layer filter
        rep = wid.verify_report(src, comp, layers={0}, log=lambda *_: None)
        assert rep["n_tensors"] == 7, rep["n_tensors"]
        # diff gating through the CLI
        bad = tmp / "verify_fail.json"
        bad.write_text(json.dumps({"decision": "fail", "quotient": None}))
        rc = wid.main(["--target", str(src), "--unlearned", str(src), "--out", str(tmp / "d1"),
                       "--verification", str(bad)])
        assert rc == 1 and not (tmp / "d1").exists()
        good = tmp / "verify_ok.json"
        assert wid.main(["--verify", str(src), str(comp), "--verify_layers", "0,1", "--verify_out", str(good)]) == 0
        rc = wid.main(["--target", str(src), "--unlearned", str(src), "--out", str(tmp / "d2"),
                       "--verification", str(good)])
        md = (tmp / "d2" / "index_diff.md").read_text()
        assert rc == 0 and "quotient=bf16" in md and "**exact**" in md, md[:400]
        assert wid.main(["--target", str(src), "--unlearned", str(src), "--out", str(tmp / "d3")]) == 1  # auto w/o verify
        res["diff_gating"] = "refuses failed/absent verification; uses the verified quotient; states it in index_diff.md"
    return res


def main():
    src = OUT / "ckpt/books/books_npo_klr_sure_s42"
    comp = OUT / "quant/books_npo_klr_sure_s42/rtn_g128_nocalib/compressed"
    target = OUT / "tiny_target"
    for p in (src, comp, target):
        assert p.exists(), f"{p} missing: run extra/tests/run_cpu_e2e.py first"
    res = {}

    n, bad = wid.verify(src, comp)
    assert n == 14 and not bad, (n, bad)
    res["verify_vs_llmcompressor"] = f"{n} tensors bit-identical"

    n, bad = wid.verify(target, comp)
    assert bad, "verify() did not notice a different source checkpoint"
    res["verify_wrong_source"] = f"{len(bad)}/{n} mismatches detected"

    rows = wid.diff(target, {"self": str(target)}, log=lambda *_: None)
    assert all(r["w_changed"] == 0 and r["idx_diff"] == 0 and r["scale_diff"] == 0 and r["dq_equal"] == r["n"]
               for r in rows), rows[:2]
    res["self_pair"] = "0 changes"

    unl = {m: str(OUT / f"ckpt/books/books_{m}_s42") for m in ("ga_gdr", "npo_klr", "ga_gdr_sure", "npo_klr_sure")}
    rows = wid.diff(target, unl, log=lambda *_: None)
    assert len(rows) == 14 * 4
    for r in rows:
        assert r["idx_diff"] <= r["n"] and r["absorbed"] <= r["w_changed"] <= r["n"], r  # a scale change can move
        # the index of an unchanged weight, so idx_diff is not bounded by w_changed
    frac = wid.summarize(rows)
    with tempfile.TemporaryDirectory() as tmp:
        wid.write_outputs(rows, frac, Path(tmp), {"test": True})
        for f in ("index_diff.csv", "index_diff_summary.json", "index_diff.md"):
            assert (Path(tmp) / f).stat().st_size > 0
    res["tiny_pairs"] = {k: {m: round(v, 4) for m, v in g["overall"].items() if m != "n"} for k, g in frac.items()}

    res.update(verify_modes(src, comp))

    print(json.dumps(res, indent=2))
    print("test_weight_index_diff: ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
