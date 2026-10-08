"""
D2 + D3 test (GPU smoke test; also runs on CPU with a tiny model).
D2 = is the released SURE's saliency mask applied?  D3 = is --sure_fast equivalent to the released path?

Three short runs of the same SURE configuration from the same target model, each `--max_steps N`:
  A  authors' SURE as-is, with the mask audit (W3) and the optimizer_step counter (W2)
  B  authors' SURE as-is (repeat of A, without the audit)        -> run-to-run noise floor
  C  --sure_fast (W4): m_S construction skipped, loss_f.backward(retain_graph=True) kept

D2 verdict: SURE.optimizer_step was called 0 times in A, B, C, AND in A at least one parameter row with mask 0
            changed during a step  =>  the saliency mask is not applied.
D3 verdict: diff(C, A) <= diff(B, A) for both the number of differing elements and the max |difference| over all
            weights  =>  the fast path may be used. Otherwise run SURE as-is.

--runs selects which of A/B/C to run (default ABC = the released smoke test, unchanged). D3 needs A, B and C.
W7 fixed presets with a mask (e.g. books_npo_klr_sure_masked; FINDINGS.md §6): only A is possible
(--sure_fast is refused with fixes). Its audit reads the mask actually applied in each step and reports, per step
and in total over the audited rows (same rows, seed 42, batches and steps as the released test):
  (a) D2a  rows outside the current mask with a non-zero gradient after masking   -> must be 0 (mask applied)
  (b) D2   rows outside the current mask whose weights changed (released D2 definition, after the step)
  (c) D2c  rows of (b) that were inside the mask in an earlier step
  plus     rows outside the current mask that optimizer.step moved BEFORE the fix's restore (AdamW momentum), and
           how many of those were inside the mask earlier.
Verdict D2_fixed_mask_applied: every step audited with a mask and (a) = 0. (b) and (c) are reported, not forced.

    python -m extra.tests.test_sure_equivalence --preset books_npo_klr_sure --steps 20 \
        --model_dir <target snapshot> --tokenizer_dir <tokenizer> --work_dir <dir> --logs_dir <dir> [--cpu_test ...]
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from extra.common import RunLog, read_json, write_json


def weights(d: Path):
    from safetensors import safe_open

    files = sorted(d.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {d}")
    for f in files:
        with safe_open(str(f), framework="pt") as fh:
            for k in fh.keys():
                yield k, fh.get_tensor(k)


def diff(d1: Path, d2: Path) -> dict:
    import torch

    t2 = dict(weights(d2)) if sum(f.stat().st_size for f in d2.glob("*.safetensors")) < 2e9 else None
    n_diff, n_tot, max_abs, tensors = 0, 0, 0.0, 0
    if t2 is None:  # large model: stream the second checkpoint per file
        from safetensors import safe_open

        idx = {}
        for f in sorted(d2.glob("*.safetensors")):
            with safe_open(str(f), framework="pt") as fh:
                for k in fh.keys():
                    idx[k] = f
    for k, a in weights(d1):
        if t2 is not None:
            b = t2[k]
        else:
            from safetensors import safe_open

            with safe_open(str(idx[k]), framework="pt") as fh:
                b = fh.get_tensor(k)
        ne = (a != b)
        n = int(ne.sum())
        n_diff += n
        n_tot += a.numel()
        tensors += int(n > 0)
        if n:
            max_abs = max(max_abs, float((a.float() - b.float()).abs().max()))
    return {"elements_differing": n_diff, "elements_total": n_tot, "tensors_differing": tensors,
            "max_abs_diff": max_abs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--delete_weights", action="store_true")
    ap.add_argument("--passthrough", default="", help="extra args for extra.unlearn_run (e.g. tiny data, --cpu_test)")
    ap.add_argument("--runs", default="ABC", help="subset of ABC (fixed presets: A only)")
    a = ap.parse_args(argv)
    work = Path(a.work_dir)
    if work.exists():
        shutil.rmtree(work)
    runlog = RunLog(a.logs_dir, "test", f"sure_equivalence_{a.preset}", vars(a))

    common = [sys.executable, "-m", "extra.unlearn_run", "--preset", a.preset, "--model_dir", a.model_dir,
              "--tokenizer_dir", a.tokenizer_dir, "--logs_dir", a.logs_dir, "--test_mode",
              "--max_steps", str(a.steps), "--save_strategy", "no", "--logging_steps", "1"] + a.passthrough.split()
    runs = {t: x for t, x in {"A": ["--audit_mask"], "B": [], "C": ["--sure_fast"]}.items() if t in a.runs}
    if "A" not in runs:
        raise SystemExit("--runs must include A (the audited run)")
    info = {}
    for tag, extra in runs.items():
        out = work / tag
        rc = subprocess.run(common + ["--out_dir", str(out), "--run_name", f"sure_eq_{a.preset}_{tag}"] + extra).returncode
        if rc != 0:
            runlog.finish(rc, failed_run=tag)
            return rc
        info[tag] = read_json(out / "unlearn_run.json")

    calls = {t: info[t]["counters"]["sure_optimizer_step_calls"] for t in info}
    audit = info["A"]["stats"].get("mask_audit", [])
    outside = sum(p["rows_changed_outside_mask"] for r in audit for p in r["params"].values())
    losses = {t: [h.get("loss") for h in info[t]["stats"].get("log_history", []) if "loss" in h] for t in info}
    fixed = "fixes" in info["A"]["config"]
    verdict = {
        "D2_optimizer_step_calls": calls,
        "D2_rows_changed_outside_mask_total": outside,
        "D2_mask_audit": audit,
    }
    if fixed:  # W7: audit of the mask actually applied in each step
        def per_step(key):
            return [sum((p.get(key) or 0) for p in r["params"].values()) for r in audit]

        steps = {k: per_step(k) for k in (
            "rows_outside_mask_nonzero_grad", "rows_changed_outside_mask", "rows_changed_outside_mask_prev_inside",
            "rows_moved_by_optimizer_outside_mask", "rows_moved_by_optimizer_outside_mask_prev_inside",
            "rows_outside_mask_prev_inside")}
        inside = sum(p["rows_changed"] - p["rows_changed_outside_mask"] for r in audit for p in r["params"].values())
        a_ok = all(p.get("rows_outside_mask_nonzero_grad") == 0 for r in audit for p in r["params"].values())
        verdict.update({
            "fixes": info["A"]["config"]["fixes"], "deviation": info["A"]["config"].get("deviation"),
            "D2a_rows_outside_mask_nonzero_grad_total": sum(steps["rows_outside_mask_nonzero_grad"]),
            "D2c_rows_changed_outside_mask_prev_inside_total": sum(steps["rows_changed_outside_mask_prev_inside"]),
            "D2_rows_moved_by_optimizer_before_restore_total": sum(steps["rows_moved_by_optimizer_outside_mask"]),
            "D2_rows_moved_by_optimizer_before_restore_prev_inside_total":
                sum(steps["rows_moved_by_optimizer_outside_mask_prev_inside"]),
            "D2_rows_changed_inside_mask_total": inside,
            "D2_per_step": steps,
            "D2_steps_audited": len(audit),
            "D2_all_steps_have_mask": bool(audit) and all(r["mask_available"] for r in audit),
            "D2_fixed_mask_applied": bool(audit) and all(r["mask_available"] for r in audit)
            and len(audit) == a.steps and a_ok,
        })
    else:
        verdict["D2_mask_is_dead_code"] = all(c == 0 for c in calls.values()) and outside > 0
    if {"A", "B", "C"} <= set(runs):
        d_ba = diff(work / "B", work / "A")
        d_ca = diff(work / "C", work / "A")
        verdict.update({
            "D3_diff_B_vs_A (noise floor)": d_ba,
            "D3_diff_C_vs_A (fast vs original)": d_ca,
            "D3_pass": d_ca["elements_differing"] <= d_ba["elements_differing"]
            and d_ca["max_abs_diff"] <= d_ba["max_abs_diff"],
        })
    verdict.update({
        "losses_per_step": losses,
        "s_per_step": {t: info[t]["stats"].get("s_per_step") for t in info},
        "peak_gpu_mem_gb": {t: info[t]["stats"].get("peak_gpu_mem_gb") for t in info},
    })
    write_json(verdict, work / "verdict.json")
    if a.delete_weights:
        for t in runs:
            for f in (work / t).glob("*.safetensors"):
                f.unlink()
    print(json.dumps({k: v for k, v in verdict.items() if k != "D2_mask_audit"}, indent=2))
    ok = verdict["D2_fixed_mask_applied"] if fixed else verdict["D2_mask_is_dead_code"]
    runlog.finish(0 if ok else 1, verdict={k: v for k, v in verdict.items() if k != "D2_mask_audit"})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
