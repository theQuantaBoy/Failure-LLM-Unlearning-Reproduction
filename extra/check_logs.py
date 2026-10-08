# Note: the RunLogs this script reads (logs/ on the Modal Volume) are not shipped in this repository.
"""
extra/check_logs.py — status of every job in a downloaded logs dir, plus the sanity fields of every quant report.
Local, read-only; replaces the ad-hoc one-liners.

    python3 extra/check_logs.py dl_acc3/logs                         # one or more logs dirs
    python3 extra/check_logs.py dl_acc3/logs --reports dl_acc3/qr_*.json dl_acc2/quant_report.json

Per job record (<stamp>_<kind>_<name>.json, written by common.RunLog): stamp, kind, name, status, exit code, minutes,
error. FLAG = status != ok or exit_code != 0, or a .out log without its .json record (still running, or killed before
the record was written: check `modal app list`). Training records also show sure_optimizer_step_calls and the
peak GPU memory; eval records the GPU model (Modal's "A10" request is served by A10 or A10G; bnb4/sampled metrics differ
slightly between them, Reproducibility.md §6) and the output dir. Two eval records with the same output dir are FLAGged
(the later writer overwrites the earlier one's files).
Quant reports (inside quant job records, key 'report', and any --reports file) are checked for:
  weights_bits 4, ignore == ['lm_head'], groups_with_more_than_16_values == 0, dequant_bf16_max_rel_err <= 2^-8,
  quantized_tensors_changed_vs_source_bf16 == 224 (all Linear layers of Llama-2-7B except lm_head),
  non_quantized_tensors_changed == [] for RTN/GPTQ, and only *_layernorm.weight tensors for AWQ (smoothing folds the
  scales into the norms; model.norm.weight is never smoothed).
Exit code 1 if anything is flagged (expected failures, e.g. a refused duplicate pull, are still listed as FLAG).
"""

import argparse
import json
import re
import sys
from pathlib import Path

N_LINEAR = 224  # 32 layers x (q, k, v, o, gate, up, down)
MAX_REL = 2 ** -8 + 1e-12
NORM = re.compile(r"^model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight$")


def check_report(rep: dict) -> list:
    """List of problems of one quant_report.json (empty = all sanity checks pass)."""
    bad = []
    m = rep.get("method")
    if rep.get("weights_bits") != 4:
        bad.append(f"weights_bits {rep.get('weights_bits')}")
    if rep.get("ignore") != ["lm_head"]:
        bad.append(f"ignore {rep.get('ignore')}")
    dq = rep.get("dequant") or {}
    if not dq:
        return bad + ["no dequant section (dequantization did not finish?)"]
    if dq.get("groups_with_more_than_16_values") != 0:
        bad.append(f"groups_with_more_than_16_values {dq.get('groups_with_more_than_16_values')}")
    if (dq.get("dequant_bf16_max_rel_err") or 1) > MAX_REL:
        bad.append(f"dequant_bf16_max_rel_err {dq.get('dequant_bf16_max_rel_err')}")
    if dq.get("quantized_tensors_changed_vs_source_bf16") != N_LINEAR:
        bad.append(f"quantized_tensors_changed {dq.get('quantized_tensors_changed_vs_source_bf16')} != {N_LINEAR}")
    nq = dq.get("non_quantized_tensors_changed")
    names = [x if isinstance(x, str) else x.get("name", str(x)) for x in (nq or [])]
    if m == "awq":
        other = [n for n in names if not NORM.match(n)]
        if other:
            bad.append(f"AWQ changed non-norm tensors: {other[:5]}")
    elif names:
        bad.append(f"{m} changed non-quantized tensors: {names[:5]}")
    return bad


def report_line(rep: dict) -> str:
    dq = rep.get("dequant") or {}
    cal = rep.get("calibration") or {}
    nq = dq.get("non_quantized_tensors_changed") or []
    return (f"{rep.get('method')} g{rep.get('group_size')} calib={cal.get('source', 'none')} "
            f"src={Path(str(rep.get('src', '?'))).name} device={rep.get('device', '?')} "
            f"peak={rep.get('peak_gpu_mem_gib', '?')} GiB quant_min={rep.get('quantize_minutes', '?')} "
            f"norms_changed={len(nq)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="downloaded logs dir(s)")
    ap.add_argument("--reports", nargs="*", default=[], help="standalone quant_report.json files")
    a = ap.parse_args(argv)
    flags = 0
    reports = []
    for logs in map(Path, a.logs):
        print(f"=== {logs} ===")
        recs = sorted(p for p in logs.glob("*.json") if re.match(r"\d{8}-\d{6}_", p.name))
        for p in recs:
            d = json.loads(p.read_text())
            ok = d.get("status") == "ok" and d.get("exit_code") == 0
            extra = ""
            if d.get("kind") == "unlearn":
                c = d.get("counters") or {}
                extra = f" sure_optimizer_step_calls={c.get('sure_optimizer_step_calls', '-')}"
            elif d.get("kind") == "eval":
                extra = (f" gpu={','.join((d.get('gpu') or {}).get('devices') or ['?'])}"
                         f" out={(d.get('config') or {}).get('out_dir', '?')}")
            elif d.get("kind") == "quant" and d.get("report"):
                reports.append((p.name, d["report"]))
            err = f"  error: {str(d.get('error'))[:160]}" if d.get("error") else ""
            print(f"{'ok  ' if ok else 'FLAG'} {p.name[:15]} {d.get('kind', '?'):10s} {d.get('name', '?'):34s} "
                  f"status={d.get('status')} exit={d.get('exit_code')} min={d.get('minutes')}{extra}{err}")
            flags += not ok
        outs = {}
        for p in recs:
            d = json.loads(p.read_text())
            if d.get("kind") == "eval" and d.get("status") == "ok":
                outs.setdefault((d.get("config") or {}).get("out_dir"), []).append(p.name[:15])
        for o, st in outs.items():
            if len(st) > 1:
                print(f"FLAG {len(st)} eval records wrote {o}: {', '.join(st)} (files = the last writer's)")
                flags += 1
        stamps = {p.name[:15] for p in recs}
        for o in sorted(logs.glob("*.out")):
            if o.name[:15] not in stamps and not any(abs(_secs(o.name[:15]) - _secs(s)) <= 2 for s in stamps):
                print(f"FLAG {o.name[:15]} no .json record for {o.name} (running, or killed before the record)")
                flags += 1
    for f in a.reports:
        reports.append((f, json.loads(Path(f).read_text())))
    if reports:
        print("=== quant reports ===")
    for name, rep in reports:
        bad = check_report(rep)
        print(f"{'ok  ' if not bad else 'FLAG'} {name}: {report_line(rep)}" + ("" if not bad else f"  -> {bad}"))
        flags += bool(bad)
    print(f"\n{flags} flagged")
    return 1 if flags else 0


def _secs(stamp: str) -> int:
    """'YYYYMMDD-HHMMSS' -> seconds (only differences matter; record and .out stamps may differ by 1 s)."""
    try:
        d, t = stamp.split("-")
        return int(d) * 86400 + int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])
    except ValueError:
        return -10 ** 9


if __name__ == "__main__":
    sys.exit(main())
