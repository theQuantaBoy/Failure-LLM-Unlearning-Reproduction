"""
compare_results.py — build every results table from saved evaluation outputs (never from published numbers).

    modal volume get failunl-runs results dl_acc1/                      # download (see COMMANDS.md §9)
    .venvs/paper/bin/python compare_results.py --results dl_acc1/results --out tables
    .venvs/paper/bin/python compare_results.py --results dl_acc1/results dl_acc2/results --out tables

Input layout (written by extra/eval_run.py via modal_app.py::evaluate):
    <results>/<corpus>/<model>/<tag>/metrics.json + per-example JSON files
    model: "target" | "retrain" | "<method>_s<train seed>"   (npo_klr, ga_gdr, npo_klr_sure, ga_gdr_sure, *_sure_masked = fixed SURE)
    tag:   "bf16" | "bf16_ep<k>" (intermediate epoch) | "bnb4" (authors' bitsandbytes FP4 path)
           | "rtn_g128_nocalib" (llm-compressor INT4 RTN)
           | "<gptq|awq>_g<32|128>_<general|books_retain>[_s<calib seed>]"

Output (<out>/): tables.md (Task 1, Tasks 2 + 3, Task 4; first table of each task visible, the rest in <details>),
one CSV per table, and tables.json (the same tables, read back by --show without rebuilding):
    .venvs/paper/bin/python compare_results.py --results dl_acc1/results dl_acc2/results --out tables
    .venvs/paper/bin/python compare_results.py --show t23_main t23_recovery --from tables     # keys: --list-tables
n/a = not run. Recovery / usefulness verdicts follow the rules in REPORT.md (Setup) and use the paired bootstrap
of extra/bootstrap_ci.py. Needs sklearn and numpy. Published values are typed from 2410.16454v3 (Table 1 p.6,
Table 3 p.10) and appear only in "Reported" lines. Task 4 dev / held-out halves come from extra/splits.json.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
METRICS8 = ["M1", "M2", "M3", "M4", "Gen", "Tru", "Fac", "Flu"]
KEYMAP = {
    "M1": "verbmem_f",
    "M2": "knowmem_f",
    "M3": "privleak",
    "M4": "knowmem_r",
    "Gen": "gen",
    "Tru": "tru",
    "Fac": "fac",
    "Flu": "flu",
}
EXTRA = {
    "M1g": "verbmem_f_greedy",
    "TruMC1": "tru_mc1",
    "FacEM": "fac_em",
    "AUC": "privleak_auc",
}
AUC_KEY = "forget_holdout_Min-40%"
AUC_RETRAIN = {
    "news": 0.47719999999999996,
    "books": 0.5392999999999999,
}  # constants.py:64,165

# ── published values (typed from the paper; "Quan.(4 bit)" there = the authors' bitsandbytes 4-bit path) ──
PAPER_T1_NEWS = {  # Table 1, NEWS block: M1, M2, M3, M4
    ("target", "bf16"): (58.4, 63.9, -99.8, 55.2),
    ("target", "4bit"): (34.2, 54.4, -99.8, 48.2),
    ("npo_klr", "bf16"): (16.6, 36.6, -94.0, 33.3),
    ("npo_klr", "4bit"): (34.1, 53.7, -99.8, 48.8),
    ("ga_gdr", "bf16"): (0.0, 28.9, 87.1, 34.2),
    ("ga_gdr", "4bit"): (25.0, 50.1, -99.1, 47.7),
}
PAPER_T3_BOOKS = {  # Table 3: M1, M2, M3, M4, Gen, Tru, Fac, Flu
    ("target", "bf16"): (99.8, 59.4, -57.5, 66.9, 28.7, 33.6, 9.1, 573.3),
    ("ga_gdr", "bf16"): (0.0, 2.9, -56.5, 44.2, 22.8, 35.1, 6.7, 563.5),
    ("ga_gdr", "4bit"): (17.9, 33.7, -35.2, 51.9, 21.4, 32.7, 6.0, 553.6),
    ("ga_gdr_sure", "bf16"): (0.0, 0.3, -6.4, 49.3, 29.2, 0.2, 0.0, 544.9),
    ("ga_gdr_sure", "4bit"): (0.0, 4.8, -6.3, 46.2, 30.4, 0.18, 0.0, 524.7),
    ("npo_klr", "bf16"): (22.6, 22.7, -54.9, 50.9, 27.5, 35.0, 7.2, 565.9),
    ("npo_klr", "4bit"): (70.9, 34.2, -60.1, 50.4, 27.0, 34.3, 6.5, 545.6),
    ("npo_klr_sure", "bf16"): (17.6, 37.8, -58.0, 49.4, 23.4, 30.2, 7.4, 588.8),
    ("npo_klr_sure", "4bit"): (16.1, 36.9, -58.9, 34.9, 23.4, 31.1, 8.0, 592.6),
}
PAPER_T1_BOOKS_TARGET_4BIT = (
    85.3,
    36.8,
    -60.1,
    50.5,
)  # Table 1 BOOKS "Target + Quan.(4 bit)" (M1-M4 only)
PAPER_T1_RETRAIN = {
    "news": (20.8, 33.1, 0.0, 55.0),
    "books": (14.3, 28.9, 0.0, 74.5),
}  # Table 1 "Retrain fretrain"


LABEL = {
    "target": "Original target",
    "retrain": "Retrained (reference)",
    "npo_klr": "NPO_KLR",
    "ga_gdr": "GA_GDR",
    "npo_klr_sure": "NPO_KLR + SURE",
    "ga_gdr_sure": "GA_GDR + SURE",
}


# ── loading ───────────────────────────────────────────────────────────────────────────────────────────────
def _model_key(run_dir: Path, corpus: str, model: str) -> str:
    """Runs evaluated before modal_app.default_name handled per-epoch checkpoints were filed under
    'checkpoint-<step>'; file them under their training run (parent of the checkpoint dir in meta.model_dir).
    """
    if not model.startswith("checkpoint-"):
        return model
    parent = Path(
        json.loads((run_dir / "metrics.json").read_text())["meta"]["model_dir"]
    ).parent.name
    return parent.removeprefix(f"{corpus}_")


def load_all(roots: list) -> dict:
    """{(corpus, model, tag): run_dir} over one or more result dirs (e.g. one per Modal account).
    The same (corpus, model, tag) in two dirs is an error unless both metrics.json files are identical.
    """
    out, dup = {}, []
    missing = [str(r) for r in roots if not Path(r).is_dir()]
    if missing:
        raise SystemExit("result dir(s) not found: " + ", ".join(missing))
    for root in roots:
        for m in sorted(Path(root).glob("*/*/*/metrics.json")):
            tag, model, corpus = (
                m.parent.name,
                m.parent.parent.name,
                m.parent.parent.parent.name,
            )
            key = (corpus, _model_key(m.parent, corpus, model), tag)
            if key in out:
                if m.read_bytes() == (out[key] / "metrics.json").read_bytes():
                    continue  # the same run downloaded twice (e.g. results/ and results_all/): keep the first
                dup.append(f"{'/'.join(key)}: {out[key]} and {m.parent}")
            out[key] = m.parent
    if dup:
        raise SystemExit(
            "duplicate runs (same corpus/model/tag in more than one place):\n  "
            + "\n  ".join(dup)
        )
    return out


def metric_values(run_dir: Path) -> dict:
    met = json.loads((run_dir / "metrics.json").read_text())["metrics"]
    vals = {k: met.get(v) for k, v in KEYMAP.items()}
    vals.update({k: met.get(v) for k, v in EXTRA.items()})
    return vals


def _auc(nonmember, member):
    """Same computation as metrics/privleak.py:58-61 (sweep) for split0=forget, split1=holdout."""
    from sklearn.metrics import auc as get_auc, roc_curve as get_roc_curve

    ppl = np.array(list(nonmember) + list(member))
    y = np.array([0] * len(nonmember) + [1] * len(member))
    fpr, tpr, _ = get_roc_curve(y, -ppl)
    return get_auc(fpr, tpr)


def subset_values(run_dir: Path, corpus: str, part: str, splits: dict) -> dict:
    """Metrics recomputed on the dev or held-out half (extra/splits.json), from per-example outputs."""
    S = splits["sets"]

    def pick(setname, records):
        want = set(S[setname][part])
        sha = S[setname]["sha1"]
        for r in records:
            if r["sha1"] != sha[r["idx"]]:
                raise ValueError(
                    f"{run_dir}: example {setname}[{r['idx']}] does not match extra/splits.json"
                )
        return [r for r in records if r["idx"] in want]

    def load(name):
        p = run_dir / name
        return json.loads(p.read_text()) if p.exists() else None

    def mean(xs):
        xs = list(xs)
        return (
            statistics.fmean(xs) if xs else None
        )  # None -> MISSING (e.g. a --limit test run)

    v = {}
    d = load("verbmem_sample.json")
    if d:
        v["M1"] = _x100(
            mean(r["rougeL"] for r in pick(f"{corpus}/verbmem", d["per_example"]))
        )
    d = load("verbmem_greedy.json")
    if d:
        v["M1g"] = _x100(
            mean(r["rougeL"] for r in pick(f"{corpus}/verbmem", d["per_example"]))
        )
    for key, f, s in (
        ("M2", "knowmem_f.json", "knowmem_f"),
        ("M4", "knowmem_r.json", "knowmem_r"),
    ):
        d = load(f)
        if d:
            v[key] = _x100(
                mean(r["rougeL"] for r in pick(f"{corpus}/{s}", d["per_example"]))
            )
    d = load("privleak.json")
    if d:
        fo = pick(f"{corpus}/privleak_forget", d["per_example"]["forget"])
        ho = pick(f"{corpus}/privleak_holdout", d["per_example"]["holdout"])
        if fo and ho:
            a = _auc([r["Min-40%"] for r in fo], [r["Min-40%"] for r in ho])
            v["AUC"] = a
            v["M3"] = (a - AUC_RETRAIN[corpus]) / AUC_RETRAIN[corpus] * 100
    for key, f, s, fld in (
        ("Gen", "mmlu.json", "mmlu", "correct"),
        ("Tru", "truthful.json", "truthful", "MC2"),
        ("TruMC1", "truthful.json", "truthful", "MC1"),
        ("Fac", "triviaqa.json", "triviaqa", "f1"),
        ("FacEM", "triviaqa.json", "triviaqa", "em"),
        ("Flu", "fluency.json", "fluency", "entropy"),
    ):
        d = load(f)
        if d:
            v[key] = _x100(mean(float(r[fld]) for r in pick(s, d["per_example"])))
    return v


def _x100(x):
    return None if x is None else 100 * x


# ── formatting ────────────────────────────────────────────────────────────────────────────────────────────
def fmt(x, nd=1):
    if x is None:
        return "MISSING"
    if isinstance(x, str):
        return x
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.{nd}f}"


SHOWN = (
    {}
)  # table name -> (title, header, rows, group boundaries) for --show (terminal printing)


def write_table(
    out_dir: Path,
    md: list,
    name: str,
    title: str,
    header: list,
    rows: list,
    notes=(),
    heading="##",
    collapsed=False,
):
    SHOWN[name] = (title, header, rows, None)
    with open(out_dir / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    _open(md, title, heading, collapsed)
    md.append(_md_row(header))
    md.append("|" + "---|" * len(header))
    for r in rows:
        md.append(_md_row(fmt(c) if not isinstance(c, str) else c for c in r))
    _close(md, notes, collapsed)


def _get(runs, corpus, model, tag):
    d = runs.get((corpus, model, tag))
    return metric_values(d) if d else None


def _delta(a, b):
    return None if a is None or b is None else a - b


# ── tables ────────────────────────────────────────────────────────────────────────────────────────────────
def _epoch_tags(runs, corpus, model):
    """Intermediate-epoch BF16 evaluations of a training run (tag bf16_ep<k>), sorted by epoch."""
    tags = [
        t
        for (c, m, t) in runs
        if c == corpus and m == model and t.startswith("bf16_ep") and t[7:].isdigit()
    ]
    return sorted(tags, key=lambda t: int(t[7:]))


def _epochs(corpus, method):
    from extra.common import PRESETS

    return PRESETS[f"{corpus}_{method}"]["epochs"]


BLOCK_TABLES = (
    {}
)  # name -> (title, header, blocks, notes) for block tables (Reported / Reproduced / Δ)


def _c(x, bold=False, signed=False):
    """One cell: (text, bold). signed = Δ formatting with an explicit '+'."""
    if (
        isinstance(x, (int, float))
        and not (isinstance(x, float) and math.isnan(x))
        and signed
    ):
        return ("0.0" if abs(x) < 0.05 else f"{x:+.1f}", bold)
    return (fmt(x), bold)


def _md_row(cells):
    """One markdown table row; a literal '|' inside a cell (e.g. 'Δ|M3|') is escaped so it does not split the cell."""
    return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"


def _cell_md(c):
    t, bold = c[0], c[1]
    suf = c[2] if len(c) > 2 else ""
    return (f"**{t}**" if bold and t not in ("MISSING", "nan") else t) + suf


def _emit_blocks(
    out, md, name, title, header, blocks, notes, collapsed=False, heading="###"
):
    """Block table (Reported / Reproduced / Δ ...): markdown (optionally collapsed), CSV, terminal registry."""
    BLOCK_TABLES[name] = (title, header, blocks, notes)
    with open(out / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for bname, lines in blocks:
            for j, (src, cells) in enumerate(lines):
                w.writerow(
                    [
                        bname if j == 0 else "",
                        src,
                        *[c[0] + (c[2] if len(c) > 2 else "") for c in cells],
                    ]
                )
    _open(md, title, heading, collapsed)
    md.append(_md_row(header))
    md.append("|" + "---|" * len(header))
    for bname, lines in blocks:
        for j, (src, cells) in enumerate(lines):
            md.append(
                _md_row([bname if j == 0 else "", src, *[_cell_md(c) for c in cells]])
            )
    _close(md, notes, collapsed)


def _open(md, title, heading, collapsed):
    if collapsed:
        md.append(f"<details>\n<summary><b>{title}</b></summary>\n")
    else:
        md.append(f"{heading} {title}\n")


def _close(md, notes, collapsed):
    if notes:
        md.append("")
        for n in notes:
            md.append(n if n.startswith("- ") else f"\n{n}")
    md.append("")
    if collapsed:
        md.append("</details>\n")


# ── Task 1 — NEWS ─────────────────────────────────────────────────────────────────────────────────────────
# Table 1a = the main Task 1 table (visible); 1b–1d = supporting tables (collapsed in tables.md).
# Main-table rows: (label, evaluation precision, model kind, our tag). RTN-INT4 = llm-compressor W4A16 g128.
TASK1_ROWS = [
    ("Original target", "BF16", "target", "bf16"),
    ("Original target + RTN", "INT4", "target", "rtn_g128_nocalib"),
    ("NPO_KLR", "BF16", "npo_klr", "bf16"),
    ("NPO_KLR + RTN", "INT4", "npo_klr", "rtn_g128_nocalib"),
    ("GA_GDR", "BF16", "ga_gdr", "bf16"),
    ("GA_GDR + RTN", "INT4", "ga_gdr", "rtn_g128_nocalib"),
]
M14 = ["M1", "M2", "M3", "M4"]
H4 = ["M1 ↓", "M2 ↓", "M3 → 0", "M4 ↑"]
RTN_NOTE = "INT4 RTN: llm-compressor 0.14.0, W4A16, symmetric, group size 128, `lm_head` not quantized."
SCALE_NOTE = "All values ×100, as in the paper (e.g. ROUGE 0.435 → 43.5)."


def _t1_model(kind, seed):
    return "target" if kind == "target" else f"{kind}_s{seed}"


def _m1_cell(v, bold):
    """M1 cell: sampled value (bold if requested) with the greedy value in parentheses."""
    if v is None:
        return _c(None)
    g = v.get("M1g")
    return (fmt(v["M1"]), bold, f" ({fmt(g)})" if g is not None else "")


def _reproduced_line(v, bold=True, label="Reproduced"):
    if v is None:
        return (label, [_c(None)] * 4)
    return (label, [_m1_cell(v, bold)] + [_c(v[m], bold=bold) for m in M14[1:]])


def _delta_cells(v, paper, dagger_m1=False):
    cells = []
    for i, m in enumerate(M14):
        a = v[m] if v else None
        d = (
            _delta(a, paper[i])
            if not (isinstance(a, float) and math.isnan(a))
            else None
        )
        txt = (
            _c(d, signed=True)[0]
            if d is not None
            else ("nan" if isinstance(a, float) else "MISSING")
        )
        cells.append(
            (txt + ("†" if i == 0 and dagger_m1 and d is not None else ""), False)
        )
    return ("Δ", cells)


def task1_taskmd(runs, md, out, seed):
    """Table 1a — the main Task 1 table: measured values only."""
    md.append("# Task 1 — NEWS\n")
    rows = []
    for label, prec, kind, tag in TASK1_ROWS:
        v = _get(runs, "news", _t1_model(kind, seed), tag)
        rows.append([label, prec, *[v[m] if v else None for m in M14]])
    r = _get(runs, "news", "retrain", "bf16")
    retrain_m3 = fmt(r["M3"]) if r else "MISSING"
    write_table(
        out,
        md,
        "task1_taskmd",
        "Table 1a — Task 1 results (main table)",
        ["Model / method", "Evaluation precision", *H4],
        rows,
        notes=[
            "- Our measurements only (no published values); "
            + SCALE_NOTE[0].lower()
            + SCALE_NOTE[1:],
            "- BF16 for training and evaluation.",
            "- M1: sampled VerbMem (T = 0.9, fixed seed), as in the authors' code.",
            "- " + RTN_NOTE,
            f"- M3 reference: retrained NEWS model (AUC {AUC_RETRAIN['news']:.4f}; its own M3 = {retrain_m3}).",
            f"- Training seed {seed}; final checkpoint (epoch {_epochs('news', 'npo_klr')}).",
            "- Comparison with the paper: Table 1b.",
        ],
        heading="##",
    )
    SHOWN["task1_taskmd"] = (
        *SHOWN["task1_taskmd"][:3],
        [0],
    )  # terminal: no separator between rows


def task1_compare(runs, md, out, seed):
    """Table 1b — Reported (paper Table 1) / Reproduced (bold, = Table 1a) / Δ for every main-table row."""
    blocks = []
    for label, prec, kind, tag in TASK1_ROWS:
        quant = tag != "bf16"
        paper = list(PAPER_T1_NEWS[(kind, "4bit" if quant else "bf16")])
        v = _get(runs, "news", _t1_model(kind, seed), tag)
        blocks.append(
            (
                f"{label} · {prec}",
                [
                    ("Reported (Table 1)", [_c(p) for p in paper]),
                    _reproduced_line(v),
                    _delta_cells(v, paper, dagger_m1=(kind == "target" and not quant)),
                ],
            )
        )
    t = _get(runs, "news", "target", "bf16")
    g = fmt(t["M1g"]) if t else "MISSING"
    notes = [
        "- Reported = paper Table 1 (NEWS); Reproduced = our measurements (bold, = Table 1a); "
        "Δ = Reproduced − Reported.",
        "- M1: sampled (T = 0.9, fixed seed); greedy in parentheses.",
        f"- † The reported target M1 appears to be MUSE's greedy value (the row equals MUSE Table 3); "
        f"our greedy M1 is {g}.",
        "- " + RTN_NOTE,
        "- The reported 4-bit rows probably come from the authors' bitsandbytes FP4 path; see Table 1c.",
    ]
    _emit_blocks(
        out,
        md,
        "task1_compare",
        "Table 1b — Comparison with the paper (Table 1, NEWS)",
        ["Row", "Source", *H4],
        blocks,
        notes,
        collapsed=True,
    )


def task1_quant(runs, md, out, seed):
    """Table 1c — the authors' 4-bit path (bitsandbytes FP4) next to RTN-INT4, both vs the reported 4-bit row."""
    blocks = []
    for kind in ("target", "npo_klr", "ga_gdr"):
        paper = list(PAPER_T1_NEWS[(kind, "4bit")])
        model = _t1_model(kind, seed)
        b = _get(runs, "news", model, "bnb4")
        r = _get(runs, "news", model, "rtn_g128_nocalib")
        blocks.append(
            (
                f"{LABEL[kind]} · 4-bit",
                [
                    ("Reported (Table 1)", [_c(p) for p in paper]),
                    _reproduced_line(
                        b, bold=False, label="bitsandbytes FP4 (authors' code)"
                    ),
                    _delta_cells(b, paper),
                    _reproduced_line(r, bold=False, label="RTN-INT4 (llm-compressor)"),
                    _delta_cells(r, paper),
                ],
            )
        )
    notes = [
        "- The paper describes uniform integer quantization (Eq. 2) and calls its 4-bit models RTN (App. E).",
        "- The only 4-bit path in the code is `BitsAndBytesConfig(load_in_4bit=True)` (utils.py:94-101), whose "
        "transformers default is FP4: non-uniform, block size 64.",
        "- " + RTN_NOTE,
        "- M1: sampled; greedy in parentheses. Δ = measured − Reported.",
        "- Which path produced the reported 4-bit rows cannot be decided from these numbers.",
    ]
    _emit_blocks(
        out,
        md,
        "task1_quant",
        "Table 1c — 4-bit path: bitsandbytes FP4 vs RTN-INT4 (NEWS)",
        ["Row", "Source", *H4],
        blocks,
        notes,
        collapsed=True,
    )


def task1_epochs(runs, md, out, seed):
    """Table 1d — BF16 evaluations of intermediate and final checkpoints (NEWS)."""
    rows = []
    for kind in ("npo_klr", "ga_gdr"):
        model = _t1_model(kind, seed)
        n = _epochs("news", kind)
        tags = _epoch_tags(runs, "news", model)
        if not tags:
            continue
        for tag in tags + ["bf16"]:
            v = _get(runs, "news", model, tag)
            ep = tag[7:] if tag != "bf16" else f"{n} (final)"
            m1 = None if v is None else f"{fmt(v['M1'])} ({fmt(v['M1g'])})"
            rows.append(
                [
                    LABEL[kind],
                    f"{ep} of {n}" if tag != "bf16" else ep,
                    m1,
                    *[v[m] if v else None for m in M14[1:]],
                ]
            )
    if not rows:
        return
    write_table(
        out,
        md,
        "task1_epochs",
        "Table 1d — BF16 checkpoints by epoch (NEWS)",
        ["Method", "Epoch", "M1 ↓ (greedy)", "M2 ↓", "M3 → 0", "M4 ↑"],
        rows,
        notes=[
            "- The paper reports only the final checkpoint.",
            "- M1: sampled; greedy in parentheses.",
        ],
        heading="###",
        collapsed=True,
    )


def _dw(s):
    """Terminal display width (wcwidth rules): wide/fullwidth = 2, combining marks and variation selectors = 0."""
    import unicodedata as u

    return sum(
        (
            0
            if (u.combining(ch) or 0xFE00 <= ord(ch) <= 0xFE0F or ch == "\u200d")
            else 2 if u.east_asian_width(ch) in "WF" else 1
        )
        for ch in str(s)
    )


def _wide_gap(s):
    """Terminal only: a space after an emoji+VS16 (e.g. ⚠️) so terminals that draw it 2 cells wide but advance 1
    do not paint over the next character."""
    import re

    return re.sub("(\ufe0f)(?=\\S)", "\\1 ", str(s))


def _ljust(s, n):
    return str(s) + " " * max(0, n - _dw(s))


def _rjust(s, n):
    return " " * max(0, n - _dw(s)) + str(s)


def render_blocks(title, header, blocks, notes, color):
    """Terminal view of BLOCK_TABLES: separator between blocks, numbers right-aligned, bold via ANSI if color."""
    rows = [
        [name if j == 0 else "", src, *cells]
        for name, lines in blocks
        for j, (src, cells) in enumerate(lines)
    ]
    text = lambda c: (
        (c[0] + (c[2] if len(c) > 2 else "")) if isinstance(c, tuple) else c
    )  # noqa: E731
    w = [max([_dw(h)] + [_dw(text(r[i])) for r in rows]) for i, h in enumerate(header)]

    def fmt_row(r):
        out = []
        for i, c in enumerate(r):
            t = text(c)
            t = _ljust(t, w[i]) if i < 2 else _rjust(t, w[i])
            if color and isinstance(c, tuple) and c[1]:
                suf = c[2] if len(c) > 2 else ""
                core = t[: len(t) - len(suf)] if suf and t.endswith(suf) else t
                t = f"\033[1m{core}\033[0m" + (t[len(core) :])
            out.append(t)
        return " | ".join(out)

    sep = "-+-".join("-" * x for x in w)
    lines = [
        f"=== TABLE {_wide_gap(title)} ===",
        fmt_row(header),
        sep.replace("-", "="),
    ]
    for k, (name, blk) in enumerate(blocks):
        if k:
            lines.append(sep)
        start = sum(len(b) for _, b in blocks[:k])
        lines += [fmt_row(r) for r in rows[start : start + len(blk)]]
    if not color:
        lines.append("(bold not shown: output is not a terminal or --no-color)")
    return "\n".join(lines + [""] + notes)


# ── Tasks 2 + 3 — BOOKS ───────────────────────────────────────────────────────────────────────────────────
# Table 2a = the combined main Task 2+3 table with the fixed SURE (visible); 2b–2i collapsed.
# SURE variants: "fixed" = extra/fixed_unlearn.py presets *_sure_masked (deviation), "released" = authors' code.
FIXED, RELEASED = "✅", "⚠️"
NA = "n/a"
H8 = ["M1 ↓", "M2 ↓", "M3 → 0", "M4 ↑", "Gen ↑", "Tru ↑", "Fac ↑", "Flu ↑"]
T23_METHODS = ["target", "npo_klr", "npo_klr_sure", "ga_gdr", "ga_gdr_sure"]
T23_TAGS = [
    ("BF16", "bf16"),
    ("INT4 / RTN", "rtn_g128_nocalib"),
    ("INT4 / GPTQ", "gptq_g128_general"),
    ("INT4 / AWQ", "awq_g128_general"),
]
QUANT_LABEL = {
    "bnb4": "bnb-FP4",
    "rtn_g128_nocalib": "RTN",
    "gptq_g128_general": "GPTQ",
    "awq_g128_general": "AWQ",
}
ANALYSIS_TAGS = ["bnb4", "rtn_g128_nocalib", "gptq_g128_general", "awq_g128_general"]
LABEL.update(
    {"npo_klr_sure_masked": "NPO_KLR + SURE", "ga_gdr_sure_masked": "GA_GDR + SURE"}
)
QUANT_NOTE = (
    "INT4 (RTN, GPTQ, AWQ): llm-compressor 0.14.0, W4A16, group size 128, `lm_head` not quantized; RTN and GPTQ "
    "symmetric, AWQ asymmetric; GPTQ/AWQ calibration: 128 × 2048 tokens of WikiText-2 (seed 0)."
)
METRIC_NOTE = (
    "Gen = MMLU accuracy, Tru = TruthfulQA MC2, Fac = TriviaQA F1, Flu = fluency (entropy); "
    "Tru = nan: undefined for a collapsed model."
)


def _sure_model(kind, variant, seed):
    """Model key for a method; SURE methods depend on the variant (fixed / released)."""
    if kind == "target":
        return "target"
    if kind.endswith("_sure") and variant == "fixed":
        return f"{kind}_masked_s{seed}"
    return f"{kind}_s{seed}"


def _method_label(kind, variant=None):
    base = LABEL[kind.removesuffix("_masked")]
    if kind.endswith("_masked"):
        return f"{base} {FIXED}"
    if kind.endswith("_sure") and variant:
        return f"{base} {FIXED if variant == 'fixed' else RELEASED}"
    return base


def _vals8(v):
    if v is None:
        return [NA] * 8
    return [v[m] for m in METRICS8]


def tasks23_header(md):
    md.append("# Tasks 2 + 3 — BOOKS\n")
    md.append(
        f"> {FIXED} = SURE, the released code with its two bugs fixed (row-level mask recomputed every step, as the "
        "code intends; forget gradient counted once; a deviation; not the paper's module-level mask fixed at θo) · "
        f"{RELEASED} = SURE as released (mask never applied) · **{NA}** = not run (outside this project's compute budget; "
        "the collapsed GA_GDR + SURE " + FIXED + " model was not quantized).\n"
    )


def _t23_table(runs, md, out, seed, variant, name, title, collapsed, extra_note):
    rows = []
    for kind in T23_METHODS:
        model = _sure_model(kind, variant, seed)
        for prec, tag in T23_TAGS:
            rows.append(
                [
                    _method_label(kind, variant),
                    prec,
                    *_vals8(_get(runs, "books", model, tag)),
                ]
            )
    r = _get(runs, "books", "retrain", "bf16")
    if r is not None:
        rows.append(
            [LABEL["retrain"], "BF16", *[r[m] if m in M14 else "—" for m in METRICS8]]
        )
    write_table(
        out,
        md,
        name,
        title,
        ["Method", "Precision / quantizer", *H8],
        rows,
        notes=[
            extra_note,
            "- Target and base methods: authors' code as released.",
            "- BF16 for training and evaluation; M1 sampled (T = 0.9, fixed seed); "
            + SCALE_NOTE[0].lower()
            + SCALE_NOTE[1:],
            "- " + QUANT_NOTE,
            "- GPTQ / AWQ rows are new experiments (no published counterpart).",
            "- " + METRIC_NOTE,
            f"- Training seed {seed}; final checkpoint (epoch {_epochs('books', 'npo_klr')}).",
            "- Retrained (reference): the MUSE BOOKS model never trained on the forget set; M1–M4 only.",
        ],
        heading="##",
        collapsed=collapsed,
    )
    SHOWN[name] = (*SHOWN[name][:3], [i for i in range(0, len(rows), len(T23_TAGS))])


def t23_main(runs, md, out, seed):
    """Table 2a — the combined main Task 2+3 table, with the fixed SURE."""
    tasks23_header(md)
    _t23_table(
        runs,
        md,
        out,
        seed,
        "fixed",
        "t23_main",
        f"Table 2a — Tasks 2 + 3 results (main table, {FIXED} SURE: released code, bugs fixed)",
        False,
        f"- SURE rows {FIXED}: the released code with its two bugs fixed (deviation; FINDINGS §6). "
        "The same table with the released SURE: Table 2b.",
    )


def t23_released(runs, md, out, seed):
    """Table 2b — the same table with the released (unfixed) SURE."""
    _t23_table(
        runs,
        md,
        out,
        seed,
        "released",
        "t23_released",
        f"Table 2b — Tasks 2 + 3 results with the released SURE ({RELEASED})",
        False,
        f"- SURE rows {RELEASED}: authors' code as released; its saliency mask is never applied.",
    )


T2_COMPARE_ROWS = [("target", "bf16")] + [
    (k, t)
    for k in ("npo_klr", "npo_klr_sure", "ga_gdr", "ga_gdr_sure")
    for t in ("bf16", "rtn_g128_nocalib")
]


def _repro8(v, bold=True, label="Reproduced"):
    if v is None:
        return (label, [(NA, False)] * 8)
    return (label, [_m1_cell(v, bold)] + [_c(v[m], bold=bold) for m in METRICS8[1:]])


def _delta8(v, paper, dagger_m1=False):
    cells = []
    for i, m in enumerate(METRICS8):
        a = v[m] if v else None
        if a is None or paper[i] is None:
            cells.append((NA if v is None else "—", False))
            continue
        if isinstance(a, float) and math.isnan(a):
            cells.append(("nan", False))
            continue
        cells.append(
            (
                _c(a - paper[i], signed=True)[0]
                + ("†" if i == 0 and dagger_m1 else ""),
                False,
            )
        )
    return ("Δ", cells)


def _compare_blocks(runs, seed, variant, kinds):
    blocks = []
    for kind, tag in T2_COMPARE_ROWS:
        if kind not in kinds:
            continue
        quant = tag != "bf16"
        paper = list(PAPER_T3_BOOKS[(kind, "4bit" if quant else "bf16")])
        v = _get(runs, "books", _sure_model(kind, variant, seed), tag)
        blocks.append(
            (
                f"{_method_label(kind, variant)} · {'INT4 RTN' if quant else 'BF16'}",
                [
                    ("Reported (Table 3)", [_c(p) for p in paper]),
                    _repro8(v),
                    _delta8(v, paper, dagger_m1=(kind == "target")),
                ],
            )
        )
    return blocks


def _compare_notes(runs, extra, dagger=True):
    t = _get(runs, "books", "target", "bf16")
    g = fmt(t["M1g"]) if t else NA
    return (
        extra
        + [
            "- Reported = paper Table 3; Reproduced = our measurements (bold); Δ = Reproduced − Reported.",
            "- M1: sampled (T = 0.9, fixed seed); greedy in parentheses.",
        ]
        + (
            [
                f"- † The reported target M1–M4 appear to be copied from MUSE (greedy M1); our greedy M1 is {g}."
            ]
            if dagger
            else []
        )
        + [
            "- Reported 4-bit rows probably use the authors' bitsandbytes FP4 path; Reproduced uses RTN (Table 2h).",
            "- Reported Tru for GA_GDR + SURE (0.2 / 0.18) is off the ×100 scale used everywhere else.",
            "- " + METRIC_NOTE,
        ]
    )


def t2_compare_fixed(runs, md, out, seed):
    """Table 2c — comparison with Table 3, fixed SURE."""
    blocks = _compare_blocks(runs, seed, "fixed", {k for k, _ in T2_COMPARE_ROWS})
    notes = _compare_notes(
        runs, [f"- SURE rows {FIXED}: fixed SURE (deviation); released SURE: Table 2d."]
    )
    _emit_blocks(
        out,
        md,
        "t2_compare_fixed",
        f"Table 2c — Comparison with the paper (Table 3), {FIXED} fixed SURE",
        ["Row", "Source", *H8],
        blocks,
        notes,
        collapsed=True,
    )


def t2_compare_released(runs, md, out, seed):
    """Table 2d — comparison with Table 3, released SURE (only the SURE rows; other rows as in Table 2c)."""
    blocks = _compare_blocks(runs, seed, "released", {"npo_klr_sure", "ga_gdr_sure"})
    notes = _compare_notes(
        runs,
        [
            f"- SURE as released ({RELEASED}); target and base-method rows are as in Table 2c."
        ],
        dagger=False,
    )
    _emit_blocks(
        out,
        md,
        "t2_compare_released",
        f"Table 2d — Comparison with the paper (Table 3), {RELEASED} released SURE",
        ["Row", "Source", *H8],
        blocks,
        notes,
        collapsed=True,
    )


def t23_seeds(runs, md, out, seed):
    """Table 2e — every configuration evaluated with two training seeds, released and fixed SURE."""
    blocks = []
    for kind in ANALYSIS_KINDS:
        seeds = [sd for _, sd in _models(runs, "books", kind)]
        if len(seeds) < 2:
            continue
        tags = ["bf16", *ANALYSIS_TAGS]
        for tag in tags:
            vs = [(sd, _get(runs, "books", f"{kind}_s{sd}", tag)) for sd in seeds]
            vs = [(sd, v) for sd, v in vs if v is not None]
            if len(vs) < 2:
                continue
            lines = [
                (f"seed {sd}", [_m1_cell(v, False)] + [_c(v[m]) for m in METRICS8[1:]])
                for sd, v in vs
            ]
            (s0, a), (s1, b) = vs[0], vs[-1]
            lines.append(
                (
                    f"Δ (seed {s1} − {s0})",
                    [_c(_delta(b[m], a[m]), signed=True) for m in METRICS8],
                )
            )
            label = _method_label(kind, "released" if kind.endswith("_sure") else None)
            blocks.append((f"{label} · {QUANT_LABEL.get(tag, 'BF16')}", lines))
    if not blocks:
        return
    notes = [
        "- Only configurations evaluated with both training seeds are shown.",
        "- M1: sampled; greedy in parentheses.",
        "- With two seeds, the difference is the only honest summary of variability.",
    ]
    _emit_blocks(
        out,
        md,
        "t23_seeds",
        "Table 2e — Seed repeats (training seeds 42 and 43)",
        ["Model · precision", "Run", *H8],
        blocks,
        notes,
        collapsed=True,
    )


def _na(x):
    return NA if x is None else x


def _ci(bt, k):
    if k not in bt:
        return NA
    d, lo, hi = bt[k]
    return f"{d:+.1f} [{lo:.1f}, {hi:.1f}]"


ANALYSIS_KINDS = [
    "npo_klr",
    "npo_klr_sure",
    "npo_klr_sure_masked",
    "ga_gdr",
    "ga_gdr_sure",
    "ga_gdr_sure_masked",
]


def _analysis_rows(runs):
    """(kind, seed, tag, v, base, target-under-same-quantizer, bootstrap) for every quantized unlearned model."""
    out = []
    for kind in ANALYSIS_KINDS:
        for model, sd in _models(runs, "books", kind):
            base = _get(runs, "books", model, "bf16")
            for tag in ANALYSIS_TAGS:
                v = _get(runs, "books", model, tag)
                if base is None or (v is None and tag == "bnb4"):
                    continue  # bnb-FP4 was run for seed 42 only; other missing runs are kept as n/a rows
                t = _get(runs, "books", "target", tag)
                out.append(
                    (
                        kind,
                        sd,
                        tag,
                        v,
                        base,
                        t,
                        _boot(runs, "books", model, tag) if v else {},
                    )
                )
    return out


def t23_recovery(runs, md, out, seed):
    """Table 2f — recovery of forgotten information and privacy change after quantization."""
    rows = []
    for kind, sd, tag, v, base, t, bt in _analysis_rows(runs):
        label = _method_label(kind, "released" if kind.endswith("_sure") else None)
        if v is None:
            rows.append([label, str(sd), QUANT_LABEL[tag], *[NA] * 8])
            continue
        per = recovery_flags(runs, "books", kind, tag)
        rows.append(
            [
                _method_label(kind, "released" if kind.endswith("_sure") else None),
                str(sd),
                QUANT_LABEL[tag],
                _ci(bt, "M1"),
                _ci(bt, "M2"),
                _ci(bt, "absM3"),
                _delta(v["M1"], t["M1"]) if t else NA,
                _delta(v["M2"], t["M2"]) if t else NA,
                _na(_frac(v, base, t, "M1")),
                _na(_frac(v, base, t, "M2")),
                recovery_verdict(per, sd) if sd in per else NA,
            ]
        )
    write_table(
        out,
        md,
        "t23_recovery",
        "Table 2f — Recovery and privacy after quantization (vs own BF16 and vs the quantized target)",
        [
            "Method",
            "Seed",
            "Quantizer",
            "ΔM1 [95% CI]",
            "ΔM2 [95% CI]",
            "Δ|M3| [95% CI]",
            "M1 − target(q)",
            "M2 − target(q)",
            "M1 recovered %",
            "M2 recovered %",
            "Recovery",
        ],
        rows,
        notes=[
            "- Δ = quantized − own BF16 checkpoint; positive ΔM1 / ΔM2 = recovered information; positive Δ|M3| = "
            "worse privacy.",
            "- 95% CI: paired bootstrap over evaluation examples (10,000 resamples); it does not cover training "
            "randomness.",
            "- target(q) = original target under the same quantizer.",
            "- Recovered % = (M_q − M_BF16) / (M_target,q − M_BF16) × 100: share of the gap to the quantized "
            "target that quantization closes.",
            f"- Recovery (rule set before choosing failure cases): ΔM1 ≥ {RECOVERY_M1:.0f} or "
            f"ΔM2 ≥ {RECOVERY_M2:.0f} with the CI above 0, in every evaluated seed.",
            "- bnb-FP4 = the authors' 4-bit path; RTN / GPTQ / AWQ as in Table 2a.",
        ],
        heading="###",
        collapsed=True,
    )


def t23_utility(runs, md, out, seed):
    """Table 2g — utility change after quantization and the usefulness gate."""
    rows = []
    for kind, sd, tag, v, base, t, bt in _analysis_rows(runs):
        if v is None:
            rows.append(
                [
                    _method_label(kind, "released" if kind.endswith("_sure") else None),
                    str(sd),
                    QUANT_LABEL[tag],
                    *[NA] * 10,
                ]
            )
            continue
        ratio = lambda m: (
            (100 * v[m] / t[m]) if t and v[m] is not None and t[m] else NA
        )  # noqa: E731
        rows.append(
            [
                _method_label(kind, "released" if kind.endswith("_sure") else None),
                str(sd),
                QUANT_LABEL[tag],
                _ci(bt, "M4"),
                _delta(v["Gen"], base["Gen"]),
                _delta(v["Tru"], base["Tru"]),
                _delta(v["Fac"], base["Fac"]),
                _delta(v["Flu"], base["Flu"]),
                ratio("M4"),
                ratio("Flu"),
                useful(v, t),
                f"{useful(v, t, USEFUL_M4_SENS[0])} / {useful(v, t, USEFUL_M4_SENS[1])}",
                useful(v, t, f_flu=USEFUL_FLU_SENS),
            ]
        )
    write_table(
        out,
        md,
        "t23_utility",
        "Table 2g — Utility after quantization and usefulness",
        [
            "Method",
            "Seed",
            "Quantizer",
            "ΔM4 [95% CI]",
            "ΔGen",
            "ΔTru",
            "ΔFac",
            "ΔFlu",
            "M4 / target(q) %",
            "Flu / target(q) %",
            "Useful",
            "Useful (M4 0.6 / 0.9)",
            "Useful (Flu 0.8)",
        ],
        rows,
        notes=[
            "- Δ = quantized − own BF16 checkpoint.",
            f"- Useful (rule set before choosing failure cases): M4 ≥ {USEFUL_M4} × M4(target, same "
            f"quantizer), Flu ≥ {USEFUL_FLU} × Flu(target, same quantizer), and Fac > 0.",
            "- Sensitivity: M4 factor 0.6 / 0.9 instead of 0.75; Flu factor 0.8 instead of 0.9. Gen and Tru are "
            "reported, not gating.",
            "- " + METRIC_NOTE,
        ],
        heading="###",
        collapsed=True,
    )


def t2_truthful_mc1(runs, md, out, seed):
    """Table 2j — TruthfulQA MC1 (the paper's stated metric) next to MC2 (what the published values match)."""
    rows, seen = [], set()
    for variant in ("fixed", "released"):
        for kind in T23_METHODS:
            model = _sure_model(kind, variant, seed)
            for prec, tag in T23_TAGS:
                if (model, tag) in seen:
                    continue
                seen.add((model, tag))
                v = _get(runs, "books", model, tag)
                rows.append(
                    [
                        _method_label(kind, variant),
                        prec,
                        NA if v is None else v.get("TruMC1"),
                        NA if v is None else v["Tru"],
                    ]
                )
    write_table(
        out,
        md,
        "t2_truthful_mc1",
        "Table 2j — TruthfulQA: MC1 (paper text) vs MC2 (reported in all other tables)",
        ["Method", "Precision / quantizer", "Tru MC1", "Tru MC2"],
        rows,
        notes=[
            "- The paper's text names MC1; its published Tru values match MC2, which the other tables report.",
            "- MC2 = nan for a collapsed model (0/0 in the authors' code); MC1 is always defined.",
        ],
        heading="###",
        collapsed=True,
    )


def t2_quant_path(runs, md, out, seed):
    """Table 2h — bitsandbytes FP4 (authors' 4-bit path) vs RTN-INT4, against the reported 4-bit rows."""
    blocks = []
    for kind, variant in (
        ("target", None),
        ("npo_klr", None),
        ("npo_klr_sure", "fixed"),
        ("npo_klr_sure", "released"),
        ("ga_gdr", None),
        ("ga_gdr_sure", "released"),
    ):
        if kind == "target":
            paper = list(PAPER_T1_BOOKS_TARGET_4BIT) + [None] * 4
            src = "Reported (Table 1)"
        else:
            paper = list(PAPER_T3_BOOKS[(kind, "4bit")])
            src = "Reported (Table 3)"
        model = _sure_model(kind, variant, seed)
        b = _get(runs, "books", model, "bnb4")
        r = _get(runs, "books", model, "rtn_g128_nocalib")
        blocks.append(
            (
                f"{_method_label(kind, variant)} · 4-bit",
                [
                    (src, [_c(p) if p is not None else ("—", False) for p in paper]),
                    _repro8(b, bold=False, label="bitsandbytes FP4 (authors' code)"),
                    _delta8(b, paper),
                    _repro8(r, bold=False, label="RTN-INT4 (llm-compressor)"),
                    _delta8(r, paper),
                ],
            )
        )
    notes = [
        "- bitsandbytes FP4 = `BitsAndBytesConfig(load_in_4bit=True)` (utils.py:94-101): non-uniform, block 64.",
        "- RTN-INT4: llm-compressor, W4A16, symmetric, group 128.",
        "- Reported target 4-bit row from Table 1 (M1–M4 only; Table 3 has no such row).",
        "- M1: sampled; greedy in parentheses. Δ = measured − Reported.",
    ]
    _emit_blocks(
        out,
        md,
        "t2_quant_path",
        "Table 2h — 4-bit path: bitsandbytes FP4 vs RTN-INT4 (BOOKS)",
        ["Row", "Source", *H8],
        blocks,
        notes,
        collapsed=True,
    )


# Recovery / usefulness definitions (written after seeing the seed-42 Task 3/4 results and before the seed-43
# GPTQ/AWQ results)
USEFUL_M4, USEFUL_M4_SENS, USEFUL_FLU, USEFUL_FLU_SENS = 0.75, (0.6, 0.9), 0.9, 0.8
RECOVERY_M1, RECOVERY_M2 = 10.0, 5.0


def useful(v, t, f_m4=USEFUL_M4, f_flu=None):
    """'y'/'n' : M4 >= f·M4(target, same quantizer) and Flu >= 0.9·Flu(target, same q) and Fac > 0 (Gen/Tru not gating)."""
    if v is None or t is None:
        return NA
    need = (v["M4"], v["Flu"], v["Fac"], t["M4"], t["Flu"])
    if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in need):
        return NA
    ok = (
        v["M4"] >= f_m4 * t["M4"]
        and v["Flu"] >= (USEFUL_FLU if f_flu is None else f_flu) * t["Flu"]
        and v["Fac"] > 0
    )
    return "y" if ok else "n"


def _frac(v, base, t, m):
    """Recovery fraction (M_q − M_bf16) / (M_target,q − M_bf16) in %; None if undefined."""
    if not (v and base and t) or None in (v[m], base[m], t[m]) or t[m] - base[m] <= 0:
        return None
    return 100 * (v[m] - base[m]) / (t[m] - base[m])


def _models(runs, corpus, method):
    """Training runs of one method present in runs, as (model key, seed), sorted by seed."""
    out = {
        (m, int(m.rpartition("_s")[2]))
        for (c, m, t) in runs
        if c == corpus
        and m.rpartition("_s")[0] == method
        and m.rpartition("_s")[2].isdigit()
    }
    return sorted(out, key=lambda x: x[1])


def _boot(runs, corpus, model, tag):
    d, b = runs.get((corpus, model, tag)), runs.get((corpus, model, "bf16"))
    if d is None or b is None:
        return {}
    from extra.bootstrap_ci import paired_deltas

    key = (str(d), str(b))
    if key not in BOOT:
        BOOT[key] = paired_deltas(d, b, corpus)
    return BOOT[key]


BOOT = {}  # (run dir, bf16 dir) -> paired bootstrap result, computed once per run


def recovery_flags(runs, corpus, method, tag):
    """{seed: 'y'/'n'} per training seed (recovery rule, without the cross-seed condition) and the combined verdict."""
    per = {}
    for model, sd in _models(runs, corpus, method):
        bt = _boot(runs, corpus, model, tag)
        if not bt:
            continue
        hit = any(
            k in bt and bt[k][0] >= thr and bt[k][1] > 0
            for k, thr in (("M1", RECOVERY_M1), ("M2", RECOVERY_M2))
        )
        per[sd] = (hit, bt.get("M1", (None,))[0], bt.get("M2", (None,))[0])
    return per


def recovery_verdict(per, sd):
    hit = per[sd][0]
    if not hit:
        return "n"
    if len(per) < 2:
        return "y (1 seed)"
    same = all(h for h, _, _ in per.values())
    return f"y ({len(per)} seeds)" if same else "n (not replicated across seeds)"


# ── Task 4 — SURE under GPTQ / AWQ configurations (BOOKS) ────────────────────────────────────────────────
T4_CONFIGS = [
    (q, calib, gs)
    for q in ("gptq", "awq")
    for calib in ("general", "books_retain")
    for gs in (32, 128)
]
CALIB_LABEL = {"general": "General text", "books_retain": "BOOKS retain-only"}


def _t4_tag(q, calib, gs):
    return f"{q}_g{gs}_{calib}"


def task4_main(runs, md, out, seed):
    """Table 4a — every Task 4 configuration, released SURE (⚠️), with the target under the same configuration."""
    md.append("# Task 4 — SURE under GPTQ / AWQ configurations (BOOKS)\n")
    rows, starts = [], []
    for kind in ("npo_klr_sure", "ga_gdr_sure", "target"):
        model = _sure_model(kind, "released", seed)
        label = LABEL[kind] if kind == "target" else _method_label(kind, "released")
        starts.append(len(rows))
        if kind != "target":
            rows.append(
                [label, "BF16", "—", "—", *_vals8(_get(runs, "books", model, "bf16"))]
            )
        for q, calib, gs in T4_CONFIGS:
            v = _get(runs, "books", model, _t4_tag(q, calib, gs))
            rows.append([label, q.upper(), CALIB_LABEL[calib], str(gs), *_vals8(v)])
    write_table(
        out,
        md,
        "task4_main",
        f"Table 4a — Task 4 results ({RELEASED} released SURE)",
        ["Model", "Quantizer", "Calibration", "Group size", *H8],
        rows,
        notes=[
            f"- SURE rows {RELEASED}: authors' code as released (mask never applied); training seed {seed}.",
            f"- {FIXED} fixed SURE: only GPTQ / AWQ g128 general were run (Table 2a); the other configurations are n/a.",
            f"- GA_GDR + SURE {RELEASED}: only g128 general was run (Table 2b); the model is collapsed in BF16 and under "
            "every standard quantizer, so the other configurations were not run.",
            "- Original target rows = the same quantizer applied to the target (reference for recovery).",
            "- GPTQ / AWQ: llm-compressor 0.14.0, W4A16, `lm_head` not quantized; GPTQ symmetric, AWQ asymmetric; "
            "calibration: General text = WikiText-2, BOOKS retain-only = BOOKS retain set.",
            "- Seed 43: GPTQ / AWQ g128 general only (Table 2e).",
            "- " + METRIC_NOTE,
        ],
        heading="##",
    )
    SHOWN["task4_main"] = (*SHOWN["task4_main"][:3], starts)


def task4_split(runs, md, out, seed, splits):
    """Table 4b — dev / held-out halves (extra/splits.json) for the released NPO_KLR + SURE, M1–M4."""
    model = _sure_model("npo_klr_sure", "released", seed)
    rows = []
    for q, calib, gs in [("bf16", None, None)] + T4_CONFIGS:
        d = runs.get(("books", model, "bf16" if q == "bf16" else _t4_tag(q, calib, gs)))
        vals = {}
        for part in ("dev", "heldout"):
            vals[part] = (
                subset_values(d, "books", part, splits) if d is not None else {}
            )
        cells = []
        for m in M14:
            for part in ("dev", "heldout"):
                x = vals[part].get(m)
                cells.append(NA if x is None else x)
        rows.append(
            ["— (BF16)", "—", "—", *cells]
            if q == "bf16"
            else [q.upper(), CALIB_LABEL[calib], str(gs), *cells]
        )
    if all(c == NA for r in rows for c in r[3:]):
        return
    write_table(
        out,
        md,
        "task4_split",
        f"Table 4b — {_method_label('npo_klr_sure', 'released')}: dev vs held-out examples",
        [
            "Quantizer",
            "Calibration",
            "Group size",
            *[
                f"{h} {p}"
                for h in ("M1", "M2", "M3", "M4")
                for p in ("dev", "held-out")
            ],
        ],
        rows,
        notes=[
            "- dev / held-out = the two halves of each evaluation set (extra/splits.json, 50/50, seed 0).",
            "- Use: choose the worst configuration on dev, report it on held-out (no selection on the reported examples).",
            "- Compare each configuration with the BF16 row of the same half; the two halves differ systematically.",
            "- M3 on a half uses the full-set retrained AUC constant.",
        ],
        heading="###",
        collapsed=True,
    )


# ── Manifest — where every result and checkpoint lives ────────────────────────────────────────────────────
import re as _re

_HF_SNAPSHOT = _re.compile(r"models--([^/]+)--([^/]+)/snapshots/([0-9a-f]{40})")


def _account(run_dir):
    """Local download dir 'dl_accN/...' -> 'accN' (no Modal account names in any output)."""
    top = Path(run_dir).parts[0]
    return top.removeprefix("dl_")


def _ckpt_label(path):
    """Checkpoint path from meta -> 'HF org/name@rev' or 'failunl-runs:/<path>' (volume ids dropped)."""
    if not path:
        return NA
    m = _HF_SNAPSHOT.search(path)
    if m:
        return f"HF {m.group(1)}/{m.group(2)}@{m.group(3)}"
    rel = _re.sub(r"^/__modal/volumes/[^/]+/|^/vol/runs/", "", path)
    return f"failunl-runs:/{rel}"


def _transfers(roots):
    """{run name: (from account, to account, hf commit, check)} from push manifests and pull reports next to the
    result dirs (dl_accN/manifest_<run>.json, dl_accN/pull_<run>.json or failunl_pull_report.json).
    """
    pushed, pulled = {}, {}
    for top in sorted({Path(r).parts[0] for r in roots}):
        for f in sorted(Path(top).glob("*.json")):
            try:
                d = json.loads(f.read_text())
            except (ValueError, OSError):
                continue
            if (
                not isinstance(d, dict)
                or d.get("status") != "ok"
                or "hf_commit" not in d
            ):
                continue
            run = d.get("run") or str(d.get("repo_id", "")).rpartition("failunl-")[2]
            if "requested_revision" in d:
                ok = (
                    d.get("verify_final") == "ok"
                    and d["requested_revision"] == d["hf_commit"]
                )
                pulled[run] = (top.removeprefix("dl_"), d["hf_commit"], ok)
            elif "files" in d:
                pushed[run] = (
                    top.removeprefix("dl_"),
                    d["hf_commit"],
                    d.get("remote_check", ""),
                )
    out = {}
    for run, (to, commit, ok) in pulled.items():
        src = pushed.get(run)
        out[run] = (
            src[0] if src else "?",
            to,
            commit,
            (
                "sha256 verified on push and pull"
                if ok and src and src[1] == commit
                else "NOT VERIFIED"
            ),
        )
    return out


def manifest(runs, roots, md, out):
    """One row per evaluated run: result location (local and on the volume), checkpoint, quantization, transfer."""
    md.append("# Manifest\n")
    xfer = _transfers(roots)
    rows = []
    for (corpus, model, tag), d in sorted(runs.items()):
        meta = json.loads((d / "metrics.json").read_text()).get("meta", {})
        q = meta.get("quant_report") or {}
        calib = q.get("calibration")
        calib = (
            calib.get("source") if isinstance(calib, dict) else ("none" if q else "—")
        )
        weights = _ckpt_label(meta.get("model_dir"))
        source = _ckpt_label(q["src"]) if q.get("src") else weights
        run_name = Path(q.get("src") or meta.get("model_dir") or "").name
        if run_name.startswith("checkpoint-"):
            run_name = Path(meta.get("model_dir", "")).parent.name
        t = xfer.get(run_name)
        acct = _account(d)
        if t and acct == t[1]:
            transfer = f"copied {t[0]} → {t[1]} via private HF repo failunl-{run_name}@{t[2][:12]} ({t[3]})"
        elif t and acct == t[0]:
            transfer = f"trained here; copied to {t[1]} (HF commit {t[2][:12]})"
        else:
            transfer = "—"
        quant = (
            f"{q.get('method', '?').upper()} g{q.get('group_size')} "
            f"{'sym' if q.get('symmetric') else 'asym'}, calib {calib}"
            if q
            else (
                "bitsandbytes FP4 (load-time)" if meta.get("quant") == "bnb4" else "—"
            )
        )
        rows.append(
            [
                corpus,
                model,
                tag,
                acct,
                str(d),
                f"failunl-runs:/results/{'/'.join(d.parts[-3:])}",
                weights,
                source,
                quant,
                (q.get("end") or "—"),
                transfer,
            ]
        )
    write_table(
        out,
        md,
        "manifest",
        "Manifest — result and checkpoint location of every evaluated run",
        [
            "Corpus",
            "Model",
            "Tag",
            "Account",
            "Local result dir",
            "Result on volume",
            "Evaluated weights",
            "Source checkpoint",
            "Quantization",
            "Quantized at",
            "Checkpoint transfer",
        ],
        rows,
        notes=[
            "- Account = local download label (acc1–acc4); each account has its own `failunl-runs` volume.",
            "- Evaluated weights = `meta.model_dir` (for INT4: the dequantized `dq` copy); source checkpoint = the BF16 "
            "weights it was made from (`quant_report.src`). HF …@rev = pinned snapshot revision.",
            "- bnb4 quantizes at load time from the BF16 checkpoint (no stored quantized copy).",
            "- Checkpoint transfer: the BF16 checkpoint was trained on one account and copied to another through a "
            "private HF repo, pinned to one commit; sha256 of every file checked after push and after pull.",
            "- Checkpoints and quantized models are not distributed (13.5 GB each); the volume paths record where "
            "they were stored.",
        ],
        heading="###",
        collapsed=True,
    )


def diag_fp32(runs, md, out):
    """KnowMem-forget under the FP32 diagnostic loader (eval_run --diag_fp32_knowmem, a DEVIATION) vs BF16."""
    rows = []
    for (c, m, t), d in sorted(runs.items()):
        if not t.startswith("fp32diag"):
            continue
        fp32 = metric_values(d)["M2"]
        b = _get(runs, c, m, "bf16")
        bf16 = b["M2"] if b else None
        paper = None
        if m == "target":
            paper = (PAPER_T1_NEWS if c == "news" else PAPER_T3_BOOKS)[
                ("target", "bf16")
            ][1]
        elif m == "retrain":
            paper = PAPER_T1_RETRAIN[c][1]
        rows.append(
            [
                c,
                m,
                t,
                bf16,
                fp32,
                _delta(fp32, bf16),
                paper if paper is not None else "—",
                _delta(fp32, paper) if paper is not None else "—",
            ]
        )
    if not rows:
        return
    write_table(
        out,
        md,
        "diag_fp32_knowmem",
        "Table 2i — Diagnostic: target M2 with an FP32 load (deviation)",
        [
            "Corpus",
            "Model",
            "Tag",
            "M2 BF16 (authors' loader)",
            "M2 FP32 (diagnostic)",
            "FP32 − BF16",
            "paper M2 (MUSE-copied)",
            "FP32 − paper",
        ],
        rows,
        heading="###",
        collapsed=True,
        notes=[
            "- Not a reproduction result: the authors' loader uses BF16 (utils.py:110-115).",
            "- Evaluation precision does not explain the M2 gap: 46–47 in both precisions.",
            "- The authors' own shipped `output.csv` has 46.9; the reported 59.4 appears to be copied from MUSE.",
        ],
    )


def _cell(x):
    return fmt(x) if not isinstance(x, str) else x


# ── terminal printing (--show) ────────────────────────────────────────────────────────────────────────────
SHOW_KEYS = [
    "task1_taskmd",
    "task1_compare",
    "task1_quant",
    "task1_epochs",
    "t23_main",
    "t23_released",
    "t2_compare_fixed",
    "t2_compare_released",
    "t23_seeds",
    "t23_recovery",
    "t23_utility",
    "t2_quant_path",
    "t2_truthful_mc1",
    "diag_fp32_knowmem",
    "task4_main",
    "task4_split",
    "manifest",
]
SAVED = "tables.json"  # every table as built, read back by --show (no rebuild)


def save_tables(out):
    data = {
        "tables": {k: list(v) for k, v in SHOWN.items()},
        "blocks": {k: list(v) for k, v in BLOCK_TABLES.items()},
    }
    (out / SAVED).write_text(json.dumps(data, ensure_ascii=False))


def load_tables(src):
    f = Path(src) / SAVED
    if not f.exists():
        raise SystemExit(
            f"{f} not found: build the tables first (run without --show, with --out {src})."
        )
    data = json.loads(f.read_text())
    tup = lambda c: (
        tuple(c) if isinstance(c, list) else c
    )  # noqa: E731  (JSON turns cell tuples into lists)
    blocks = {
        k: (
            t,
            h,
            [
                (n, [(src_, [tup(c) for c in cells]) for src_, cells in lines])
                for n, lines in b
            ],
            notes,
        )
        for k, (t, h, b, notes) in data["blocks"].items()
    }
    return data["tables"], blocks


def _is_num(s):
    try:
        float(str(s).rstrip("†!").split("±")[0].split(" ")[0])
        return True
    except ValueError:
        return False


def render_ascii(title, header, rows, starts=None):
    """Fixed-width block: '=== TABLE title ===', header, separator lines between groups (method blocks)."""
    cells = [[_cell(c) for c in r] for r in rows]
    w = [max([_dw(h)] + [_dw(r[i]) for r in cells]) for i, h in enumerate(header)]
    numeric = [
        any(_is_num(r[i]) for r in cells) for i in range(len(header))
    ]  # numbers right-aligned, text left
    line = lambda r: " | ".join(
        _rjust(c, w[i]) if numeric[i] else _ljust(c, w[i])  # noqa: E731
        for i, c in enumerate(r)
    )
    sep = "-+-".join("-" * x for x in w)
    out = [
        f"=== TABLE {_wide_gap(title)} ===",
        line(header),
        sep.replace("-+-", "=+=").replace("-", "="),
    ]
    if starts is None:  # group consecutive rows by their first column
        starts = [i for i, r in enumerate(cells) if i == 0 or r[0] != cells[i - 1][0]]
    for i, r in enumerate(cells):
        if i in starts and i:
            out.append(sep)
        out.append(line(r))
    return "\n".join(out)


def show_tables(keys, src, color=False):
    """Print saved tables from <src>/tables.json; a key that is not there is an error."""
    tables, blocks = load_tables(src)
    missing = [k for k in keys if k not in tables and k not in blocks]
    if missing:
        raise SystemExit(
            f"not in {Path(src) / SAVED}: {', '.join(missing)} "
            "(no data for it when the file was built, or the file is older than this script: rebuild)."
        )
    for k in keys:
        if k in blocks:
            print(render_blocks(*blocks[k], color=color) + "\n")
        else:
            print(render_ascii(*tables[k]) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        nargs="+",
        default=["modal_results/results"],
        help="one or more result dirs (e.g. one per Modal account); runs are merged; identical duplicates are skipped, differing ones are an error",
    )
    ap.add_argument("--out", default="tables")
    ap.add_argument(
        "--seed", type=int, default=42, help="training seed of the main runs"
    )
    ap.add_argument(
        "--show",
        nargs="+",
        choices=SHOW_KEYS,
        metavar="TABLE",
        help="print these tables from <--from>/tables.json (no rebuild); see --list-tables",
    )
    ap.add_argument(
        "--from",
        dest="src",
        default="tables",
        help="--show: directory with a previously built tables.json (default: tables)",
    )
    ap.add_argument(
        "--list-tables",
        action="store_true",
        help="print the table keys for --show and exit",
    )
    ap.add_argument(
        "--no-color",
        action="store_true",
        help="--show: no ANSI bold (default: bold only on a terminal)",
    )
    a = ap.parse_args(argv)
    if a.list_tables:
        print("\n".join(SHOW_KEYS))
        return
    if a.show:
        import os
        import sys

        color = not a.no_color and sys.stdout.isatty() and "NO_COLOR" not in os.environ
        show_tables(a.show, a.src, color)
        return
    roots, out = [Path(r) for r in a.results], Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_all(roots)
    splits = json.loads((HERE / "extra" / "splits.json").read_text())
    md = [
        "# Results tables (generated by compare_results.py — measured values only; 'paper' columns are "
        "published values)\n",
        f"Runs found: {len(runs)} under {', '.join(map(str, roots))}\n",
    ]
    limited = sorted(
        "/".join(k)
        for k, d in runs.items()
        if json.loads((d / "metrics.json").read_text())["meta"].get("limit")
    )
    if limited:
        md.append(
            "**WARNING — runs evaluated on a subset (`--limit`, tests only); their values are NOT comparable: "
            + ", ".join(limited)
            + "**\n"
        )
    task1_taskmd(runs, md, out, a.seed)
    task1_compare(runs, md, out, a.seed)
    task1_quant(runs, md, out, a.seed)
    task1_epochs(runs, md, out, a.seed)
    t23_main(runs, md, out, a.seed)
    t23_released(runs, md, out, a.seed)
    t2_compare_fixed(runs, md, out, a.seed)
    t2_compare_released(runs, md, out, a.seed)
    t23_seeds(runs, md, out, a.seed)
    t23_recovery(runs, md, out, a.seed)
    t23_utility(runs, md, out, a.seed)
    t2_quant_path(runs, md, out, a.seed)
    diag_fp32(runs, md, out)
    t2_truthful_mc1(runs, md, out, a.seed)
    task4_main(runs, md, out, a.seed)
    task4_split(runs, md, out, a.seed, splits)
    manifest(runs, roots, md, out)
    (out / "tables.md").write_text("\n".join(md))
    save_tables(out)
    print(f"wrote {out/'tables.md'} and {out/SAVED} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
