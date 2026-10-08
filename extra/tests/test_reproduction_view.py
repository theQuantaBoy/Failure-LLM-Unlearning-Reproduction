"""
Offline test of compare_results.py's "Reproduction view" (Paper / Ours / Δ blocks) and of --show / --list-tables.

A fake results dir holds a few runs whose measured values are sentinels that differ from every published value
(paper + 0.37 / + 5.37 / ...). Checks: block structure (Paper, Ours, Δ lines; 'Ours (new)' only for rows without a
published value), '!' exactly where |Δ| > threshold (and nowhere else), the † MUSE rows (Δ M1 = greedy − paper, †
only on M1–M4), MISSING / 'not run' / nan kept explicit, no paper value in any Ours cell, the threshold flag, and the
fixed-width terminal output (every line of a block has the same column boundaries).

    .venvs/paper/bin/python -m extra.tests.test_reproduction_view
"""

import contextlib
import io
import json
import math
import re
import tempfile
from pathlib import Path

import compare_results as cr

# measured = paper + offset: |offset| <= 2 -> no '!', > 2 -> '!'
MEAS = {
    # NEWS target BF16 (MUSE row): sampled M1 far from paper, greedy within 0.37 -> Δ M1 uses greedy, no '!'
    ("news", "target", "bf16"): {"verbmem_f": 58.4 - 15.0, "verbmem_f_greedy": 58.4 + 0.37, "knowmem_f": 63.9 + 2.5,
                                 "privleak": -99.8 + 0.1, "knowmem_r": 55.2 - 0.37},
    # NEWS NPO_KLR BF16: M2 off by 5.37 -> '!' on paper M2 only
    ("news", "npo_klr_s42", "bf16"): {"verbmem_f": 16.6 + 1.9, "verbmem_f_greedy": 1.11, "knowmem_f": 36.6 + 5.37,
                                      "privleak": -94.0 - 0.37, "knowmem_r": 33.3 + 0.37},
    # NEWS GA_GDR epoch 2: no published value -> 'Ours (new)' only
    ("news", "ga_gdr_s42", "bf16_ep2"): {"verbmem_f": 2.22, "verbmem_f_greedy": 3.33, "knowmem_f": 4.44,
                                         "privleak": 5.55, "knowmem_r": 6.66},
    # NEWS retrain: only M3 measured
    ("news", "retrain", "bf16"): {"privleak": 0.37},
    # BOOKS NPO_KLR + SURE BF16: Tru nan, Flu 3 points off
    ("books", "npo_klr_sure_s42", "bf16"): {"verbmem_f": 17.6 + 0.37, "verbmem_f_greedy": 7.77, "knowmem_f": 37.8 - 2.37,
                                            "privleak": -58.0 + 0.37, "knowmem_r": 49.4 + 0.37, "gen": 23.4 + 0.37,
                                            "tru": float("nan"), "fac": 7.4 + 0.37, "flu": 588.8 + 3.0},
    # BOOKS target BF16: utility paper values are the paper's own (no †)
    ("books", "target", "bf16"): {"verbmem_f": 50.0, "verbmem_f_greedy": 99.8 + 0.37, "knowmem_f": 59.4 - 12.37,
                                  "privleak": -57.5 + 0.37, "knowmem_r": 66.9 + 0.37, "gen": 28.7 + 0.37,
                                  "tru": 33.6 + 0.37, "fac": 9.1 + 0.37, "flu": 573.3 + 0.37},
    ("books", "target", "fp32diag_knowmem"): {"knowmem_f": 59.4 - 13.37},
}


def write_fake(root: Path):
    for (corpus, model, tag), m in MEAS.items():
        d = root / corpus / model / tag
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"meta": {"model_dir": str(d), "limit": None}, "metrics": m}))


def blocks(md_text: str, title_prefix: str):
    """Rows (lists of cells) of one '### <title>' table of the reproduction view."""
    sec = md_text.split(f"### {title_prefix}", 1)[1].split("\n### ", 1)[0].split("\n†", 1)[0]
    lines = [ln for ln in sec.splitlines() if ln.startswith("| ")]
    return [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[1:]]  # skip the header row


def main():
    paper_strings = set()
    for tab in (cr.PAPER_T1_NEWS, cr.PAPER_T3_BOOKS):
        for vals in tab.values():
            paper_strings.update(f"{v:.1f}" for v in vals)
    for vals in list(cr.PAPER_T1_RETRAIN.values()) + [cr.PAPER_T1_BOOKS_TARGET_4BIT]:
        paper_strings.update(f"{v:.1f}" for v in vals)
    measured_strings = {f"{v:.1f}" for m in MEAS.values() for v in m.values() if not math.isnan(v)}
    results = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_fake(tmp / "res")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cr.SHOWN.clear()
            cr.main(["--results", str(tmp / "res"), "--out", str(tmp / "out"), "--show", "task1", "books", "fp32diag"])
        md = (tmp / "out" / "tables.md").read_text()
        term = buf.getvalue()

        # 1 structure + marks, NEWS
        news = blocks(md, "NEWS")
        cols = ["M1", "M1 greedy", "M2", "M3", "M4"]
        by = {}
        i = 0
        while i < len(news):
            r = news[i]
            if r[2] == "Ours (new)":
                by[(r[0], r[1])] = {"new": r}
                i += 1
                continue
            assert r[2] == "Paper" and news[i + 1][2] == "Ours" and news[i + 2][2] == "Δ", news[i:i + 3]
            assert news[i + 1][:2] == ["", ""] and news[i + 2][:2] == ["", ""]
            by[(r[0], r[1])] = {"paper": dict(zip(cols, r[3:])), "ours": dict(zip(cols, news[i + 1][3:])),
                                "delta": dict(zip(cols, news[i + 2][3:]))}
            i += 3
        t = by[("Original target", "BF16")]
        assert t["paper"]["M1"] == "58.4†", t  # Δ M1 = greedy − paper = 0.37 -> no '!'
        assert t["delta"]["M1"] == "0.4†", t
        assert t["ours"]["M1"] == "43.4" and t["ours"]["M1 greedy"] == "58.8", t
        assert t["paper"]["M2"] == "63.9†!" and t["delta"]["M2"] == "2.5", t  # 2.5 > 2.0 -> '!'
        assert t["paper"]["M3"] == "-99.8†" and t["paper"]["M4"] == "55.2†", t
        n = by[("NPO_KLR", "BF16, epoch 10 (final)")]
        assert n["paper"] == {"M1": "16.6", "M1 greedy": "—", "M2": "36.6!", "M3": "-94.0", "M4": "33.3"}, n
        assert n["delta"]["M1"] == "1.9" and n["delta"]["M2"] == "5.4", n
        assert by[("GA_GDR", "BF16, epoch 2 of 10 (intermediate checkpoint)")]["new"][3:] == \
            ["2.2", "3.3", "4.4", "5.5", "6.7"]
        rt = by[("Retrained (reference)", "BF16")]
        assert rt["ours"] == {"M1": "not run", "M1 greedy": "not run", "M2": "not run", "M3": "0.4", "M4": "not run"}
        assert rt["delta"]["M3"] == "0.4" and rt["delta"]["M1"] == "not run", rt
        missing = by[("NPO_KLR", "4-bit / authors' bnb-FP4 (reproduction)")]
        assert set(missing["ours"].values()) == {"MISSING"} and missing["paper"]["M1"] == "34.1", missing
        assert missing["delta"]["M2"] == "MISSING", missing
        results["news_blocks"] = f"{len(by)} blocks ok"

        # 2 BOOKS: nan kept, † only on M1–M4, '!' only where |Δ| > 2
        books = blocks(md, "BOOKS")
        bcols = ["M1", "M1 greedy", "M2", "M3", "M4", "Gen", "Tru", "Fac", "Flu"]
        idx = {(r[0], r[1]): k for k, r in enumerate(books) if r[2] == "Paper"}
        k = idx[("NPO_KLR + SURE", "BF16")]
        p, o, d = (dict(zip(bcols, books[k + j][3:])) for j in range(3))
        assert o["Tru"] == "nan" and d["Tru"] == "nan" and p["Tru"] == "30.2", (p, o, d)
        assert p["M2"] == "37.8!" and p["Flu"] == "588.8!" and p["M1"] == "17.6", p
        k = idx[("Original target", "BF16")]
        p, o, d = (dict(zip(bcols, books[k + j][3:])) for j in range(3))
        assert p["M1"] == "99.8†" and d["M1"] == "0.4†" and p["M2"] == "59.4†!", (p, d)
        assert p["Gen"] == "28.7" and p["Flu"] == "573.3", p  # paper's own utility values: no †
        k = idx[("Original target", "4-bit / authors' bnb-FP4 (reproduction)")]
        p = dict(zip(bcols, books[k][3:]))
        assert p["M1"] == "85.3" and p["Gen"] == "—" and p["Flu"] == "—", p  # Table 1 row: M1–M4 only
        assert any(r[2] == "Ours (new)" and r[1] == "INT4 / GPTQ (g128, general)" for r in books)
        results["books_blocks"] = "nan, †, '!', '—' ok"

        # 3 no paper value in an Ours cell: every numeric Ours cell is one of the measured sentinels
        for table in (news, books, blocks(md, "FP32")):
            for r in table:
                if r[2].startswith("Ours"):
                    for c in r[3:]:
                        if c in ("MISSING", "not run", "nan"):
                            continue
                        assert c in measured_strings, f"Ours cell {c!r} is not a measured value: {r}"
                        assert c not in paper_strings - measured_strings, r
        fp = blocks(md, "FP32")
        assert [r[3] for r in fp] == ["59.4†!", "46.0", "-13.4"], fp
        results["no_paper_in_ours"] = "ok"

        # 4 terminal output: header line, equal widths, same column boundaries in every line of a table
        tables = re.split(r"\n(?==== TABLE )", term.split("\n", 2)[2].strip())
        assert len(tables) == 3 and all(tb.startswith("=== TABLE Reproduction view") for tb in tables), term[:300]
        for tb in tables:
            lines = tb.splitlines()[1:]
            bars = {tuple(m.start() for m in re.finditer(r"(?<= )\|(?= |$)|(?<=[-=])\+(?=[-=])", ln)) for ln in lines}
            assert len(bars) == 1, f"misaligned columns in:\n{tb}"
            assert len({len(ln) for ln in lines}) == 1, f"ragged lines in:\n{tb}"
        assert "| Paper " in term and "| Ours " in term and "| Δ " in term
        results["terminal"] = f"{len(tables)} tables aligned"

        # 5 threshold flag: with 10 points only |Δ| > 10 is marked
        cr.SHOWN.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            cr.main(["--results", str(tmp / "res"), "--out", str(tmp / "out10"), "--diff-threshold", "10"])
        md10 = (tmp / "out10" / "tables.md").read_text()
        marks = re.findall(r"\| ([-0-9.]+†?!) ", md10.split("## Reproduction view", 1)[1])
        assert sorted(marks) == ["59.4†!", "59.4†!"], marks  # BOOKS target M2 (-12.4) and FP32 diag (-13.4)
        results["threshold"] = "ok"

        # 6 --list-tables
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cr.main(["--list-tables"])
        assert set(ln.split()[0] for ln in buf.getvalue().splitlines()) == set(cr.SHOW_KEYS)
        results["list_tables"] = "ok"

    print(json.dumps(results, indent=2))
    print("test_reproduction_view: ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
