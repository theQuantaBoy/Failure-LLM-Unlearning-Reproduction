"""
extra/bootstrap_ci.py — paired bootstrap CIs for the change of M1, M2, |M3| and M4 between two evaluations of the same
examples (a quantized model vs its own BF16 checkpoint), from the saved per-example outputs. No GPU, no model.

    .venvs/paper/bin/python -m extra.bootstrap_ci --a dl_acc2/results/books/npo_klr_s42/bnb4 \
        --b dl_acc2/results/books/npo_klr_s42/bf16 --corpus books

compare_results.py calls paired_deltas() for every quantized BOOKS row (section "Paired bootstrap").

Per metric, resample example indices with replacement (B = 10,000, numpy default_rng(seed)), the SAME indices for both
runs (paired), and take the 2.5 / 97.5 percentiles of Δ = metric(a) − metric(b).
- M1 = VerbMem (sampled, the primary column), M2 = KnowMem-forget, M4 = KnowMem-retain: mean per-example ROUGE-L x100
  (what eval_run writes as verbmem_f / knowmem_f / knowmem_r).
- M3 = PrivLeak = (AUC − AUC_retrain) / AUC_retrain x100 with the full-set retrained constant (constants.py:64,165);
  AUC over forget (label 0) vs holdout (label 1) with score −Min-40% (metrics/privleak.py:58-61). Forget and holdout
  are resampled independently (paired between a and b). Δ|M3| = |M3(a)| − |M3(b)|.
The AUC is computed as the Mann-Whitney statistic (ties count 1/2), which equals sklearn's trapezoidal ROC AUC; the
point estimates are checked against metrics.json (|diff| < 1e-9) before any resampling.
"""

import argparse
import json
from pathlib import Path

import numpy as np

AUC_RETRAIN = {"news": 0.47719999999999996, "books": 0.5392999999999999}  # = compare_results.AUC_RETRAIN
B_DEFAULT = 10_000
FILES = {"M1": ("verbmem_sample.json", "verbmem_f"), "M2": ("knowmem_f.json", "knowmem_f"),
         "M4": ("knowmem_r.json", "knowmem_r")}


def _paired(a_recs, b_recs, field, where):
    """Per-example arrays of a and b aligned on idx; the example text hashes must agree."""
    bm = {r["idx"]: r for r in b_recs}
    if set(bm) != {r["idx"] for r in a_recs}:
        raise ValueError(f"{where}: the two runs cover different examples")
    xa, xb = [], []
    for r in sorted(a_recs, key=lambda r: r["idx"]):
        s = bm[r["idx"]]
        if r["sha1"] != s["sha1"]:
            raise ValueError(f"{where}: example {r['idx']} differs between the runs (sha1)")
        xa.append(float(r[field]))
        xb.append(float(s[field]))
    return np.array(xa), np.array(xb)


def auc_rows(forget, holdout):
    """Row-wise AUC (Mann-Whitney) of score = −ppl: P(holdout scores higher than forget), ties 1/2.
    forget: [R, nf], holdout: [R, nh] Min-40% values (lower = more 'member-like')."""
    f = forget[:, :, None]
    h = holdout[:, None, :]
    return ((h < f).sum(axis=(1, 2)) + 0.5 * (h == f).sum(axis=(1, 2))) / (forget.shape[1] * holdout.shape[1])


def _ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def paired_deltas(run_a: Path, run_b: Path, corpus: str, n_boot: int = B_DEFAULT, seed: int = 0) -> dict:
    """{metric: (Δ point, CI low, CI high)} for M1, M2, absM3, M4; a metric whose files are missing is skipped."""
    run_a, run_b = Path(run_a), Path(run_b)
    ma = json.loads((run_a / "metrics.json").read_text())["metrics"]
    mb = json.loads((run_b / "metrics.json").read_text())["metrics"]
    rng = np.random.default_rng(seed)
    out = {}
    for key, (fname, mkey) in FILES.items():
        if not ((run_a / fname).exists() and (run_b / fname).exists()):
            continue
        xa, xb = _paired(json.loads((run_a / fname).read_text())["per_example"],
                         json.loads((run_b / fname).read_text())["per_example"], "rougeL", f"{run_a} {fname}")
        for x, m in ((xa, ma), (xb, mb)):
            if abs(100 * x.mean() - m[mkey]) > 1e-9:
                raise ValueError(f"{fname}: per-example mean {100 * x.mean()} != metrics.json {mkey} {m[mkey]}")
        idx = rng.integers(0, len(xa), size=(n_boot, len(xa)))
        d = 100 * (xa[idx].mean(axis=1) - xb[idx].mean(axis=1))
        out[key] = (100 * (xa.mean() - xb.mean()), *_ci(d))
    pa, pb = run_a / "privleak.json", run_b / "privleak.json"
    if pa.exists() and pb.exists():
        A, Bj = json.loads(pa.read_text())["per_example"], json.loads(pb.read_text())["per_example"]
        fa, fb = _paired(A["forget"], Bj["forget"], "Min-40%", f"{run_a} privleak forget")
        ha, hb = _paired(A["holdout"], Bj["holdout"], "Min-40%", f"{run_a} privleak holdout")
        c = AUC_RETRAIN[corpus]
        m3 = lambda auc: (auc - c) / c * 100  # noqa: E731
        for f, h, m in ((fa, ha, ma), (fb, hb, mb)):
            auc = float(auc_rows(f[None], h[None])[0])
            if abs(auc - m["privleak_auc"]) > 1e-9:
                raise ValueError(f"privleak: recomputed AUC {auc} != metrics.json {m['privleak_auc']}")
        fi = rng.integers(0, len(fa), size=(n_boot, len(fa)))
        hi = rng.integers(0, len(ha), size=(n_boot, len(ha)))
        d = np.empty(n_boot)
        for s in range(0, n_boot, 1000):  # chunks keep the [R, nf, nh] comparison small
            sl = slice(s, s + 1000)
            d[sl] = np.abs(m3(auc_rows(fa[fi[sl]], ha[hi[sl]]))) - np.abs(m3(auc_rows(fb[fi[sl]], hb[hi[sl]])))
        point = abs(m3(auc_rows(fa[None], ha[None])[0])) - abs(m3(auc_rows(fb[None], hb[None])[0]))
        out["absM3"] = (float(point), *_ci(d))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run dir of the changed model (e.g. quantized)")
    ap.add_argument("--b", required=True, help="run dir of the reference (e.g. its own BF16)")
    ap.add_argument("--corpus", default="books")
    ap.add_argument("--n-boot", type=int, default=B_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    for k, (p, lo, hi) in paired_deltas(Path(a.a), Path(a.b), a.corpus, a.n_boot, a.seed).items():
        print(f"Δ{k:6s} {p:8.2f}  95% CI [{lo:8.2f}, {hi:8.2f}]")


if __name__ == "__main__":
    main()
