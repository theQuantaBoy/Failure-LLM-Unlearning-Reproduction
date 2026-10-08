# Note: this script reads per-step training logs that are not shipped in this repository (they stay on the Modal Volume).
"""
extra/plot_trajectory.py — M1–M4 of one training run at its evaluated epochs, against the per-step loss components
the run recorded (W6, <ckpt dir>/loss_components.jsonl). Measured values only; the paper's final-epoch values are
drawn as hollow markers and labelled.

    modal volume get failunl-runs ckpt/news/news_ga_gdr_s42/loss_components.jsonl modal_results/train/news_ga_gdr_s42/
    .venvs/paper/bin/python -m extra.plot_trajectory --corpus news --model ga_gdr_s42 \
        --loss modal_results/train/news_ga_gdr_s42/loss_components.jsonl --out figures/news_ga_gdr_trajectory.png

Epoch 0 = the target model (results/<corpus>/target/bf16); epoch k = results/<corpus>/<model>/bf16_ep<k>; the last
epoch = results/<corpus>/<model>/bf16. Prints the per-epoch loss summary and the metric table it plotted.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from compare_results import PAPER_T1_NEWS, load_all, metric_values  # noqa: E402
from extra.common import PRESETS  # noqa: E402

# reference palette (dataviz skill references/palette.md), categorical slots 1-6 in fixed order, light surface;
# validated with scripts/validate_palette.js (all checks pass; slots 3-5 < 3:1 contrast -> direct labels + table)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_FORGET, C_RETAIN, C_M1, C_M2, C_M4, C_M3 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"


def rolling_median(xs, w):
    h = w // 2
    return [statistics.median(xs[max(0, i - h): i + h + 1]) for i in range(len(xs))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="news")
    ap.add_argument("--model", default="ga_gdr_s42", help="results/<corpus>/<model>")
    ap.add_argument("--loss", required=True, help="loss_components.jsonl of that training run")
    ap.add_argument("--results", nargs="+", default=["modal_results/results"])
    ap.add_argument("--out", default="figures/trajectory.png")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method = a.model.rpartition("_s")[0]
    n_ep = PRESETS[f"{a.corpus}_{method}"]["epochs"]
    rec = [json.loads(line) for line in open(a.loss)]
    spe = len(rec) / n_ep
    x = [r["step"] / spe for r in rec]
    ce_f, ce_r = [r["ce_forget"] for r in rec], [r["ce_retain"] for r in rec]

    print("epoch  CE_forget mean/median  CE_retain mean/median/max")
    for e in range(n_ep):
        f = ce_f[round(e * spe): round((e + 1) * spe)]
        g = ce_r[round(e * spe): round((e + 1) * spe)]
        print(f"{e + 1:5d}  {statistics.fmean(f):8.2f} {statistics.median(f):8.2f}   "
              f"{statistics.fmean(g):7.2f} {statistics.median(g):7.2f} {max(g):7.2f}")

    runs = load_all([Path(r) for r in a.results])
    points = {0: ("target", "bf16")}
    for (c, m, t) in runs:
        if c == a.corpus and m == a.model and t.startswith("bf16_ep") and t[7:].isdigit():
            points[int(t[7:])] = (m, t)
    points[n_ep] = (a.model, "bf16")
    traj = {}
    for ep, (m, t) in sorted(points.items()):
        d = runs.get((a.corpus, m, t))
        if d is not None:
            traj[ep] = metric_values(d)
    print("\nepoch  M1    M2    M3     M4")
    for ep, v in traj.items():
        print(f"{ep:5d}  {v['M1']:5.1f} {v['M2']:5.1f} {v['M3']:6.1f} {v['M4']:5.1f}")
    paper = PAPER_T1_NEWS.get((method, "bf16")) if a.corpus == "news" else None

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.titlelocation": "left"})
    fig, axs = plt.subplots(3, 1, figsize=(8, 8.4), sharex=True, gridspec_kw={"height_ratios": [1.25, 1.25, 0.8]},
                            facecolor=SURFACE)
    for ax in axs:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for ep in traj:
            ax.axvline(ep, color=GRID, linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)

    # A: loss components (log scale; raw steps faint, 25-step rolling median on top)
    ax = axs[0]
    for ys, col, name in ((ce_f, C_FORGET, "forget CE (GA term)"), (ce_r, C_RETAIN, "retain CE (GDR term)")):
        ax.plot(x, ys, color=col, linewidth=0.5, alpha=0.25)
        sm = rolling_median(ys, 25)
        ax.plot(x, sm, color=col, linewidth=2, label=name)
        ax.annotate(name.split(" (")[0], (x[-1], sm[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK2,
                    fontsize=8, annotation_clip=False)
    ax.set_yscale("log")
    ax.set_ylabel("cross-entropy (nats/token)")
    ax.set_title(f"A  Training loss components, {a.corpus.upper()} {method.upper()} (per step; line = 25-step median)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_ylim(0.2, 600)

    # B: M1, M2, M4 (same 0-100 ROUGE scale)
    ax = axs[1]
    eps = list(traj)
    px = n_ep + 0.35  # paper markers sit just right of the last epoch so they do not cover the measured labels
    # value-label offsets per series (points): M2 above, M4 below, M1 to the left -> no collisions
    offs = {"M1": ((-9, 0), "right", "center"), "M2": ((0, 7), "center", "bottom"), "M4": ((0, -8), "center", "top")}
    for key, col, mk, name, pidx in (("M1", C_M1, "o", "M1 VerbMem", 0), ("M2", C_M2, "s", "M2 KnowMem forget", 1),
                                     ("M4", C_M4, "D", "M4 KnowMem retain", 3)):
        ys = [traj[e][key] for e in eps]
        ax.plot(eps, ys, color=col, linewidth=2, marker=mk, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=2, label=name, zorder=3)
        (dx, dy), ha, va = offs[key]
        for e, y in zip(eps, ys):
            ax.annotate(f"{y:.1f}", (e, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=7.5,
                        color=INK2)
        ax.annotate(name, (px, ys[-1]), xytext=(12, {"M2": 6, "M4": -6}.get(key, 0)), textcoords="offset points", va="center", color=INK2,
                    fontsize=8, annotation_clip=False)
        if paper:
            ax.plot([px], [paper[pidx]], marker=mk, markersize=8, markerfacecolor="none", markeredgecolor=col,
                    markeredgewidth=1.5, linestyle="none", zorder=4)
    if paper:
        ax.plot([], [], marker="o", markerfacecolor="none", markeredgecolor=INK2, linestyle="none",
                label=f"paper, epoch {n_ep} (hollow, right of the last epoch)")
    ax.set_ylim(-12, 120)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("ROUGE-L × 100")
    ax.set_title("B  M1, M2, M4 at evaluated epochs (epoch 0 = target)")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)

    # C: M3 on its own axis (different scale)
    ax = axs[2]
    ys = [traj[e]["M3"] for e in eps]
    ax.plot(eps, ys, color=C_M3, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
            zorder=3)
    for e, y in zip(eps, ys):
        ax.annotate(f"{y:.1f}", (e, y), xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.5, color=INK2)
    if paper:
        ax.annotate(f"paper {paper[2]:.1f}", (n_ep + 0.35, paper[2]), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK2, annotation_clip=False)
        ax.plot([n_ep + 0.35], [paper[2]], marker="o", markersize=8, markerfacecolor="none", markeredgecolor=C_M3,
                markeredgewidth=1.5, linestyle="none")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_ylim(-125, 150)
    ax.set_ylabel("PrivLeak (→ 0)")
    ax.set_title("C  M3 PrivLeak")
    ax.set_xlabel("epoch")
    ax.set_xticks(range(0, n_ep + 1))
    ax.set_xlim(-0.5, n_ep + 0.6)

    fig.tight_layout(rect=(0, 0, 0.84, 1))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
