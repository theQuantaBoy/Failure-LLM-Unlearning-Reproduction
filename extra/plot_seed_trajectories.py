# Note: this script reads per-step training logs that are not shipped in this repository (they stay on the Modal Volume).
"""
extra/plot_seed_trajectories.py — two training runs of the same preset (different seeds) side by side, from their
per-step loss components (W6, loss_components.jsonl) and the Trainer's grad_norm log lines (every 500 steps, before
clipping). Prints a per-epoch table (medians) and the first step from which the 50-step rolling medians of a component
differ by more than --rel (relative) for 50 consecutive steps; writes one figure with one panel per component.

    .venvs/paper/bin/python -m extra.plot_seed_trajectories \
        --a dl_acc2/train_logs/books_npo_klr_sure_loss_components.jsonl --a-name s42 \
        --a-out modal_results/logs/20261005-075330_unlearn_books_npo_klr_sure_s42.out \
        --b dl_acc3/loss_s43.jsonl --b-name s43 --b-out dl_acc3/logs/20261005-180930_unlearn_books_npo_klr_sure_s43.out \
        --epochs 5 --title "BOOKS NPO_KLR + SURE" --out figures/books_npo_klr_sure_s42_vs_s43.png
"""

import argparse
import ast
import json
import re
import statistics
from pathlib import Path

from extra.plot_trajectory import GRID, INK, INK2, SURFACE, rolling_median

C_A, C_B = "#2a78d6", "#eb6834"  # categorical slots 1, 2 of the validated palette used in plot_trajectory.py
COMPONENTS = [("kl_raw", "retain KL (raw, before ×α)", True), ("ce_forget", "forget CE (nats/token)", True),
              ("ce_retain", "retain CE (nats/token)", True), ("npo_logsigmoid_mean", "NPO mean log σ(·)", False)]


def grad_norms(out_log):
    """[(epoch, grad_norm)] from the Trainer's "{'loss': ..., 'grad_norm': ...}" lines."""
    if not out_log or not Path(out_log).exists():
        return []
    return [(d["epoch"], d["grad_norm"]) for d in
            (ast.literal_eval(m) for m in re.findall(r"\{'loss'[^}]*\}", Path(out_log).read_text()))]


def first_divergence(a, b, rel, w=50):
    ra, rb = rolling_median(a, w), rolling_median(b, w)
    run = 0
    for i, (x, y) in enumerate(zip(ra, rb)):
        run = run + 1 if abs(x - y) > rel * max(abs(x), abs(y), 1e-12) else 0
        if run == w:
            return i - w + 2  # 1-based step where the run started
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    for s in ("a", "b"):
        ap.add_argument(f"--{s}", required=True, help="loss_components.jsonl")
        ap.add_argument(f"--{s}-name", required=True)
        ap.add_argument(f"--{s}-out", default=None, help="the run's .out log (for grad_norm)")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--rel", type=float, default=0.25)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    A = [json.loads(x) for x in open(a.a)]
    B = [json.loads(x) for x in open(a.b)]
    assert len(A) == len(B), (len(A), len(B))
    spe = len(A) / a.epochs

    print(f"per-epoch medians ({a.a_name} / {a.b_name})")
    print("epoch  " + "  ".join(f"{k:>24s}" for k, _, _ in COMPONENTS))
    for e in range(a.epochs):
        sl = slice(round(e * spe), round((e + 1) * spe))
        cells = [f"{statistics.median(r[k] for r in A[sl]):11.3f} / {statistics.median(r[k] for r in B[sl]):10.3f}"
                 for k, _, _ in COMPONENTS]
        print(f"{e + 1:5d}  " + "  ".join(cells))
    print(f"\nfirst step with 50-step rolling medians differing by > {a.rel:.0%} for 50 consecutive steps:")
    for k, _, _ in COMPONENTS:
        s = first_divergence([r[k] for r in A], [r[k] for r in B], a.rel)
        print(f"  {k:22s} {'none' if s is None else f'step {s} (epoch {s / spe:.2f})'}")
    ga, gb = grad_norms(a.a_out), grad_norms(a.b_out)
    print("\nTrainer grad_norm (before clipping to 1.0):")
    for (ea, xa), (eb, xb) in zip(ga, gb):
        print(f"  epoch {ea:5.2f}: {a.a_name} {xa:10.1f}   {a.b_name} {xb:10.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.titlelocation": "left"})
    n = len(COMPONENTS) + (1 if ga else 0)
    fig, axs = plt.subplots(n, 1, figsize=(8, 2.1 * n + 0.6), sharex=True, facecolor=SURFACE)
    x = [r["step"] / spe for r in A]
    for i, ax in enumerate(axs):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for ep in range(1, a.epochs):
            ax.axvline(ep, color=GRID, linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    for ax, (k, label, logy) in zip(axs, COMPONENTS):
        ends = {}
        for R, col, name in ((A, C_A, a.a_name), (B, C_B, a.b_name)):
            ys = [r[k] for r in R]
            ax.plot(x, ys, color=col, linewidth=0.5, alpha=0.2)
            sm = rolling_median(ys, 50)
            ax.plot(x, sm, color=col, linewidth=2, label=name)
            ends[name] = sm[-1]
        hi = max(ends, key=ends.get)  # the higher line's label goes above, the other below: no overlap
        for name, y in ends.items():
            ax.annotate(name, (x[-1], y), xytext=(6, 6 if name == hi else -6), textcoords="offset points",
                        va="center", color=INK2, fontsize=8, annotation_clip=False)
        if logy and min(min(r[k] for r in A), min(r[k] for r in B)) > 0:
            ax.set_yscale("log")
        elif logy:
            ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(label)
    axs[0].legend(loc="upper right", frameon=False, fontsize=8, title="training seed (line = 50-step median)",
                  title_fontsize=8)
    if ga:
        ax = axs[-1]
        for G, col, name in ((ga, C_A, a.a_name), (gb, C_B, a.b_name)):
            ax.plot([e for e, _ in G], [g for _, g in G], color=col, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=name)
        ax.set_yscale("log")
        ax.set_title("Trainer grad_norm before clipping (logged every 500 steps)")
    axs[-1].set_xlabel("epoch")
    axs[-1].set_xlim(0, a.epochs + 0.05)
    if a.title:
        fig.suptitle(f"{a.title}: {a.a_name} vs {a.b_name}", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
