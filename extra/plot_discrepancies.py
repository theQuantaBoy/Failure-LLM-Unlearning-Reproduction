# Note: this script reads per-step training logs that are not shipped in this repository (they stay on the Modal Volume).
"""
extra/plot_discrepancies.py — figure for FINDINGS.md §4: per-step loss components (wrapper W6) of the five
BOOKS training runs, rolling median over 25 steps, epoch boundaries every 553 steps. CPU only, reads local files.

    .venvs/paper/bin/python -m extra.plot_discrepancies --out figures/books_loss_components.png
"""

import argparse
import json
import math

RUNS = [  # (label, file, colour = categorical slots 1-5 of the dataviz default palette, fixed order)
    ("NPO_KLR s42", "dl_acc2/train_logs/books_npo_klr_loss_components.jsonl", "#2a78d6"),
    ("NPO_KLR+SURE s42", "dl_acc2/train_logs/books_npo_klr_sure_loss_components.jsonl", "#eb6834"),
    ("NPO_KLR+SURE s43", "dl_acc3/loss_s43.jsonl", "#1baf7a"),
    ("GA_GDR s42", "dl_acc2/train_logs/books_ga_gdr_loss_components.jsonl", "#eda100"),
    ("GA_GDR+SURE s42", "dl_acc2/train_logs/books_ga_gdr_sure_loss_components.jsonl", "#e87ba4"),
]
STEPS_PER_EPOCH = 553


def rolling_median(xs, w=25):
    out = []
    for i in range(len(xs)):
        win = sorted(v for v in xs[max(0, i - w + 1):i + 1] if v is not None and math.isfinite(v))
        out.append(win[len(win) // 2] if win else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/books_loss_components.png")
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    panels = [("ce_forget", "Forget CE (nats/token)"), ("ce_retain", "Retain CE (nats/token)"),
              ("kl_abs", "|KLR term before α| (KLR runs)")]
    for label, path, colour in RUNS:
        rows = [json.loads(line) for line in open(path)]
        steps = [r["step"] for r in rows]
        for ax, (key, _) in zip(axes, panels):
            if key == "kl_abs":
                if rows[0]["kl_raw"] is None:
                    continue
                vals = [abs(r["kl_raw"]) if r["kl_raw"] is not None else None for r in rows]
            else:
                vals = [r[key] for r in rows]
            ax.plot(steps, rolling_median(vals), color=colour, lw=2, label=label)
    for ax, (_, title) in zip(axes, panels):
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("training step (epoch boundaries dotted)")
        for e in range(1, 5):
            ax.axvline(e * STEPS_PER_EPOCH, color="#c3c2b7", lw=0.8, ls=":")
        ax.grid(axis="y", color="#e6e5dd", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[2].annotate("raw-logit 'KL' (iterative.py:173-178):\nnegative in 2578/2765 steps, ~1e18",
                     xy=(1500, 1.3e18), xytext=(700, 1e9), fontsize=8, color="#555",
                     arrowprops=dict(arrowstyle="->", color="#999"))
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("BOOKS training runs: per-step loss components (rolling median of 25 steps)", fontsize=12, x=0.01,
                 ha="left")
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
