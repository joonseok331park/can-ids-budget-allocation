"""Effect figure: source effect on AP, Delta(P) = AP(r=1) - AP(r=0), per vehicle, budget and profile condition.

Two panels (B = 30 s, B = 120 s) share one y scale. Each dot is one seed (paired by seed), the bar is the seed mean.
Black: short profile, grey: reference profile.
Usage: python3 src/fig_effect.py results/runs/v2_main.csv results/fig_effect.png [metric]
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, MUTED, GRID, BASE = "black", "0.3", "0.5", "0.88", "0.75"
COL = {"short": "black", "ref": "0.55"}
LABEL = {"short": "short profile (B/2 s)", "ref": "reference profile (250 s)"}


def deltas(d, metric):
    p = d.pivot_table(index=["target", "B", "profile", "seed"], columns="r", values=metric).reset_index()
    p["delta"] = p[1] - p[0]
    return p


def main(csv, out, metric="AP"):
    p = deltas(pd.read_csv(csv), metric)
    vehicles = sorted(p.target.unique())
    budgets = sorted(p.B.unique())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": BASE,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.4 * len(budgets), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    lim = max(0.02, float(np.nanmax(np.abs(p.delta))) * 1.15)
    for ax, B in zip(axes, budgets):
        ax.axhline(0, color=BASE, lw=1, zorder=1)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        for i, v in enumerate(vehicles):
            for j, pc in enumerate(("short", "ref")):
                x = i + (j - 0.5) * 0.34
                g = p[(p.target == v) & (p.B == B) & (p.profile == pc)].delta.values
                if g.size == 0:
                    continue
                ax.bar(x, g.mean(), width=0.28, color=COL[pc], alpha=0.35, zorder=2,
                       label=LABEL[pc] if (i == 0 and ax is axes[0]) else None)
                ax.scatter(np.full(g.size, x), g, s=16, color=COL[pc], edgecolor="white", lw=0.8, zorder=3)
        ax.set_xticks(range(len(vehicles)))
        ax.set_xticklabels([v.capitalize() for v in vehicles])
        ax.set_title(f"B = {B} s", color=INK, fontsize=9.5, loc="left")
        ax.set_ylim(-lim, lim)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(f"Δ{metric} = {metric}(r=1) − {metric}(r=0)")
    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=200, facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "AP")
