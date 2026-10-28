"""Positions every 30 s at B = 120 (results/runs/rev_sweep.csv, rev_sweepsrc.csv): per vehicle, the seed-paired
difference at each position for three contrasts. Each dot is one position (mean over seeds 0-2); the bar marks the
median over positions. Usage: python3 src/fig_sweep.py results/figures/fig_sweep_B120.pdf
"""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from figstyle import WIDTH, setup, tt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = [("S_sameC", "T_main", "AP", "AP"), ("T_split", "T_main", "held_pos_frac", "held-out alarm share"),
          ("S_srcC", "T_split", "held_pos_frac", "held-out alarm share")]
VEH = ["forester", "impala", "silverado", "traverse"]


def main(out):
    plt = setup()
    d = pd.concat([pd.read_csv(os.path.join(ROOT, "results/runs/rev_sweep.csv")),
                   pd.read_csv(os.path.join(ROOT, "results/runs/rev_sweepsrc.csv"))])
    fig, axes = plt.subplots(1, len(PANELS), figsize=(WIDTH, 2.05), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.03, wspace=0.06)
    rng = np.random.default_rng(1)
    for ax, (a, b, m, lab) in zip(axes, PANELS):
        q = d[d.cond.isin([a, b])].pivot_table(index=["target", "u", "seed"], columns="cond", values=m)
        p = (q[a] - q[b]).groupby(["target", "u"]).mean()
        ax.axhline(0, color="0.55", lw=0.5, zorder=0)
        for i, v in enumerate(VEH):
            y = p.loc[v].values
            x = i + rng.uniform(-0.17, 0.17, y.size)
            ax.plot(x, y, "o", ms=2.6, mfc="white", mec="black", mew=0.55, zorder=2)
            ax.plot([i - 0.28, i + 0.28], [np.median(y)] * 2, color="black", lw=1.4, zorder=3)
        ax.set_xticks(range(len(VEH)))
        ax.set_xticklabels([v.capitalize() for v in VEH], fontsize=6.6)
        ax.set_xlim(-0.6, len(VEH) - 0.4)
        ax.set_title(f"{tt(a)} $-$ {tt(b)}")
        ax.set_ylabel(f"difference in {lab}")
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.tick_params(axis="x", length=0)
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1])
