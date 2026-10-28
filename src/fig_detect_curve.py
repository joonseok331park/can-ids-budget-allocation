"""Detection curve against two normal references: share of attack captures with a new alarm within t s of the first
attack frame, next to the share of virtual start points followed by a new alarm within t s, in the held-out normal
capture and in the attack captures' normal prefixes. One panel per condition; median over the 8 vehicle x position
cells, vertical bars give the range.
Usage: python3 src/fig_detect_curve.py results/runs/v3_main.csv 120 results/figures/fig_detect_curve_B120.png (or .pdf)
"""
import sys

import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullLocator

from figstyle import WIDTH, setup, tt

T = [1, 2, 5, 10, 30, 60]
ORDER = ["T_main", "S_sameC", "S_allC", "T_split", "U_allC"]
SERIES = [  # column prefix, colour, line dashes, marker, marker face, x offset factor, legend label
    ("new_within_", "black", (), "o", "black", 0.9, r"attack captures, $R_{\mathrm{att},t}$"),
    ("held_virtual_new_", "0.35", (3.2, 1.6), "s", "white", 1.0, r"held-out reference"),
    ("prefix_virtual_new_", "0.6", (1.0, 1.4), "^", "white", 1.11, r"attack-prefix reference"),
]


def main(csv, B, out):
    plt = setup()
    d = pd.read_csv(csv)
    d = d[d.B == float(B)]
    cols = [f"{p}{t}s" for p, *_ in SERIES for t in T]
    cells = d.groupby(["target", "u", "cond"])[cols].mean().reset_index()

    fig, axes = plt.subplots(1, len(ORDER), figsize=(WIDTH, 1.95), sharey=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.04)
    x = np.array(T, float)
    for ax, cond in zip(axes, ORDER):
        g = cells[cells.cond == cond]
        for pref, col, dash, mk, face, off, lab in SERIES:
            v = g[[f"{pref}{t}s" for t in T]].to_numpy()
            ax.vlines(x * off, np.nanmin(v, 0), np.nanmax(v, 0), color=col, lw=0.45, alpha=0.55, zorder=1)
            ax.plot(x * off, np.nanmedian(v, 0), color=col, dashes=dash, marker=mk, ms=3.0, mfc=face, mec=col,
                    mew=0.7, zorder=3, label=lab)
        ax.set_xscale("log")
        ax.set_xlim(0.72, 85)
        ax.xaxis.set_major_locator(FixedLocator(T))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([str(t) for t in T])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}$"))
        ax.set_title(tt(cond))
    axes[0].set_ylabel("share with a new alarm")
    fig.supxlabel("$t$ (s)", fontsize=8)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=3)
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
