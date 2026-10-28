"""Dot plot of the allocation study: rows = conditions, columns = metrics, one row of panels per vehicle.

E5 uses the held-out normal reference. Each symbol is the seed mean at one budget position (filled: u = 250 - B/2, open: u = 0); a thin line joins the two
positions. Usage: python3 src/fig_alloc.py results/runs/v3_main.csv 120 results/figures/fig_alloc_B120.png (or .pdf)
"""
import sys

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

from figstyle import WIDTH, setup, tt

ORDER = ["T_main", "S_sameC", "S_allC", "T_split", "U_allC"]
METRICS = [("AP", "AP", (0, 1)), ("held_pos_frac", "held-out alarm share", (0, 1)),
           ("excess5", "$E_5$ (held-out reference)", (-1, 1))]


def main(csv, B, out):
    plt = setup()
    B = float(B)
    d = pd.read_csv(csv)
    d = d[d.B == B]
    d["pos"] = np.where(d.u == 0, "start", "old")
    # E5: share of attack captures with a new alarm within 5 s minus the same share for virtual starts in the held-out
    # normal capture
    d["excess5"] = d["new_within_5s"] - d["held_virtual_new_5s"]
    g = d.groupby(["target", "pos", "cond"])[[m for m, _, _ in METRICS]].mean().reset_index()
    vehicles = sorted(g.target.unique())

    fig, axes = plt.subplots(len(vehicles), len(METRICS), figsize=(WIDTH, 0.98 * len(vehicles) + 0.55),
                             sharex="col", sharey=True, squeeze=False, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.035, wspace=0.03, hspace=0.02)
    y = {c: i for i, c in enumerate(ORDER[::-1])}
    for r, v in enumerate(vehicles):
        for c, (m, title, lim) in enumerate(METRICS):
            ax = axes[r, c]
            for k in y.values():
                ax.axhline(k, color="0.8", lw=0.4, dashes=(1, 1.6), zorder=0)
            if lim[0] < 0:
                ax.axvline(0, color="0.55", lw=0.5, zorder=0)
            for cond in ORDER:
                sel = g[(g.target == v) & (g.cond == cond)].set_index("pos")[m]
                if "old" in sel and "start" in sel:
                    ax.plot([sel["old"], sel["start"]], [y[cond]] * 2, color="0.6", lw=0.8, zorder=1,
                            solid_capstyle="butt")
                if "start" in sel:
                    ax.plot(sel["start"], y[cond], "o", ms=5.0, mfc="white", mec="black", mew=0.7, zorder=3)
                if "old" in sel:
                    ax.plot(sel["old"], y[cond], "o", ms=3.6, mfc="black", mec="black", mew=0, zorder=4)
            ax.set_xlim(lim[0] - (0.02 if lim[0] < 0 else 0.03), lim[1] + 0.03)
            ax.set_ylim(-0.6, len(ORDER) - 0.4)
            ax.xaxis.set_major_locator(MultipleLocator(0.5 if lim[0] < 0 else 0.2))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}$"))
            ax.tick_params(axis="y", length=0)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_yticks(list(y.values()))
                ax.set_yticklabels([tt(k) for k in y])
                ax.set_ylabel(v.capitalize(), labelpad=5)
    hold = dict(ls="none", marker="o")
    handles = [plt.Line2D([], [], ms=3.6, mfc="black", mec="black", mew=0, **hold),
               plt.Line2D([], [], ms=5.0, mfc="white", mec="black", mew=0.7, **hold)]
    labels = [f"$u = 250 - B/2$ ({250 - B / 2:g} s)", "$u = 0$"]
    fig.legend(handles, labels, loc="outside upper center", ncol=2, handlelength=1.0, handletextpad=0.5)
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
