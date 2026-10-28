"""Condition summaries quoted in the paper, from the stored runs.

results/v3_headline.md  CTT: median [min, max] over the 8 vehicle x position cells per budget and condition
results/road_summary.md ROAD: per position and condition, AP by attack type and alarm share per held-out capture
Usage: python3 src/summary_tables.py
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["T_main", "S_sameC", "S_allC", "T_split", "U_allC"]


def med_range(x, fmt):
    return f"{np.median(x):{fmt[0]}} [{np.min(x):{fmt[1]}}, {np.max(x):{fmt[1]}}]"


def headline():
    d = pd.read_csv(os.path.join(ROOT, "results/runs/v3_main.csv"))
    d["excess5"] = d["new_within_5s"] - d["prefix_virtual_new_5s"]
    cols = {"AP": (".3f", ".2f"), "held_pos_frac": (".3f", ".2f"), "held_fa_per_h": (".0f", ".0f"),
            "on_at_start_frac": (".3f", ".2f"), "new_within_5s": (".3f", ".2f"), "excess5": (".3f", ".2f")}
    cells = d.groupby(["B", "cond", "target", "u"])[list(cols)].mean().reset_index()
    rows = []
    for B in sorted(cells.B.unique()):
        for c in ORDER:
            g = cells[(cells.B == B) & (cells.cond == c)]
            rows.append({"B": int(B), "cond": c, **{k: med_range(g[k], f) for k, f in cols.items()}})
    lines = ["# Summary by condition (median [min, max] over 4 vehicles × 2 positions = 8 cells; "
             "each cell is the mean over 3 seeds)", "",
             "excess5 = share with a new alarm within 5 s minus the same share for virtual start points in the "
             "normal prefixes of the attack captures.", "",
             pd.DataFrame(rows).to_markdown(index=False)]
    open(os.path.join(ROOT, "results/v3_headline.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def road():
    d = pd.read_csv(os.path.join(ROOT, "results/runs/road_v3.csv"))
    d["excess5"] = d["new_within_5s"] - d["prefix_virtual_new_5s"]
    cols = {"AP": "AP", "held_pos_frac": "held pos", "held_fa_per_h": "held ev/h", "on_at_start_frac": "on@start",
            "new_alarm_frac": "new", "new_within_5s": "new≤5s", "prefix_virtual_new_5s": "chance≤5s prefix",
            "held_virtual_new_5s": "chance≤5s held", "excess5": "excess5", "accel_pos_frac": "accelerator pos"}
    main = d.groupby(["u", "cond"])[list(cols)].mean().reset_index().rename(columns=cols)
    ap = [c for c in d.columns if c.startswith("AP__")]
    held = [c for c in d.columns if c.startswith("held_pos__")]
    by_attack = d.groupby("cond")[ap].mean().T.round(3)
    by_attack.index = [c[len("AP__"):] for c in by_attack.index]
    by_capture = d.groupby("cond")[held].mean().T.round(3)
    by_capture.index = [c[len("held_pos__"):] for c in by_capture.index]
    for t in (by_attack, by_capture):
        t.columns.name = None
    bank = d.loc[d.donor_rows.notna(), "donor_rows"].iloc[0]
    lines = ["# ROAD external check (B = 120, P = 50 %, ROAD rules fixed before the results, mean over 3 seeds)", "",
             f"The source is all four CTT vehicles (213 bank rows: {describe_bank(bank)}). Held-out normal data are "
             f"11 normal captures ({d.held_hours.iloc[0]:.2f} h in total). The 4 accelerator captures have no injection "
             "interval, so they are left out of AP and new alarms and only their alarm share over all windows is "
             "reported.", "",
             main.to_markdown(index=False, floatfmt=".3f"), "",
             "## AP by attack type (mean of the two positions)", "", by_attack.to_markdown(), "",
             "## Positive-window share per held-out normal capture (mean of the two positions)", "",
             by_capture.to_markdown()]
    open(os.path.join(ROOT, "results/road_summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def describe_bank(s):
    n = dict(x.split(":") for x in s.split(";"))
    big = [v for v in n if int(n[v]) == max(int(k) for k in n.values())]
    rest = [v for v in n if v not in big]
    return f"{n[big[0]]} {big[0].capitalize()}, {n[rest[0]]} from each of the other three"


if __name__ == "__main__":
    headline()
    road()
