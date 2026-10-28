"""Tables for the fixed-protocol allocation study (results/runs/v3_main.csv) -> results/v3_summary.md

1. Per vehicle x B x position: seed means for each condition (AP, held-out positive-window share, held alarm events
   per hour, share of attack captures with the alarm already on at attack start, new-alarm share, new alarm within
   5 s, chance rate of a new alarm within 5 s at virtual starts on held-out normal and on attack-capture prefixes).
2. Seed-paired differences per vehicle x position for the planned contrasts:
     model swap at the same C      S_sameC - T_main
     longer C, same model          S_allC - S_sameC
     source + long C vs target-only reallocation   S_allC - T_split
     IF vs fixed score at the same C               S_allC - U_allC
     target-only reallocation      T_split - T_main
Seeds are algorithmic repeats on the same data; positions overlap at B = 300.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLS = {"AP": "AP", "held_pos_frac": "held pos", "held_fa_per_h": "held ev/h", "on_at_start_frac": "on@start",
        "new_alarm_frac": "new", "new_within_5s": "new≤5s", "held_virtual_new_5s": "chance≤5s held",
        "prefix_virtual_new_5s": "chance≤5s prefix", "prefix_pos_frac": "prefix pos"}
CONTRASTS = [("S_sameC", "T_main", "model swap, same C"), ("S_allC", "S_sameC", "longer C, same model"),
             ("S_allC", "T_split", "source+long C vs target realloc"), ("S_allC", "U_allC", "IF vs fixed score"),
             ("T_split", "T_main", "target-only realloc")]
ORDER = ["T_main", "S_sameC", "S_allC", "T_split", "U_allC"]


def main(csv=ROOT + "/results/runs/v3_main.csv", out=ROOT + "/results/v3_summary.md"):
    d = pd.read_csv(csv)
    d["pos"] = np.where(d.u == 0, "start", "old")
    lines = ["# Allocation study results (fixed protocol, CTT)", "",
             "Values are means over 3 seeds. held = other attack-free captures kept out entirely, prefix = normal part of an attack capture before its first attack. "
             "on@start = share of attack captures whose alarm was already on at attack start, new = share with a new 0->1 alarm during the attack, "
             "new≤5s = share with a new alarm within 5 s, chance≤5s = share of virtual start points in normal traffic followed by a new alarm within 5 s.", ""]
    for B in sorted(d.B.unique()):
        lines += [f"## B = {B:g} s", ""]
        g = d[d.B == B].groupby(["target", "pos", "cond"])[list(COLS)].mean().reset_index()
        g["cond"] = pd.Categorical(g.cond, ORDER, ordered=True)
        g = g.sort_values(["target", "pos", "cond"]).rename(columns=COLS)
        lines += [g.to_markdown(index=False, floatfmt=".3f"), ""]
    lines += ["## Pre-specified contrasts (mean seed-paired difference, per vehicle and position)", ""]
    for metric in ("AP", "held_pos_frac", "new_within_5s", "new_alarm_frac", "on_at_start_frac"):
        rows = []
        for B in sorted(d.B.unique()):
            for (t, pos), g in d[d.B == B].groupby(["target", "pos"]):
                p = g.pivot_table(index="seed", columns="cond", values=metric)
                r = {"B": int(B), "target": t, "pos": pos}
                for a, b, name in CONTRASTS:
                    r[name] = float((p[a] - p[b]).mean())
                rows.append(r)
        lines += [f"### {COLS[metric]}", "", pd.DataFrame(rows).to_markdown(index=False, floatfmt="+.3f"), ""]
    # sign counts over the 8 vehicle x position cells per B (lower is better for held pos and on@start)
    better_low = {"held_pos_frac", "on_at_start_frac"}
    srows = []
    for metric in ("AP", "held_pos_frac", "new_within_5s", "on_at_start_frac"):
        for B in sorted(d.B.unique()):
            for a, b, name in CONTRASTS:
                diffs = []
                for (t, pos), g in d[d.B == B].groupby(["target", "pos"]):
                    p = g.pivot_table(index="seed", columns="cond", values=metric)
                    diffs.append(float((p[a] - p[b]).mean()))
                diffs = np.array(diffs)
                good = (diffs < -0.01) if metric in better_low else (diffs > 0.01)
                bad = (diffs > 0.01) if metric in better_low else (diffs < -0.01)
                srows.append({"metric": COLS[metric], "B": int(B), "contrast": f"{a} − {b}",
                              "first better": int(good.sum()), "about equal (±0.01)": int((~good & ~bad).sum()),
                              "second better": int(bad.sum()), "median diff": round(float(np.median(diffs)), 3),
                              "min": round(float(diffs.min()), 3), "max": round(float(diffs.max()), 3)})
    lines += ["## Sign counts (4 vehicles × 2 positions = 8 cells, seed-mean differences)", "",
              "Lower is better for held pos and on@start. Differences within ±0.01 count as about equal.", "",
              pd.DataFrame(srows).to_markdown(index=False), ""]
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main(*sys.argv[1:])
