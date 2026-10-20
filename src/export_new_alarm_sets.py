"""Per-capture new-alarm table for B = 120 (the sets behind the gains and losses in results/final_tables.md).

One row per vehicle x position x condition x seed x evaluated attack capture: whether a new 0->1 alarm ends in
(first attack frame, last attack frame + 1 s]. Re-scores the stored protocol deterministically; no training choices
change. Writes results/new_alarm_sets_B120.csv and prints the gains/losses recomputed from it.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import alloc_v3 as A  # noqa: E402
from alloc_models import eval_data, seg_windows  # noqa: E402
from eval_v3 import onsets  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDS = ["T_main", "S_sameC", "S_allC", "T_split"]


def rows_for(v, u, cond, seed):
    sc, _ = A.scorer(v, u, 120.0, 0.5, cond, seed)
    _, C = A.segments(120.0, 0.5, cond)
    tau = sc(seg_windows(v, u, 60.0, *C)).max()
    d, meta = eval_data(v, u, 60.0)
    al = sc(d["Xatt"]) > tau
    out = []
    for i, info in enumerate(meta["attack_info"]):
        m = np.flatnonzero(d["att_file"] == i)
        if m.size == 0:
            continue
        m = m[np.argsort(d["att_ts"][m], kind="stable")]
        te = d["att_te"][m]
        hit = bool((onsets(al[m]) & (te > info["first_attack"]) & (te <= info["last_attack"] + 1.0)).any())
        out.append({"target": v, "u": u, "cond": cond, "seed": seed, "file": info["file"], "new_alarm": hit})
    return out


if __name__ == "__main__":
    rows = []
    for v in ("forester", "impala", "silverado", "traverse"):
        for u in (190.0, 0.0):
            for cond in CONDS:
                for s in (0, 1, 2):
                    rows += rows_for(v, u, cond, s)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(ROOT, "results/new_alarm_sets_B120.csv"), index=False)
    w = t.pivot_table(index=["target", "u", "seed", "file"], columns="cond", values="new_alarm").astype(bool)
    for a, b in (("S_sameC", "S_allC"), ("T_main", "T_split")):
        lost = (w[a] & ~w[b]).groupby(["target", "u", "seed"]).sum()
        gained = (~w[a] & w[b]).groupby(["target", "u", "seed"]).sum()
        g = pd.DataFrame({"lost": lost, "gained": gained}).groupby(["target", "u"]).agg(["min", "max"])
        print(f"{a}->{b}\n{g}\n")
