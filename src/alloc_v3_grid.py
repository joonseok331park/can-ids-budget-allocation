"""Runs the fixed-protocol allocation experiments.

--what main      B in {30, 120, 300} x u in {250 - B/2, 0} x {T_main, S_sameC, S_allC, T_split, U_allC} x seeds 0-2
--what controls  B = 120, both positions: S_small, S_large_fixed, S_large_auto (+ T_main for reference)
--what caldiag   B = 120, both positions: S_allC with C spread to n(T_main C) windows; T_main, S_sameC, S_allC,
                 T_split and U_allC with the 95 % quantile threshold
--what pratio    B = 120, both positions: p in {0.25, 0.75} for the four IF conditions and U_allC (optional sensitivity analysis)
Output: results/runs/v3_<what>.csv
"""
from __future__ import annotations
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import alloc_v3 as A  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VEH = ["forester", "impala", "silverado", "traverse"]
MAIN = ["T_main", "S_sameC", "S_allC", "T_split", "U_allC"]


def positions(B):
    return [250.0 - B / 2, 0.0]


def job(args):
    what, v, seeds = args
    rows = []
    if what == "main":
        for B in (30.0, 120.0, 300.0):
            for u in positions(B):
                for c in MAIN:
                    for s in seeds:
                        rows.append(A.run(v, u, B, c, s))
    elif what == "controls":
        for u in positions(120.0):
            for c in ("T_main", "S_small", "S_large_fixed", "S_large_auto"):
                for s in seeds:
                    rows.append(A.run(v, u, 120.0, c, s))
    elif what == "caldiag":
        for u in positions(120.0):
            for s in seeds:
                n_short = int((0.2 * 120 - 1) / 0.5) + 1  # windows in T_main's C (24 s)
                rows.append(A.run(v, u, 120.0, "S_allC", s, c_pick=("spread", n_short), tag="S_allC_spread"))
                for c in ("T_main", "S_sameC", "S_allC", "T_split", "U_allC"):
                    rows.append(A.run(v, u, 120.0, c, s, thr="q95", tag=f"{c}_q95"))
    elif what == "pratio":
        for u in positions(120.0):
            for p in (0.25, 0.75):
                for c in MAIN:
                    for s in seeds:
                        rows.append(A.run(v, u, 120.0, c, s, p=p, tag=f"{c}_p{int(p * 100)}"))
    print(f"{what} {v} done", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=["main", "controls", "caldiag", "pratio"])
    ap.add_argument("--targets", default=",".join(VEH))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(job, [(a.what, v, seeds) for v in a.targets.split(",")]) for r in rs]
    out = ROOT + f"/results/runs/v3_{a.what}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("rows", len(rows), "->", out)
