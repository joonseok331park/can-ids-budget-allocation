"""Exploratory allocation grid: how should a target budget B be split between profile, training and calibration,
and can other vehicles' normal residuals replace the target's training windows?

For each vehicle, offset o, budget B, profile length Lp (< B), with R = B - Lp seconds left:
  target-only   Lf = q * R for q in {0.25, 0.5, 0.75}, Lc = R - Lf
  source-only   Lf = 0, n_src source windows, Lc = R
  source+target Lf = 0.5 * R plus n_src source windows, Lc = 0.5 * R
  source-only (matched n) Lf = 0, n_src = number of target windows in the target_q50 arm, Lc = R
    (separates "whose data" from "how many training windows")
Seeds re-run IF (and the source sample). Output: results/runs/alloc_grid.csv
"""
from __future__ import annotations
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from alloc_models import run  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_SRC = 213


def arms(B, Lp):
    R = B - Lp
    out = [("target_q25", 0.25 * R, 0), ("target_q50", 0.5 * R, 0), ("target_q75", 0.75 * R, 0),
           ("source_only", 0.0, N_SRC), ("source_plus_q50", 0.5 * R, N_SRC),
           ("source_only_matched", 0.0, max(int((0.5 * R - 1.0) / 0.5) + 1, 3))]
    return [(name, lf, n) for name, lf, n in out if (lf == 0 or lf >= 1.0) and R - lf >= 1.0]


def vehicle_job(args):
    v, offsets, budgets, plens, seeds = args
    rows = []
    for o in offsets:
        for B in budgets:
            for Lp in plens:
                if Lp >= B:
                    continue
                for name, lf, n in arms(B, Lp):
                    for s in seeds:
                        rows.append(run(v, o, B, Lp, lf, n, s, tag=name))
        print(f"{v} o={o:g} done", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="forester,impala,silverado,traverse")
    ap.add_argument("--offsets", default="0,150,300")
    ap.add_argument("--budgets", default="60,120")
    ap.add_argument("--plens", default="15,30,60,90")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=ROOT + "/results/runs/alloc_grid.csv")
    a = ap.parse_args()
    f = lambda s: [float(x) for x in s.split(",")]  # noqa: E731
    jobs = [(v, f(a.offsets), f(a.budgets), f(a.plens), [int(x) for x in a.seeds.split(",")]) for v in a.targets.split(",")]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(vehicle_job, jobs) for r in rs]
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("rows", len(rows))
