"""Extra donor features for the revision analyses.

1. Donor windows against a profile of length L (data/features_v2/source/<vehicle>_P<L>.npz), as in
   nested_features.source_job: profile [250 - L, 250) s of the donor's budget capture, windows from [250, 430) s.
   L = 150 is needed for the matched-profile runs at B = 300.
2. Donor calibration windows (data/features_v2/source_cal/<vehicle>_P<L>.npz): the same profile, windows from
   430 s to the end of the donor's budget capture. They never overlap the windows the source bank is drawn from.
Usage: python3 src/source_extra.py --lengths 15,60,150,250
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from canio import load_ctt_csv  # noqa: E402
from experiment import CTT, ROOT, load_roles  # noqa: E402
from features import build_profile, slice_time  # noqa: E402
from nested_features import ANCHOR, SRC_SPAN, seg_windows, source_job  # noqa: E402

VEH = ["forester", "impala", "silverado", "traverse"]
OUT = os.path.join(ROOT, "data/features_v2/source_cal")


def cal_job(v, lengths):
    roles = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))
    cap = load_ctt_csv(os.path.join(CTT, roles[v]["budget"]))
    t0 = cap["t"][0]
    os.makedirs(OUT, exist_ok=True)
    for L in lengths:
        out = os.path.join(OUT, f"{v}_P{L:g}.npz")
        if os.path.exists(out):
            continue
        prof = build_profile(slice_time(cap, t0 + ANCHOR - L, t0 + ANCHOR))
        w = seg_windows(cap, prof, t0 + ANCHOR + SRC_SPAN, cap["t"][-1] + 1e-6)
        np.savez_compressed(out, X=w["X"])
        print(v, L, w["X"].shape, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="15,60,150,250")
    a = ap.parse_args()
    lengths = [float(x) for x in a.lengths.split(",")]
    for v in VEH:
        source_job((v, lengths))
        cal_job(v, lengths)
