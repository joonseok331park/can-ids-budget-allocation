"""Profile and feature diagnostics for short/reference profiles (no IF).

Per vehicle x B x profile condition:
  profile size, IDs seen once (no period estimate), share of IDs whose IAT scale hit a floor,
  held-out coverage (share of held-out frames whose ID is in the profile),
  F-window features that are constant (no information for IF splits),
  median of each residual feature on held-out normal (first 300 s) -> how "abnormal" normal looks
Output: results/profile_diagnostics.csv, results/feature_heldout_medians.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from canio import load_ctt_csv  # noqa: E402
from experiment import CTT, ROOT, load_roles  # noqa: E402
from features import FEATURES, build_profile, slice_time, window_features  # noqa: E402

ANCHOR = 250.0


def main(budgets=(30, 120, 300)):
    roles = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))
    rows, med_rows = [], []
    for v in sorted(roles):
        b = load_ctt_csv(os.path.join(CTT, roles[v]["budget"]))
        h = load_ctt_csv(os.path.join(CTT, roles[v]["reference"]))
        t0 = b["t"][0]
        h300 = slice_time(h, h["t"][0], h["t"][0] + 300)
        for B in budgets:
            F = (t0 + ANCHOR, t0 + ANCHOR + 0.3 * B)
            for pc, P in (("short", (t0 + ANCHOR - B / 2, t0 + ANCHOR)), ("ref", (t0, t0 + ANCHOR))):
                seg = slice_time(b, *P)
                prof = build_profile(seg)
                ids, cnt = np.unique(seg["aid"], return_counts=True)
                floor = np.isclose(prof.iat_scale, np.maximum(0.05 * prof.iat_med, 1e-4)) | (prof.iat_scale <= 1e-4 + 1e-12)
                cover = float(np.isin(h300["aid"], prof.vocab).mean())
                fw = window_features(slice_time(b, *F), prof, 1.0, 0.5, t_origin=F[0], t_stop=F[1])
                nuniq = [len(np.unique(fw["X"][:, j])) for j in range(len(FEATURES))]
                const = [f for f, n in zip(FEATURES, nuniq) if n <= 1]
                hw = window_features(h300, prof, 1.0, 0.5)
                rows.append({"vehicle": v, "B": B, "profile": pc, "P_len_s": P[1] - P[0], "profile_ids": int(prof.vocab.size),
                             "ids_seen_once": int((cnt == 1).sum()), "ids_seen_lt5": int((cnt < 5).sum()),
                             "iat_floor_share": round(float(floor.mean()), 3),
                             "heldout_frame_id_coverage": round(cover, 5),
                             "n_F_windows": int(len(fw["X"])), "constant_F_features": ";".join(const),
                             "n_constant_F_features": len(const)})
                med = np.median(hw["X"], 0)
                trm = np.median(fw["X"], 0)
                for f, a, c in zip(FEATURES, med, trm):
                    med_rows.append({"vehicle": v, "B": B, "profile": pc, "feature": f,
                                     "F_median": round(float(c), 4), "heldout_median": round(float(a), 4)})
        print("done", v, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results/profile_diagnostics.csv"), index=False)
    pd.DataFrame(med_rows).to_csv(os.path.join(ROOT, "results/feature_heldout_medians.csv"), index=False)


if __name__ == "__main__":
    main()
