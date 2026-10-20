"""Clock/period shift between captures of the same vehicle (CTT).

For each vehicle, take the 5 most frequent IDs in the budget (attack-free) capture. For every other capture of
that vehicle (held-out attack-free, accessory, attack captures using only frames before the first attack frame),
compute each ID's median inter-arrival time and its ratio to the budget capture. A ratio far from 1 for all
periodic IDs at once points to a timestamp/clock difference between recording sessions rather than a change in
one ECU.  Output: results/period_shift.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from canio import ensure_sorted, load_ctt_csv  # noqa: E402
from experiment import CTT, ROOT  # noqa: E402

m = pd.read_csv(os.path.join(ROOT, "results/manifest_ctt.csv")).drop_duplicates("sha256")
rows = []
for v, g in m.groupby("vehicle"):
    af = g[g.kind == "attack-free"].sort_values("duration_s", ascending=False)
    base = load_ctt_csv(os.path.join(CTT, af.path.iloc[0]))
    ids, cnt = np.unique(base["aid"], return_counts=True)
    top = ids[np.argsort(-cnt)[:5]]
    bmed = {a: np.median(np.diff(base["t"][base["aid"] == a])) for a in top}
    for _, r in g.iterrows():
        cap = ensure_sorted(load_ctt_csv(os.path.join(CTT, r.path)))
        att = np.flatnonzero(cap["label"] == 1)
        n = att[0] if att.size else cap["t"].size
        t, aid = cap["t"][:n], cap["aid"][:n]
        ratios = []
        for a in top:
            ta = t[aid == a]
            if ta.size > 20:
                ratios.append(np.median(np.diff(ta)) / bmed[a])
        rows.append({"vehicle": v, "path": r.path, "kind": r.kind, "set": r.set, "subset": r.subset,
                     "frames_used": int(n), "n_ids": len(ratios),
                     "period_ratio_median": float(np.median(ratios)) if ratios else np.nan,
                     "period_ratio_min": float(np.min(ratios)) if ratios else np.nan,
                     "period_ratio_max": float(np.max(ratios)) if ratios else np.nan})
    print("done", v, flush=True)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "results/period_shift.csv"), index=False)
s = out.groupby(["vehicle", "kind"]).period_ratio_median.agg(["count", "min", "median", "max"]).round(4)
print(s.to_string())
