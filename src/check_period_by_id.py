"""Per-ID period ratio between each CTT capture and the same vehicle's budget (longest attack-free) capture.

Follow-up to check_period_shift.py, which used only the five most frequent IDs (all 10 ms messages). Here every
periodic ID (median IAT < 1 s in the budget capture, >= 100 frames) is compared, so a uniform clock-scale change
(all ratios equal) can be told apart from a period-dependent timing change.
Attack captures use frames before the first attack frame only and need >= 30 s of prefix.
Output: results/period_by_id.csv (long), results/period_by_id_summary.csv, results/period_by_id_short.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from canio import ensure_sorted, load_ctt_csv  # noqa: E402
from experiment import CTT, ROOT  # noqa: E402

BUCKETS = np.array([10, 12.5, 20, 25, 33.3, 40, 50, 100, 200, 250, 500, 1000])


def per_id_median_iat(t, aid, min_frames=20):
    """ID -> (median IAT, mean IAT, IQR of IAT)."""
    out = {}
    for a in np.unique(aid):
        ta = t[aid == a]
        if ta.size >= min_frames:
            x = np.diff(ta)
            q1, q3 = np.percentile(x, [25, 75])
            out[int(a)] = (float(np.median(x)), float(x.mean()), float(q3 - q1))
    return out


m = pd.read_csv(os.path.join(ROOT, "results/manifest_ctt.csv")).drop_duplicates("sha256")
rows = []
for v, g in m.groupby("vehicle"):
    af = g[g.kind == "attack-free"].sort_values("duration_s", ascending=False)
    base = load_ctt_csv(os.path.join(CTT, af.path.iloc[0]))
    bmed = {a: p for a, p in per_id_median_iat(base["t"], base["aid"], 100).items() if p[0] < 1.0}
    for _, r in g.iterrows():
        if r.path == af.path.iloc[0]:
            continue
        cap = ensure_sorted(load_ctt_csv(os.path.join(CTT, r.path)))
        att = np.flatnonzero(cap["label"] == 1)
        n = att[0] if att.size else cap["t"].size
        if n < 2 or cap["t"][n - 1] - cap["t"][0] < 30:
            continue
        cmed = per_id_median_iat(cap["t"][:n], cap["aid"][:n])
        for a, (p, pm, piqr) in bmed.items():
            if a in cmed:
                q, qm, qiqr = cmed[a]
                rows.append({"vehicle": v, "path": r.path, "kind": r.kind, "aid": a, "ref_period_ms": p * 1000,
                             "bucket_ms": float(BUCKETS[np.argmin(np.abs(BUCKETS - p * 1000))]),
                             "ratio": q / p, "mean_ratio": qm / pm, "iqr_ratio": qiqr / max(piqr, 1e-6),
                             "ref_iqr_us": piqr * 1e6, "cap_iqr_us": qiqr * 1e6, "cap_iqr_over_period": qiqr / p})
    print("done", v, flush=True)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(ROOT, "results/period_by_id.csv"), index=False)
d["cls"] = np.where(d.kind.isin(["attack-free", "accessory"]), d.kind, "attack")
s = d.groupby(["vehicle", "cls", "bucket_ms"]).agg(n=("ratio", "size"), median_ratio=("ratio", "median"),
                                                  min_ratio=("ratio", "min"), max_ratio=("ratio", "max"),
                                                  mean_iat_ratio=("mean_ratio", "median"),
                                                  iqr_ratio=("iqr_ratio", "median"),
                                                  ref_iqr_us=("ref_iqr_us", "median"), cap_iqr_us=("cap_iqr_us", "median"),
                                                  cap_iqr_over_period=("cap_iqr_over_period", "median")).round(4).reset_index()
s.to_csv(os.path.join(ROOT, "results/period_by_id_summary.csv"), index=False)
print(s.to_string(index=False))
# per capture: spread of ratios across period buckets
t = d[d.bucket_ms <= 25].groupby(["vehicle", "cls"]).agg(median_ratio=("ratio", "median"),
                                                        mean_iat_ratio=("mean_ratio", "median"),
                                                        iqr_ratio=("iqr_ratio", "median"),
                                                        ref_iqr_us=("ref_iqr_us", "median"),
                                                        cap_iqr_us=("cap_iqr_us", "median"),
                                                        cap_iqr_over_period=("cap_iqr_over_period", "median")).round(3)
t.to_csv(os.path.join(ROOT, "results/period_by_id_short.csv"))
print(t.to_string())
