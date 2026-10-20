"""Order-robust reuse check: per-ID payload sequences.

check_reuse.py hashes 32 consecutive frames across all IDs, so a replay that keeps every payload but changes the
interleaving of IDs (timing jitter) would not match. Here, for each vehicle, take the IDs of the budget
capture with the least repetitive payload sequences (top 5 by the share of 16-grams that occur once, among IDs
with >= 1000 frames) and
hash k = 16 consecutive payloads of each such ID separately. A capture built from a recorded drive keeps those
per-ID sequences even if its frame order or timestamps change.
Attack captures use frames before the first attack frame.
Output: results/reuse_perid.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from canio import ensure_sorted, load_ctt_csv  # noqa: E402
from experiment import CTT, ROOT, load_roles  # noqa: E402

K = 16
BASE = np.uint64(1099511628211)


def payload_hash(data):
    h = np.zeros(len(data), dtype=np.uint64)
    for j in range(8):
        h = h * np.uint64(257) + data[:, j].astype(np.uint64)
    return h


def kgrams(h, k=K):
    if h.size < k:
        return np.zeros(0, dtype=np.uint64)
    out = np.zeros(h.size - k + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(k):
            out = out * BASE + h[j:j + out.size]
    return out


def id_grams(cap, ids, n=None):
    t = {}
    aid = cap["aid"][:n] if n is not None else cap["aid"]
    data = cap["data"][:n] if n is not None else cap["data"]
    for a in ids:
        t[a] = kgrams(payload_hash(data[aid == a]))
    return t


def frac(q, ref):
    """Share of the query's informative grams found in the reference. Grams repeated within the reference
    (static payload stretches) are excluded from numerator and denominator, so idle periods do not match by chance."""
    num = den = 0
    for a in q:
        u, c = np.unique(ref[a], return_counts=True)
        rep, single = u[c > 1], u[c == 1]
        qi = q[a][~np.isin(q[a], rep)]
        num += np.isin(qi, single).sum(); den += qi.size
    return num / den if den else np.nan


def main():
    roles = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))
    rows = []
    for v in sorted(roles):
        r = roles[v]
        b = load_ctt_csv(os.path.join(CTT, r["budget"]))
        ids, cnt = np.unique(b["aid"], return_counts=True)
        cand = ids[cnt >= 1000]
        uniq = []  # share of an ID's 16-grams that occur exactly once in the budget capture
        for a in cand:
            g = kgrams(payload_hash(b["data"][b["aid"] == a]))
            _, c = np.unique(g, return_counts=True)
            uniq.append((c == 1).sum() / max(g.size, 1))
        top = cand[np.argsort(-np.array(uniq))[:5]]
        t0 = b["t"][0]
        n400 = int(np.searchsorted(b["t"], t0 + 400))
        g_budget = id_grams(b, top)
        g_train = id_grams(b, top, n400)
        h = load_ctt_csv(os.path.join(CTT, r["reference"]))
        g_held = id_grams(h, top)
        n_half = int(np.searchsorted(b["t"], t0 + 200))
        rows.append({"vehicle": v, "file": r["budget"] + " [first 200 s]", "role": "positive_control",
                     "ids": " ".join(hex(a) for a in top), "frac_in_train400": frac(id_grams(b, top, n_half), g_train),
                     "frac_in_budget": frac(id_grams(b, top, n_half), g_budget)})
        rows.append({"vehicle": v, "file": r["reference"], "role": "heldout_attack_free", "ids": " ".join(hex(a) for a in top),
                     "frac_in_train400": frac(g_held, g_train), "frac_in_budget": frac(g_held, g_budget)})
        for p, kind in zip(r["attacks"], r["attack_kinds"]):
            c = ensure_sorted(load_ctt_csv(os.path.join(CTT, p)))
            att = np.flatnonzero(c["label"] == 1)
            n = att[0] if att.size else c["t"].size
            g = id_grams(c, top, n)
            rows.append({"vehicle": v, "file": p, "role": f"attack:{kind}", "prefix_s": float(c["t"][n - 1] - c["t"][0]) if n else 0.0,
                         "frac_in_train400": frac(g, g_train), "frac_in_budget": frac(g, g_budget),
                         "frac_in_heldout": frac(g, g_held)})
        print("done", v, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(ROOT, "results/reuse_perid.csv"), index=False)
    print(d.groupby("vehicle")[["frac_in_train400", "frac_in_budget", "frac_in_heldout"]].agg(["mean", "max"]).round(3).to_string())
    print(d[(d[["frac_in_train400", "frac_in_budget", "frac_in_heldout"]].fillna(0) > 0.01).any(axis=1)][
        ["vehicle", "file", "frac_in_train400", "frac_in_budget", "frac_in_heldout"]].round(3).to_string())


if __name__ == "__main__":
    main()
