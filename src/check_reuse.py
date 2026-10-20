"""Data check: is the same normal driving reused across CTT files?

A k-gram is k consecutive frames' (ID, DLC, payload) ignoring timestamps. If an evaluation capture shares many
k-grams with the training region of the budget capture, its "normal" traffic is (partly) the same recording,
so it cannot count as an independent normal evaluation. k = 32 consecutive identical frames are very unlikely
to repeat by chance because payload counters and signals change.
Output: results/reuse_check.csv (one row per vehicle x evaluation file)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from canio import ensure_sorted, load_ctt_csv  # noqa: E402
from experiment import CTT, ROOT, load_roles  # noqa: E402

K = 32
MASK = (1 << 64) - 1
BASE = 1099511628211  # odd -> invertible mod 2^64


def frame_hash(cap) -> np.ndarray:
    d64 = cap["data"].copy().view(">u8").ravel().astype(np.uint64)
    meta = (cap["aid"].astype(np.uint64) << np.uint64(8)) | cap["dlc"].astype(np.uint64)
    with np.errstate(over="ignore"):
        return (d64 * np.uint64(0x9E3779B97F4A7C15)) ^ (meta * np.uint64(0xC2B2AE3D27D4EB4F))


def kgrams(fh: np.ndarray, k: int = K) -> np.ndarray:
    n = fh.size
    if n < k:
        return np.zeros(0, np.uint64)
    pre = np.empty(n + 1, dtype=object)
    s = 0
    pre[0] = 0
    vals = fh.tolist()
    for i, v in enumerate(vals):
        s = (s * BASE + v) & MASK
        pre[i + 1] = s
    bk = pow(BASE, k, 1 << 64)
    out = [(pre[i + k] - pre[i] * bk) & MASK for i in range(n - k + 1)]
    return np.array(out, dtype=np.uint64)


def match_frac(q: np.ndarray, ref: np.ndarray) -> float:
    return float(np.isin(q, ref).mean()) if q.size else float("nan")


def main():
    roles = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))
    rows = []
    for v in sorted(roles):
        r = roles[v]
        b = load_ctt_csv(os.path.join(CTT, r["budget"]))
        t0 = b["t"][0]
        train = {k: x[b["t"] < t0 + 400] if isinstance(x, np.ndarray) and x.shape[:1] == b["t"].shape else x for k, x in b.items()}
        kg_train = np.unique(kgrams(frame_hash(train)))
        kg_budget = np.unique(kgrams(frame_hash(b)))
        h = load_ctt_csv(os.path.join(CTT, r["reference"]))
        kg_held = kgrams(frame_hash(h))
        rows.append({"vehicle": v, "file": r["reference"], "role": "heldout_attack_free",
                     "frac_in_train400": match_frac(kg_held, kg_train), "frac_in_budget_capture": match_frac(kg_held, kg_budget)})
        kg_held_u = np.unique(kg_held)
        for p, kind in zip(r["attacks"], r["attack_kinds"]):
            c = ensure_sorted(load_ctt_csv(os.path.join(CTT, p)))
            att = np.flatnonzero(c["label"] == 1)
            first = att[0] if att.size else c["t"].size
            pre = {k: x[:first] if isinstance(x, np.ndarray) and x.shape[:1] == c["t"].shape else x for k, x in c.items()}
            kg_pre = kgrams(frame_hash(pre))
            normal_only = {k: x[c["label"] == 0] if isinstance(x, np.ndarray) and x.shape[:1] == c["t"].shape else x for k, x in c.items()}
            kg_norm = kgrams(frame_hash(normal_only))
            rows.append({"vehicle": v, "file": p, "role": f"attack:{kind}",
                         "prefix_s": float(c["t"][first - 1] - c["t"][0]) if first > 0 else 0.0,
                         "frac_in_train400": match_frac(kg_pre, kg_train),
                         "frac_in_budget_capture": match_frac(kg_pre, kg_budget),
                         "frac_in_heldout": match_frac(kg_pre, kg_held_u),
                         "normal_frames_frac_in_budget_capture": match_frac(kg_norm, kg_budget),
                         "normal_frames_frac_in_heldout": match_frac(kg_norm, kg_held_u)})
        print("done", v, flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results/reuse_check.csv"), index=False)


if __name__ == "__main__":
    main()
