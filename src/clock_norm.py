"""Exploratory control (not part of the fixed protocol): label-free capture-level clock normalization of evaluation captures.

Motivation: results/period_shift.csv shows that in most CTT attack captures the five most frequent periodic IDs run
up to about 4 % faster or slower than in the same vehicle's attack-free captures, with the same ratio across IDs.
That looks like a recording-clock difference, which a timing residual reads as an anomaly before any attack starts.

For each evaluation capture (held-out attack-free capture and every attack capture) and each profile condition:
  ratio = median over the K most frequent periodic profile IDs of (median IAT of that ID in the whole capture /
          profile median IAT of that ID)
  t' = t0 + (t - t0) / ratio
The estimate uses no labels; it is transductive (it sees the whole capture, attack frames included). Injected
frames can shorten one ID's median IAT, so the median over K IDs is used. Training and calibration windows come
from the profile's own capture and are not rescaled.

Output: data/features_v2/<v>/B<B>_<pcond>_clk.npz with the evaluation arrays of the main file plus
  att_first (rescaled first attack time per attack file), held_ratio, att_ratio
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from canio import ensure_sorted, load_ctt_csv  # noqa: E402
from experiment import CTT, ROOT, load_roles  # noqa: E402
from features import build_profile, slice_time, window_features  # noqa: E402
from nested_features import ANCHOR, OUT, S, W  # noqa: E402

K = 10


def clock_ratio(cap, prof, k=K):
    periodic = np.flatnonzero((prof.iat_med > 0) & (prof.iat_med < 0.2))
    top = periodic[np.argsort(-prof.rate[periodic])[:k]]
    ratios = []
    for j in top:
        tj = cap["t"][cap["aid"] == prof.vocab[j]]
        if tj.size > 20:
            ratios.append(np.median(np.diff(tj)) / prof.iat_med[j])
    return float(np.median(ratios)) if ratios else 1.0


def rescale(cap, ratio):
    t0 = cap["t"][0]
    out = dict(cap)
    out["t"] = t0 + (cap["t"] - t0) / ratio
    return out


def job(args):
    v, budgets = args
    r = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))[v]
    bcap = load_ctt_csv(os.path.join(CTT, r["budget"]))
    hcap = load_ctt_csv(os.path.join(CTT, r["reference"]))
    acaps = [ensure_sorted(load_ctt_csv(os.path.join(CTT, p))) for p in r["attacks"]]
    t0 = bcap["t"][0]
    for B in budgets:
        for pcond, P in (("short", (t0 + ANCHOR - B / 2, t0 + ANCHOR)), ("ref", (t0, t0 + ANCHOR))):
            out = os.path.join(OUT, v, f"B{B}_{pcond}_clk.npz")
            if os.path.exists(out):
                continue
            prof = build_profile(slice_time(bcap, *P))
            d = {}
            hr = clock_ratio(hcap, prof)
            hw = window_features(rescale(hcap, hr), prof, W, S)
            d["Xheld"], d["held_ts"], d["held_te"], d["held_ratio"] = hw["X"], hw["t_start"], hw["t_end"], hr
            Xs, ls, ts, te, fid, first, rat = [], [], [], [], [], [], []
            for i, c in enumerate(acaps):
                q = clock_ratio(c, prof)
                c2 = rescale(c, q)
                w = window_features(c2, prof, W, S)
                Xs.append(w["X"]); ls.append(w["label"]); ts.append(w["t_start"]); te.append(w["t_end"])
                fid.append(np.full(len(w["label"]), i))
                first.append(float(c2["t"][np.flatnonzero(c2["label"] == 1)[0]]))
                rat.append(q)
            d["Xatt"] = np.vstack(Xs); d["att_label"] = np.concatenate(ls)
            d["att_ts"] = np.concatenate(ts); d["att_te"] = np.concatenate(te); d["att_file"] = np.concatenate(fid)
            d["att_first"] = np.array(first); d["att_ratio"] = np.array(rat)
            np.savez_compressed(out, **d)
            print(f"[{time.strftime('%H:%M:%S')}] {v} B={B} {pcond} clk saved (held ratio {hr:.4f}, "
                  f"attack ratios {np.min(rat):.4f}-{np.max(rat):.4f})", flush=True)
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="forester,impala,silverado,traverse")
    ap.add_argument("--budgets", default="120,30")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(job, [(v, budgets) for v in a.targets.split(",")]))
    print("all done", flush=True)
