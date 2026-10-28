"""Stage 1 (v2): same-capture nested profiles and a fully held-out normal capture.

For target vehicle v (its longer attack-free capture = budget capture, origin t0 = first frame):
  reference P  [t0,           t0 + 250)
  short P      [t0 + 250 - B/2, t0 + 250)       (subset of the reference P)
  F (IF train) [t0 + 250,     t0 + 250 + 0.3B)
  C (calib)    [t0 + 250 + 0.3B, t0 + 250 + 0.3B + 60)   stored up to 60 s; the main analysis uses the first 0.2B s
Evaluation (identical for every condition):
  held-out normal: the other attack-free capture, whole
  attack captures: all windows (AP), per-capture detection, clean prefix windows (end <= first attack frame)
Source vehicle s: P = [t0_s + 250 - L, t0_s + 250) with L = 250 (main) or matched lengths; source windows from
  [t0_s + 250, t0_s + 250 + 180), i.e. never from the source's own profile segment.
Outputs: data/features_v2/<vehicle>/B<B>_<short|ref>.npz, data/features_v2/source/<vehicle>_P<L>.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from canio import ensure_sorted, load_ctt_csv  # noqa: E402
from experiment import CTT, ROOT, load_roles  # noqa: E402
from features import build_profile, slice_time, window_features  # noqa: E402

OUT = os.path.join(ROOT, "data/features_v2")
ANCHOR = 250.0
C_MAX = 60.0
SRC_SPAN = 180.0
W, S = 1.0, 0.5


def seg_windows(cap, prof, a, b):
    return window_features(slice_time(cap, a, b), prof, W, S, t_origin=a, t_stop=b)


def target_job(args):
    v, budgets = args
    roles = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))
    r = roles[v]
    bcap = load_ctt_csv(os.path.join(CTT, r["budget"]))
    hcap = load_ctt_csv(os.path.join(CTT, r["reference"]))
    acaps = [(p, k, ensure_sorted(load_ctt_csv(os.path.join(CTT, p)))) for p, k in zip(r["attacks"], r["attack_kinds"])]
    t0 = bcap["t"][0]
    os.makedirs(os.path.join(OUT, v), exist_ok=True)
    meta = {"budget_capture": r["budget"], "heldout_capture": r["reference"], "attacks": r["attacks"],
            "kinds": r["attack_kinds"], "anchor_s": ANCHOR, "c_max_s": C_MAX}
    att_info = []
    for i, (p, k, c) in enumerate(acaps):
        idx = np.flatnonzero(c["label"] == 1)
        att_info.append({"file": p, "kind": k, "t_first_frame": float(c["t"][0]),
                         "first_attack": float(c["t"][idx[0]]), "last_attack": float(c["t"][idx[-1]]),
                         "n_attack_frames": int(idx.size)})
    meta["attack_info"] = att_info
    json.dump(meta, open(os.path.join(OUT, v, "meta.json"), "w", encoding="utf-8"), indent=1)
    for B in budgets:
        F = (t0 + ANCHOR, t0 + ANCHOR + 0.3 * B)
        Cx = (F[1], F[1] + C_MAX)
        for pcond, P in (("short", (t0 + ANCHOR - B / 2, t0 + ANCHOR)), ("ref", (t0, t0 + ANCHOR))):
            out = os.path.join(OUT, v, f"B{B}_{pcond}.npz")
            if os.path.exists(out):
                continue
            prof = build_profile(slice_time(bcap, *P))
            d = {"P": np.array(P) - t0, "F": np.array(F) - t0, "C": np.array(Cx) - t0, "profile_ids": prof.vocab.size}
            fw = seg_windows(bcap, prof, *F)
            cw = seg_windows(bcap, prof, *Cx)
            d["Xtr"], d["Xcal"], d["cal_ts"] = fw["X"], cw["X"], cw["t_start"] - Cx[0]
            hw = window_features(hcap, prof, W, S)
            d["Xheld"], d["held_ts"], d["held_te"] = hw["X"], hw["t_start"], hw["t_end"]
            Xs, ls, ts, te, fid = [], [], [], [], []
            for i, (p, k, c) in enumerate(acaps):
                w = window_features(c, prof, W, S)
                Xs.append(w["X"]); ls.append(w["label"]); ts.append(w["t_start"]); te.append(w["t_end"])
                fid.append(np.full(len(w["label"]), i))
            d["Xatt"] = np.vstack(Xs); d["att_label"] = np.concatenate(ls)
            d["att_ts"] = np.concatenate(ts); d["att_te"] = np.concatenate(te); d["att_file"] = np.concatenate(fid)
            np.savez_compressed(out, **d)
            print(f"[{time.strftime('%H:%M:%S')}] {v} B={B} {pcond} saved", flush=True)
    return v


def source_job(args):
    v, lengths = args
    roles = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))
    cap = load_ctt_csv(os.path.join(CTT, roles[v]["budget"]))
    t0 = cap["t"][0]
    os.makedirs(os.path.join(OUT, "source"), exist_ok=True)
    for L in lengths:
        out = os.path.join(OUT, "source", f"{v}_P{L:g}.npz")
        if os.path.exists(out):
            continue
        prof = build_profile(slice_time(cap, t0 + ANCHOR - L, t0 + ANCHOR))
        w = seg_windows(cap, prof, t0 + ANCHOR, t0 + ANCHOR + SRC_SPAN)
        np.savez_compressed(out, X=w["X"])
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="silverado,forester")
    ap.add_argument("--budgets", default="120,30")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]
    targets = a.targets.split(",")
    roles = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))
    lengths = sorted({250.0} | {B / 2 for B in budgets})
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(source_job, [(v, lengths) for v in sorted(roles)]))
        print("sources done", flush=True)
        list(ex.map(target_job, [(v, budgets) for v in targets]))
    print("all done", flush=True)
