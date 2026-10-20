"""Stage 1 for the allocation study: profile-dependent evaluation windows for (vehicle, offset o, profile length Lp).

Profile P = [t0 + o, t0 + o + Lp) of the vehicle's budget capture (t0 = first frame). Only the evaluation windows,
which depend on the profile but not on how the rest of the budget is split, are cached here:
  held-out attack-free capture (whole) and all attack captures.
Target training/calibration windows are cheap and are windowed per segment in stage 2 (alloc_models.py), so each
segment keeps its own frame context.
Output: data/features_alloc/<vehicle>/o<o>_P<Lp>.npz with Xheld, held_ts, held_te, Xatt, att_label, att_ts,
att_te, att_file; data/features_alloc/<vehicle>/meta.json (same attack_info as features_v2).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from canio import ensure_sorted, load_ctt_csv  # noqa: E402
from experiment import CTT, ROOT, load_roles  # noqa: E402
from features import build_profile, slice_time, window_features  # noqa: E402

OUT = os.path.join(ROOT, "data/features_alloc")
W, S = 1.0, 0.5


def job(args):
    v, configs = args  # configs: list of (o, Lp)
    r = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))[v]
    bcap = load_ctt_csv(os.path.join(CTT, r["budget"]))
    hcap = load_ctt_csv(os.path.join(CTT, r["reference"]))
    acaps = [ensure_sorted(load_ctt_csv(os.path.join(CTT, p))) for p in r["attacks"]]
    t0 = bcap["t"][0]
    os.makedirs(os.path.join(OUT, v), exist_ok=True)
    src_meta = os.path.join(ROOT, "data/features_v2", v, "meta.json")
    shutil.copy(src_meta, os.path.join(OUT, v, "meta.json"))
    for o, Lp in configs:
        out = os.path.join(OUT, v, f"o{o:g}_P{Lp:g}.npz")
        if os.path.exists(out):
            continue
        prof = build_profile(slice_time(bcap, t0 + o, t0 + o + Lp))
        d = {"o": o, "Lp": Lp, "profile_ids": prof.vocab.size}
        hw = window_features(hcap, prof, W, S)
        d["Xheld"], d["held_ts"], d["held_te"] = hw["X"], hw["t_start"], hw["t_end"]
        Xs, ls, ts, te, fid = [], [], [], [], []
        for i, c in enumerate(acaps):
            w = window_features(c, prof, W, S)
            Xs.append(w["X"]); ls.append(w["label"]); ts.append(w["t_start"]); te.append(w["t_end"])
            fid.append(np.full(len(w["label"]), i))
        d["Xatt"] = np.vstack(Xs); d["att_label"] = np.concatenate(ls)
        d["att_ts"] = np.concatenate(ts); d["att_te"] = np.concatenate(te); d["att_file"] = np.concatenate(fid)
        np.savez_compressed(out, **d)
        print(f"[{time.strftime('%H:%M:%S')}] {v} o={o:g} P={Lp:g} saved", flush=True)
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="forester,impala,silverado,traverse")
    ap.add_argument("--offsets", default="0")
    ap.add_argument("--plens", default="15,30,60")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = [(float(o), float(p)) for o in a.offsets.split(",") for p in a.plens.split(",")]
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(job, [(v, cfg) for v in a.targets.split(",")]))
    print("all done", flush=True)
