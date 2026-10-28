"""Stage 1: compute and cache window features for every (target, budget, offset, profile condition).

Model fitting, thresholds and metrics live in stage2 so they can be changed without recomputing features.
Cache layout: data/features/<target>/B<budget>_o<offset>_<profile>.npz and data/features/source/<vehicle>_p<len>.npz
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
from experiment import CTT, ROOT, load_roles, seg_split  # noqa: E402
from features import build_profile, slice_time, window_features  # noqa: E402
from manifest import attack_segments  # noqa: E402

FEAT = os.path.join(ROOT, "data/features")


def offsets_for(dur, B, cfg):
    step = max(B, cfg["offset_step_min_s"])
    return [float(o) for o in np.arange(0, dur - B, step)][: cfg["n_offsets"]]


def target_job(args):
    tv, cfg = args
    roles = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))
    r = roles[tv]
    W, S, fr = cfg["width"], cfg["stride"], cfg["split"]
    bcap = load_ctt_csv(os.path.join(CTT, r["budget"]))
    rcap = load_ctt_csv(os.path.join(CTT, r["reference"])) if r["reference"] else None
    acaps = [(p, k, ensure_sorted(load_ctt_csv(os.path.join(CTT, p)))) for p, k in zip(r["attacks"], r["attack_kinds"])]
    segs = [attack_segments(c["t"], c["label"]) for _, _, c in acaps]
    bdur = bcap["t"][-1] - bcap["t"][0]
    os.makedirs(os.path.join(FEAT, tv), exist_ok=True)
    for B in cfg["budgets"]:
        for o in offsets_for(bdur, B, cfg):
            t0 = bcap["t"][0] + o
            (pa, pb), (ta, tb), (ca, cb) = seg_split(t0, B, fr)
            for pcond in cfg["profile_conditions"]:
                out = os.path.join(FEAT, tv, f"B{B}_o{int(o)}_{pcond}.npz")
                if os.path.exists(out):
                    continue
                if pcond == "budget":
                    prof = build_profile(slice_time(bcap, pa, pb))
                else:
                    if rcap is None:
                        continue
                    prof = build_profile(slice_time(rcap, rcap["t"][0], rcap["t"][0] + cfg["reference_profile_s"]))
                d = {}
                tr = window_features(slice_time(bcap, ta, tb), prof, W, S)
                ca_ = window_features(slice_time(bcap, ca, cb), prof, W, S)
                d["Xtr"], d["Xcal"] = tr["X"], ca_["X"]
                if rcap is not None and pcond == "budget":
                    hw = window_features(rcap, prof, W, S)
                    d["Xheld"], d["held_ts"], d["held_te"] = hw["X"], hw["t_start"], hw["t_end"]
                Xs, ls, tss, tes, fid, = [], [], [], [], []
                seg_rows = []
                for i, ((p, k, c), sg) in enumerate(zip(acaps, segs)):
                    w = window_features(c, prof, W, S)
                    Xs.append(w["X"]); ls.append(w["label"]); tss.append(w["t_start"]); tes.append(w["t_end"])
                    fid.append(np.full(len(w["label"]), i))
                    for a, b in sg:
                        seg_rows.append((i, a, b))
                d["Xatt"] = np.vstack(Xs); d["att_label"] = np.concatenate(ls)
                d["att_ts"] = np.concatenate(tss); d["att_te"] = np.concatenate(tes); d["att_file"] = np.concatenate(fid)
                d["segments"] = np.array(seg_rows, dtype=float).reshape(-1, 3)
                d["profile_ids"] = np.int64(prof.vocab.size)
                np.savez_compressed(out, **d)
            print(f"[{time.strftime('%H:%M:%S')}] {tv} B={B} o={o} features saved", flush=True)
    json.dump({"attacks": r["attacks"], "kinds": r["attack_kinds"], "budget": r["budget"], "reference": r["reference"]},
              open(os.path.join(FEAT, tv, "files.json"), "w", encoding="utf-8"), indent=1)
    return tv


def source_job(args):
    sv, cfg = args
    roles = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))
    cap = load_ctt_csv(os.path.join(CTT, roles[sv]["budget"]))
    os.makedirs(os.path.join(FEAT, "source"), exist_ok=True)
    lens = sorted({round(B * cfg["split"][0], 3) for B in cfg["budgets"]} | {float(cfg["source_profile_long_s"])})
    for plen in lens:
        out = os.path.join(FEAT, "source", f"{sv}_p{plen:g}.npz")
        if os.path.exists(out):
            continue
        t0 = cap["t"][0] + cfg["source_offset_s"]
        prof = build_profile(slice_time(cap, t0, t0 + plen))
        seg = slice_time(cap, t0 + plen, t0 + plen + cfg["source_window_span_s"])
        np.savez_compressed(out, X=window_features(seg, prof, cfg["width"], cfg["stride"])["X"])
    return sv


if __name__ == "__main__":
    from experiment import DEFAULT
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = dict(DEFAULT)
    if a.config:
        cfg.update(json.load(open(a.config)))
    roles = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))
    targets = cfg["targets"] or sorted(roles)
    os.makedirs(FEAT, exist_ok=True)
    json.dump(cfg, open(os.path.join(FEAT, "stage1_config.json"), "w", encoding="utf-8"), indent=1)
    with ProcessPoolExecutor(a.workers) as ex:
        for v in ex.map(source_job, [(v, cfg) for v in sorted(roles)]):
            print("source done", v, flush=True)
        for v in ex.map(target_job, [(v, cfg) for v in targets]):
            print("target done", v, flush=True)
