"""ROAD external check with the fixed B = 120 protocol.

Stage 1 (--stage features): for u in {190, 0}, profile P = [u, u + 60) of ambient_dyno_drive_basic_long (0xFFF
removed); evaluation windows for the 11 other ambient captures (each kept separately), the 29 attack captures with
an injection interval (window positive if it overlaps the interval) and the 4 accelerator captures (reported apart).
Stage 2 (--stage models): T_main, S_sameC, S_allC, T_split, U_allC x u x seeds 0-2 -> results/runs/road_v3.csv
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(__file__))
from alloc_v3 import u_score  # noqa: E402
from canio import load_road_log  # noqa: E402
from eval_v3 import CURVE_T, evaluate_v3, normal_burden, virtual_starts  # noqa: E402
from features import build_profile, drop_ids, slice_time, window_features  # noqa: E402
from models_v2 import ap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(ROOT, "data/raw/road_zenodo/road")
OUT = os.path.join(ROOT, "data/features_road")
FV2 = os.path.join(ROOT, "data/features_v2")
BUDGET = "ambient_dyno_drive_basic_long"
B, P_FRAC = 120.0, 0.5
DONORS = ["forester", "impala", "silverado", "traverse"]


def load(sub, name):
    c = load_road_log(os.path.join(RD, sub, name + ".log"))
    c = drop_ids(c, [0xFFF])
    c["label"] = np.zeros(c["t"].size, np.int8)
    return c


def meta():
    amb = json.load(open(os.path.join(RD, "ambient", "capture_metadata.json")))
    att = json.load(open(os.path.join(RD, "attacks", "capture_metadata.json")))
    return amb, att


def stage_features():
    os.makedirs(OUT, exist_ok=True)
    amb, att = meta()
    bud = load("ambient", BUDGET)
    t0 = bud["t"][0]
    held = [k for k in sorted(amb) if k != BUDGET]
    caps_h = {k: load("ambient", k) for k in held}
    with_int = [k for k in sorted(att) if att[k].get("injection_interval")]
    accel = [k for k in sorted(att) if not att[k].get("injection_interval")]
    caps_a = {k: load("attacks", k) for k in with_int + accel}
    for u in (190.0, 0.0):
        out = os.path.join(OUT, f"u{u:g}.npz")
        if os.path.exists(out):
            continue
        prof = build_profile(slice_time(bud, t0 + u, t0 + u + P_FRAC * B))
        d = {"u": u}
        Xh, hts, hte, hid = [], [], [], []
        for i, k in enumerate(held):
            w = window_features(caps_h[k], prof, 1.0, 0.5)
            Xh.append(w["X"]); hts.append(w["t_start"]); hte.append(w["t_end"]); hid.append(np.full(len(w["X"]), i))
        d.update(Xheld=np.vstack(Xh), held_ts=np.concatenate(hts), held_te=np.concatenate(hte), held_cap=np.concatenate(hid))
        Xa, la, ats, ate, aid_ = [], [], [], [], []
        info = []
        for i, k in enumerate(with_int):
            c = caps_a[k]
            s, e = att[k]["injection_interval"]
            ct0 = c["t"][0]
            w = window_features(c, prof, 1.0, 0.5, attack_intervals=[(ct0 + s, ct0 + e)])
            Xa.append(w["X"]); la.append(w["label"]); ats.append(w["t_start"]); ate.append(w["t_end"])
            aid_.append(np.full(len(w["X"]), i))
            kind = k.replace("_masquerade", "").rstrip("_0123456789")
            info.append({"file": k, "kind": kind, "group": k.replace("_masquerade", ""),
                         "first_attack": float(ct0 + s), "last_attack": float(ct0 + e)})
        d.update(Xatt=np.vstack(Xa), att_label=np.concatenate(la), att_ts=np.concatenate(ats),
                 att_te=np.concatenate(ate), att_file=np.concatenate(aid_))
        Xacc, acc_id = [], []
        for i, k in enumerate(accel):
            w = window_features(caps_a[k], prof, 1.0, 0.5)
            Xacc.append(w["X"]); acc_id.append(np.full(len(w["X"]), i))
        d.update(Xacc=np.vstack(Xacc), acc_file=np.concatenate(acc_id))
        np.savez_compressed(out, **d)
        json.dump({"held": held, "attack_info": info, "accelerator": accel, "budget": BUDGET},
                  open(os.path.join(OUT, "meta.json"), "w"), indent=1)
        print("saved", out, flush=True)
    return bud


_BUD = {}


def budget_windows(u, a, z):
    if "cap" not in _BUD:
        _BUD["cap"] = load("ambient", BUDGET)
    bud = _BUD["cap"]
    t0 = bud["t"][0]
    key = ("prof", u)
    if key not in _BUD:
        _BUD[key] = build_profile(slice_time(bud, t0 + u, t0 + u + P_FRAC * B))
    A, Z = t0 + u + a, t0 + u + z
    return window_features(slice_time(bud, A, Z), _BUD[key], 1.0, 0.5, t_origin=A, t_stop=Z)["X"]


def source_bank(seed, n=213):
    rng = np.random.default_rng(zlib.crc32(f"road-bank|{seed}".encode()))
    per = [n // 4 + (1 if i < n % 4 else 0) for i in range(4)]
    xs = []
    for v, k in zip(DONORS, per):
        X = np.load(os.path.join(FV2, "source", f"{v}_P250.npz"))["X"]
        xs.append(X[rng.choice(len(X), k, replace=False)])
    return np.vstack(xs), dict(zip(DONORS, per))


def segs(cond):
    pB = P_FRAC * B
    R = B - pB
    return {"T_main": ((pB, pB + 0.6 * R), (pB + 0.6 * R, B)), "S_sameC": (None, (pB + 0.6 * R, B)),
            "S_allC": (None, (pB, B)), "T_split": ((pB, pB + 0.3 * R), (pB + 0.3 * R, B)),
            "U_allC": (None, (pB, B))}[cond]


def run(u, cond, seed, d, m):
    F, C = segs(cond)
    info = {"n_F": 0, "n_fit": 0, "max_samples_": np.nan, "donor_rows": ""}
    if cond == "U_allC":
        sc = u_score
    else:
        if F is not None:
            X = budget_windows(u, *F)
            info.update(n_F=len(X))
        else:
            X, counts = source_bank(seed)
            info["donor_rows"] = ";".join(f"{k}:{c}" for k, c in counts.items())
        clf = IsolationForest(n_estimators=200, max_samples=min(256, len(X)), random_state=seed).fit(X)
        info.update(n_fit=len(X), max_samples_=int(clf.max_samples_))
        sc = lambda Z: -clf.score_samples(Z)  # noqa: E731
    Xc = budget_windows(u, *C)
    s_cal = sc(Xc)
    tau = float(s_cal.max())
    s_att, s_held, s_acc = sc(d["Xatt"]), sc(d["Xheld"]), sc(d["Xacc"])
    meta_ = {"attack_info": m["attack_info"]}
    first = d["held_cap"] == 0
    dd = {k: d[k] for k in ("att_label", "att_ts", "att_te", "att_file")}
    dd.update(held_ts=d["held_ts"][first], held_te=d["held_te"][first])
    row = {"cond": cond, "u": u, "seed": seed, "n_C": len(Xc), "tau": tau, **info, "AP": ap(s_att, d["att_label"])}
    kinds = np.array([m["attack_info"][i]["kind"] for i in d["att_file"]])
    for k in sorted(set(kinds)):
        row[f"AP__{k}"] = ap(s_att[kinds == k], d["att_label"][kinds == k])
    row.update(evaluate_v3(dd, meta_, s_att, s_held[first], tau))
    # held-out normal aggregated over all 11 captures (each capture its own time axis)
    a_held = s_held > tau
    hrs = ev = 0.0
    vs = {t: [] for t in CURVE_T}
    for i, name in enumerate(m["held"]):
        mk = d["held_cap"] == i
        h, n, _, pos = normal_burden(d["held_ts"][mk], d["held_te"][mk], a_held[mk])
        hrs += h; ev += n
        row[f"held_pos__{name}"] = pos
        for t, val in virtual_starts(d["held_ts"][mk], d["held_te"][mk], a_held[mk]).items():
            if np.isfinite(val):
                ts_, te_ = d["held_ts"][mk], d["held_te"][mk]
                n_v = np.arange(ts_[0] + 5.0, te_[-1] - max(CURVE_T), 5.0).size  # same grid as virtual_starts
                vs[t].append((val, n_v))
    row.update(held_hours=hrs, held_events=int(ev), held_fa_per_h=ev / hrs if hrs else np.nan,
               held_pos_frac=float(a_held.mean()))
    for t in CURVE_T:
        w = sum(n for _, n in vs[t])
        row[f"held_virtual_new_{t}s"] = sum(v * n for v, n in vs[t]) / w if w else np.nan
    row["accel_pos_frac"] = float((s_acc > tau).mean())
    return row


def stage_models():
    m = json.load(open(os.path.join(OUT, "meta.json")))
    rows = []
    for u in (190.0, 0.0):
        d = dict(np.load(os.path.join(OUT, f"u{u:g}.npz")))
        for cond in ("T_main", "S_sameC", "S_allC", "T_split", "U_allC"):
            for s in (0, 1, 2):
                rows.append(run(u, cond, s, d, m))
        print("u", u, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results/runs/road_v3.csv"), index=False)
    print("rows", len(rows))


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", choices=["features", "models"], required=True)
    a = ap_.parse_args()
    stage_features() if a.stage == "features" else stage_models()
