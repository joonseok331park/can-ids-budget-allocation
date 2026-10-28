"""ROAD analyses added in revision (after the fixed-protocol results were known).

Same setting as road_v3.py (B = 120, u in {190, 0}, five conditions), seeds 0-9, and for the source-only conditions
also a donor profile of 60 s (= B/2) instead of 250 s. Extra columns per row:
  AP_fab, AP_masq                   AP over fabrication captures and over masquerade captures separately
  APg__<attack>                     AP per base attack (fabrication and masquerade variants kept apart)
  AP_held, pAUC01_held              attack-positive windows against all held-out normal windows
  held_pos_excl                     held-out alarm share without the two captures recorded to contain unusual
                                    benign behaviour (ambient_dyno_drive_benign_anomaly, ambient_dyno_exercise_all_bits)
  held_pos_dyno, held_pos_road      held-out alarm share over dynamometer and on-road captures
Output: results/runs/road_rev.csv; with --srcC, results/runs/road_rev_srcC.csv (threshold from CTT donor windows)
"""
from __future__ import annotations

import json
import os
import sys
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import road_v3 as R  # noqa: E402
from alloc_v3 import u_score  # noqa: E402
from models_v2 import ap  # noqa: E402

SPECIAL = {"ambient_dyno_drive_benign_anomaly", "ambient_dyno_exercise_all_bits"}


def bank(seed, L, n=213):
    rng = np.random.default_rng(zlib.crc32(f"road-bank|{seed}".encode()))
    per = [n // 4 + (1 if i < n % 4 else 0) for i in range(4)]
    xs = []
    for v, k in zip(R.DONORS, per):
        X = np.load(os.path.join(R.FV2, "source", f"{v}_P{L:g}.npz"))["X"]
        xs.append(X[rng.choice(len(X), k, replace=False)])
    return np.vstack(xs), dict(zip(R.DONORS, per))


def extras(u, cond, seed, L, d, m):
    F, C = R.segs(cond)
    if cond == "U_allC":
        sc = u_score
    else:
        X = R.budget_windows(u, *F) if F is not None else bank(seed, L)[0]
        clf = IsolationForest(n_estimators=200, max_samples=min(256, len(X)), random_state=seed).fit(X)
        sc = lambda Z: -clf.score_samples(Z)  # noqa: E731
    tau = float(sc(R.budget_windows(u, *C)).max())
    s_att, s_held = sc(d["Xatt"]), sc(d["Xheld"])
    lab = d["att_label"]
    files = np.array([m["attack_info"][i]["file"] for i in d["att_file"]])
    masq = np.char.find(files.astype(str), "_masquerade") >= 0
    out = {"AP_fab": ap(s_att[~masq], lab[~masq]), "AP_masq": ap(s_att[masq], lab[masq])}
    groups = np.array([m["attack_info"][i]["group"] for i in d["att_file"]])
    for g in sorted(set(groups)):
        for name, mk in (("fab", ~masq), ("masq", masq)):
            sel = (groups == g) & mk
            if sel.any():
                out[f"APg__{g}__{name}"] = ap(s_att[sel], lab[sel])
    pos = s_att[lab == 1]
    y = np.r_[np.ones(pos.size), np.zeros(s_held.size)]
    s = np.r_[pos, s_held]
    out.update(AP_held=ap(s, y), pAUC01_held=float(roc_auc_score(y, s, max_fpr=0.01)))
    names = np.array(m["held"])[d["held_cap"]]
    alarm = s_held > tau
    out["held_pos_excl"] = float(alarm[~np.isin(names, list(SPECIAL))].mean())
    dyno = np.char.find(names.astype(str), "_dyno_") >= 0
    out.update(held_pos_dyno=float(alarm[dyno].mean()), held_pos_road=float(alarm[~dyno].mean()))
    return out


def main():
    m = json.load(open(os.path.join(R.OUT, "meta.json")))
    rows = []
    for u in (190.0, 0.0):
        d = dict(np.load(os.path.join(R.OUT, f"u{u:g}.npz")))
        for cond in ("T_main", "S_sameC", "S_allC", "T_split", "U_allC"):
            for L in ((250.0, 60.0) if cond.startswith("S_") else (250.0,)):
                R.source_bank = lambda seed, n=213, L=L: bank(seed, L, n)  # used inside road_v3.run
                for s in range(10):
                    row = R.run(u, cond, s, d, m)
                    row["donor_len"] = L
                    row.update(extras(u, cond, s, L, d, m))
                    rows.append(row)
        print("u", u, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(R.ROOT, "results/runs/road_rev.csv"), index=False)
    print("rows", len(rows))




def source_threshold_rows():
    """S_srcC on ROAD: the CTT-trained forest with tau = max over the CTT donors' calibration windows (430 s to the
    end of each donor's budget capture, never in the bank). No ROAD data enter the threshold. -> road_rev_srcC.csv"""
    m = json.load(open(os.path.join(R.OUT, "meta.json")))
    rows = []
    for u in (190.0, 0.0):
        d = dict(np.load(os.path.join(R.OUT, f"u{u:g}.npz")))
        names = np.array(m["held"])[d["held_cap"]]
        for L in (250.0, 60.0):
            cal = np.vstack([np.load(os.path.join(R.FV2, "source_cal", f"{v}_P{L:g}.npz"))["X"] for v in R.DONORS])
            for s in range(10):
                X = bank(s, L)[0]
                clf = IsolationForest(n_estimators=200, max_samples=min(256, len(X)), random_state=s).fit(X)
                sc = lambda Z: -clf.score_samples(Z)  # noqa: E731
                tau = float(sc(cal).max())
                s_att, s_held, s_acc = sc(d["Xatt"]), sc(d["Xheld"]), sc(d["Xacc"])
                first = d["held_cap"] == 0
                dd = {k: d[k] for k in ("att_label", "att_ts", "att_te", "att_file")}
                dd.update(held_ts=d["held_ts"][first], held_te=d["held_te"][first])
                row = {"cond": "S_srcC", "u": u, "seed": s, "donor_len": L, "tau": tau, "AP": ap(s_att, d["att_label"])}
                row.update(R.evaluate_v3(dd, {"attack_info": m["attack_info"]}, s_att, s_held[first], tau))
                alarm = s_held > tau
                hrs = ev = 0.0
                for i, name in enumerate(m["held"]):
                    mk = d["held_cap"] == i
                    h, n, _, pos = R.normal_burden(d["held_ts"][mk], d["held_te"][mk], alarm[mk])
                    hrs += h; ev += n
                    row[f"held_pos__{name}"] = pos
                row.update(held_hours=hrs, held_events=int(ev), held_fa_per_h=ev / hrs, held_pos_frac=float(alarm.mean()),
                           held_pos_excl=float(alarm[~np.isin(names, list(SPECIAL))].mean()),
                           accel_pos_frac=float((s_acc > tau).mean()))
                rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(R.ROOT, "results/runs/road_rev_srcC.csv"), index=False)
    print("rows", len(rows))


if __name__ == "__main__":
    main() if "--srcC" not in sys.argv else source_threshold_rows()
