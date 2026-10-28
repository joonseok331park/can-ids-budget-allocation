"""Analyses added after the first version of the paper. All of them were run after the fixed-protocol results were known.

--what seeds10   main grid (B in {30, 120, 300}, u in {250 - B/2, 0}, five conditions) with seeds 0-9
--what matched   S_sameC, S_allC, S_small with donor profiles of length B/2 (same budgets, positions, seeds 0-9)
--what decomp    T_splitF_mainC: the T_split forest with the T_main calibration segment (seeds 0-9)
--what featsets  B = 120: IF on the 13 features of U (T_main_u13, S_allC_u13) and the four IF conditions without
                 the five timing features (*_notime), seeds 0-9
--what oem       B = 120, GM targets: 71 donor windows from the Forester only (S_fo71) or from the two other GM
                 vehicles (S_gm71), next to S_small (71 from all three donors); psi 71, T_main's C; seeds 0-9
--what srccal    all budgets: source forest with tau = max over donor normal windows outside the bank (S_srcC), with
                 donor profiles of 250 s and of B/2; seeds 0-9
--what sweep     B = 120, u every 30 s over each budget capture, five conditions, seeds 0-2
--what sweepsrc  the same positions for S_srcC
Every row adds to the alloc_v3 columns:
  AP_held, AUC_held, pAUC01_held   attack-positive windows against held-out normal windows (pAUC: FPR <= 0.01,
                                   McClish-standardised as in scikit-learn)
  orc{1,5}_*                       oracle threshold giving a held-out alarm share of 1 % or 5 % (diagnostic only)
Scores of the B = 120 runs at the two fixed positions are kept in data/scores_rev/ for the bootstrap.
Output: results/runs/rev_<what>.csv
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import alloc_v3 as A  # noqa: E402
from alloc_models import FV2, eval_data, seg_windows  # noqa: E402
from eval_v3 import evaluate_v3  # noqa: E402
from features import FEATURES  # noqa: E402
from models_v2 import ap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES = os.path.join(ROOT, "data/scores_rev")
VEH = ["forester", "impala", "silverado", "traverse"]
GM = ["impala", "silverado", "traverse"]
MAIN = ["T_main", "S_sameC", "S_allC", "T_split", "U_allC"]
TIMING = ["iat_z_mean", "iat_z_gt3", "iat_short_frac", "id_max_iat_z", "id_max_short"]
FSETS = {"all": np.arange(len(FEATURES)), "u13": A.U_IDX,
         "notime": np.array([i for i, f in enumerate(FEATURES) if f not in TIMING])}
LENGTHS = {"capture": {"forester": 572.0, "impala": 1350.2, "silverado": 666.6, "traverse": 444.6}}


@functools.lru_cache(maxsize=64)
def pool(v, L):
    return {u: np.load(os.path.join(FV2, "source", f"{u}_P{L:g}.npz"))["X"] for u in VEH if u != v}


@functools.lru_cache(maxsize=64)
def bank(v, seed, L):
    """Same draw as alloc_v3.bank (the pools of every profile length have the same rows), so banks are paired."""
    rng = np.random.default_rng(zlib.crc32(f"bank|{v}|{seed}".encode()))
    p = pool(v, L)
    return {u: p[u][rng.choice(len(p[u]), A.PER_DONOR, replace=False)] for u in sorted(p)}


def rows_from(v, seed, L, donors, n):
    b = bank(v, seed, L)
    donors = sorted(donors)
    per = [n // len(donors) + (1 if i < n % len(donors) else 0) for i in range(len(donors))]
    return np.vstack([b[u][:k] for u, k in zip(donors, per)]), ";".join(f"{u}:{k}" for u, k in zip(donors, per))


@functools.lru_cache(maxsize=16)
def source_cal(v, L):
    return np.vstack([np.load(os.path.join(FV2, "source_cal", f"{u}_P{L:g}.npz"))["X"] for u in VEH if u != v])


def base(cond):
    """Condition name without feature-set suffix."""
    for s in ("_u13", "_notime"):
        if cond.endswith(s):
            return cond[: -len(s)], s[1:]
    return cond, "all"


def segments(B, cond, p=0.5):
    pB, R = p * B, (1 - p) * B
    main_C, split_F = (pB + 0.6 * R, B), (pB, pB + 0.3 * R)
    if cond in ("T_main",):
        return (pB, pB + 0.6 * R), main_C
    if cond in ("S_sameC", "S_small", "S_fo71", "S_gm71"):
        return None, main_C
    if cond == "T_split":
        return split_F, (pB + 0.3 * R, B)
    if cond == "T_splitF_mainC":
        return split_F, main_C
    if cond in ("S_allC", "U_allC"):
        return None, (pB, B)
    if cond == "S_srcC":
        return None, None
    raise ValueError(cond)


def run(v, u, B, cond, seed, donor_len=250.0, keep_scores=None):
    c0, fs = base(cond)
    idx = FSETS[fs]
    Lp = 0.5 * B
    F, C = segments(B, c0)
    info = {"n_F": 0, "n_fit": 0, "max_samples_": np.nan, "donor_rows": ""}
    if c0 == "U_allC":
        sc = A.u_score
    else:
        if c0.startswith("T_"):
            X = seg_windows(v, u, Lp, *F)
            psi = 256
            info["n_F"] = len(X)
        else:
            n, psi, donors = {"S_small": (71, 71, [x for x in VEH if x != v]),
                              "S_fo71": (71, 71, ["forester"]),
                              "S_gm71": (71, 71, [x for x in GM if x != v])}.get(
                c0, (213, 256, [x for x in VEH if x != v]))
            X, info["donor_rows"] = rows_from(v, seed, donor_len, donors, n)
        clf = IsolationForest(n_estimators=200, max_samples=min(psi, len(X)), random_state=seed).fit(X[:, idx])
        info.update(n_fit=len(X), max_samples_=int(clf.max_samples_))
        sc = lambda Z: -clf.score_samples(Z[:, idx])  # noqa: E731
    d, meta = eval_data(v, u, Lp)
    s_att, s_held = sc(d["Xatt"]), sc(d["Xheld"])
    if c0 == "S_srcC":
        n_C, tau = 0, float(sc(source_cal(v, donor_len)).max())
    else:
        Xc = seg_windows(v, u, Lp, *C)
        n_C, tau = len(Xc), float(sc(Xc).max())
    row = {"cond": cond, "target": v, "u": u, "B": B, "seed": seed, "donor_len": donor_len, "n_C": n_C, "tau": tau,
           **info, "AP": ap(s_att, d["att_label"])}
    kinds = np.array([meta["attack_info"][i]["kind"] for i in d["att_file"]])
    for k in sorted(set(kinds)):
        row[f"AP__{k}"] = ap(s_att[kinds == k], d["att_label"][kinds == k])
    no_prefix = frozenset(d["_no_prefix"])
    row.update(evaluate_v3(d, meta, s_att, s_held, tau, no_prefix=no_prefix))
    row.update(held_metrics(d, meta, s_att, s_held, no_prefix))
    if keep_scores:
        os.makedirs(SCORES, exist_ok=True)
        np.savez_compressed(os.path.join(SCORES, f"{keep_scores}__{v}__u{u:g}__{cond}__L{donor_len:g}__s{seed}.npz"),
                            s_att=s_att.astype(np.float32), s_held=s_held.astype(np.float32), tau=tau)
    return row


def held_metrics(d, meta, s_att, s_held, no_prefix):
    pos = s_att[d["att_label"] == 1]
    y = np.r_[np.ones(pos.size), np.zeros(s_held.size)]
    s = np.r_[pos, s_held]
    out = {"AP_held": ap(s, y), "AUC_held": float(roc_auc_score(y, s)),
           "pAUC01_held": float(roc_auc_score(y, s, max_fpr=0.01))}
    for a in (0.01, 0.05):
        tau = float(np.quantile(s_held, 1 - a))
        e = evaluate_v3(d, meta, s_att, s_held, tau, no_prefix=no_prefix)
        k = f"orc{int(a * 100)}_"
        out.update({k + "held_pos_frac": e["held_pos_frac"], k + "new_within_5s": e["new_within_5s"],
                    k + "held_virtual_new_5s": e["held_virtual_new_5s"],
                    k + "prefix_virtual_new_5s": e["prefix_virtual_new_5s"],
                    k + "att_pos_window_tpr": e["att_pos_window_tpr"]})
    return out


def positions(B):
    return [250.0 - B / 2, 0.0]


def sweep_positions(v, B=120.0):
    return [float(u) for u in range(0, int(LENGTHS["capture"][v] - B) + 1, 30)]


def job(args):
    what, v, seeds = args
    rows = []
    budgets = (30.0, 120.0, 300.0)
    keep = lambda B, u: what if (B == 120.0 and u in positions(120.0)) else None  # noqa: E731
    if what == "seeds10":
        for B in budgets:
            for u in positions(B):
                for c in MAIN:
                    for s in seeds:
                        rows.append(run(v, u, B, c, s, keep_scores=keep(B, u)))
    elif what == "matched":
        for B in budgets:
            for u in positions(B):
                for c in ("S_sameC", "S_allC", "S_small"):
                    for s in seeds:
                        rows.append(run(v, u, B, c, s, donor_len=B / 2, keep_scores=keep(B, u)))
    elif what == "decomp":
        for B in budgets:
            for u in positions(B):
                for s in seeds:
                    rows.append(run(v, u, B, "T_splitF_mainC", s))
    elif what == "featsets":
        for u in positions(120.0):
            for c in ("T_main_u13", "S_allC_u13", "T_main_notime", "S_sameC_notime", "S_allC_notime",
                      "T_split_notime"):
                for s in seeds:
                    rows.append(run(v, u, 120.0, c, s))
    elif what == "oem":
        if v in GM:
            for u in positions(120.0):
                for c in ("S_small", "S_fo71", "S_gm71"):
                    for s in seeds:
                        rows.append(run(v, u, 120.0, c, s))
    elif what == "srccal":
        for B in budgets:
            for u in positions(B):
                for L in (250.0, B / 2):
                    for s in seeds:
                        rows.append(run(v, u, B, "S_srcC", s, donor_len=L))
    elif what == "sweepsrc":
        for u in sweep_positions(v):
            for s in seeds:
                rows.append(run(v, u, 120.0, "S_srcC", s))
    elif what == "sweep":
        for u in sweep_positions(v):
            for c in MAIN:
                for s in seeds:
                    rows.append(run(v, u, 120.0, c, s))
    print(f"{what} {v} done", flush=True)
    return rows


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--what", required=True,
                     choices=["seeds10", "matched", "decomp", "featsets", "oem", "srccal", "sweep", "sweepsrc"])
    ap_.add_argument("--targets", default=",".join(VEH))
    ap_.add_argument("--seeds", default=None)
    ap_.add_argument("--workers", type=int, default=2)
    a = ap_.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else (list(range(3)) if a.what.startswith("sweep") else list(range(10)))
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(job, [(a.what, v, seeds) for v in a.targets.split(",")]) for r in rs]
    out = os.path.join(ROOT, f"results/runs/rev_{a.what}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("rows", len(rows), "->", out)
