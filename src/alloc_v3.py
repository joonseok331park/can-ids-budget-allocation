"""Allocation study with the fixed protocol of the paper.

Budget segment D_B = [u, u + B) of the target's budget capture (seconds from its first frame), profile P = [u, u + pB)
(p = 0.5 in the main design). With R = (1 - p) B left after the profile:
  T_main   target IF on F = [pB, pB + 0.6R), C = [pB + 0.6R, B)          (P 50 / F 30 / C 20 % when p = 0.5)
  S_sameC  source-only IF, C = [pB + 0.6R, B) (F segment unused)
  S_allC   the same source-only IF, C = [pB, B)
  T_split  target IF on F = [pB, pB + 0.3R), C = [pB + 0.3R, B)          (P 50 / F 15 / C 35 %)
  U_allC   no model: fixed score U, C = [pB, B)
Controls (same P and C as T_main): S_small (71 source windows, donor balanced, psi 71), S_large_fixed (213, psi 71),
S_large_auto (213, psi 213 = S_sameC).
Threshold: max of C scores (main) or the 95 % empirical quantile (diagnostic). Alarm = score > tau.
Source bank: per (target, seed) 71 windows from each of the three donors, reused for every B and u, so the
source-only forest depends only on (target, seed, bank size, psi).
"""
from __future__ import annotations

import functools
import os
import sys
import zlib

import numpy as np
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(__file__))
from alloc_models import eval_data, seg_windows, source_pool  # noqa: E402
from eval_v3 import evaluate_v3  # noqa: E402
from features import FEATURES  # noqa: E402
from models_v2 import ap  # noqa: E402

PER_DONOR = 71
# Fixed score U: 13 window-wide features, |x| for the log ratio, x/3 for mean z-scores, x otherwise
U_FEATURES = ["rate_logratio", "oov_frac", "missing_frac", "id_js", "id_rate_dev", "iat_z_mean", "iat_z_gt3",
              "iat_short_frac", "range_viol_frac", "byte_change_dev", "ham_z_mean", "dlc_mismatch_frac",
              "new_transition_frac"]
U_IDX = np.array([FEATURES.index(f) for f in U_FEATURES])
U_DIV = np.array([3.0 if f in ("iat_z_mean", "ham_z_mean") else 1.0 for f in U_FEATURES])


def u_score(X):
    g = X[:, U_IDX].copy()
    g[:, 0] = np.abs(g[:, 0])
    return (g / U_DIV).max(1)


@functools.lru_cache(maxsize=64)
def bank(v, seed):
    """Donor-balanced source bank: dict donor -> (71, 18) array, fixed per (target, seed)."""
    rng = np.random.default_rng(zlib.crc32(f"bank|{v}|{seed}".encode()))
    pool = source_pool(v)
    return {u: pool[u][rng.choice(len(pool[u]), PER_DONOR, replace=False)] for u in sorted(pool)}


def source_rows(v, seed, n):
    """First n rows of the bank, balanced across donors (71 per donor when n = 213)."""
    b = bank(v, seed)
    donors = sorted(b)
    per = [n // len(donors) + (1 if i < n % len(donors) else 0) for i in range(len(donors))]
    return np.vstack([b[u][:k] for u, k in zip(donors, per)]), dict(zip(donors, per))


@functools.lru_cache(maxsize=256)
def source_forest(v, seed, n, psi):
    X, counts = source_rows(v, seed, n)
    clf = IsolationForest(n_estimators=200, max_samples=min(psi, len(X)), random_state=seed).fit(X)
    return clf, counts


def segments(B, p, cond):
    pB = p * B
    R = B - pB
    if cond in ("T_main", "S_sameC", "S_small", "S_large_fixed", "S_large_auto"):
        F = (pB, pB + 0.6 * R) if cond == "T_main" else None
        C = (pB + 0.6 * R, B)
    elif cond == "T_split":
        F, C = (pB, pB + 0.3 * R), (pB + 0.3 * R, B)
    elif cond in ("S_allC", "U_allC"):
        F, C = None, (pB, B)
    else:
        raise ValueError(cond)
    return F, C


def scorer(v, u, B, p, cond, seed):
    """Returns (score function, info dict) for one condition."""
    Lp = p * B
    F, _ = segments(B, p, cond)
    info = {"n_F": 0, "n_fit": 0, "max_samples_": np.nan, "donor_rows": ""}
    if cond == "U_allC":
        return u_score, info
    if cond in ("T_main", "T_split"):
        Xf = seg_windows(v, u, Lp, *F)
        clf = IsolationForest(n_estimators=200, max_samples=min(256, len(Xf)), random_state=seed).fit(Xf)
        info.update(n_F=len(Xf), n_fit=len(Xf))
    else:
        n, psi = {"S_sameC": (213, 256), "S_allC": (213, 256), "S_large_auto": (213, 256),
                  "S_large_fixed": (213, 71), "S_small": (71, 71)}[cond]
        clf, counts = source_forest(v, seed, n, psi)
        info.update(n_fit=n, donor_rows=";".join(f"{k}:{c}" for k, c in counts.items()))
    info["max_samples_"] = int(clf.max_samples_)
    return (lambda X: -clf.score_samples(X)), info


def run(v, u, B, cond, seed, p=0.5, thr="max", c_pick=None, tag=None):
    """c_pick: None (all C windows) or ('spread', n) = n windows at even spacing over the C segment."""
    Lp = p * B
    _, C = segments(B, p, cond)
    sc, info = scorer(v, u, B, p, cond, seed)
    Xc = seg_windows(v, u, Lp, *C)
    if c_pick is not None and c_pick[0] == "spread":
        idx = np.unique(np.round(np.linspace(0, len(Xc) - 1, c_pick[1])).astype(int))
        Xc = Xc[idx]
    d, meta = eval_data(v, u, Lp)
    s_cal, s_att, s_held = sc(Xc), sc(d["Xatt"]), sc(d["Xheld"])
    tau = float(s_cal.max()) if thr == "max" else float(np.quantile(s_cal, 0.95))
    row = {"tag": tag or cond, "cond": cond, "target": v, "u": u, "B": B, "p": p, "seed": seed, "thr": thr,
           "c_pick": "all" if c_pick is None else f"{c_pick[0]}{c_pick[1]}", "n_C": len(Xc), "tau": tau,
           "C_start": C[0], "C_end": C[1], **info, "AP": ap(s_att, d["att_label"])}
    kinds = np.array([meta["attack_info"][i]["kind"] for i in d["att_file"]])
    for k in sorted(set(kinds)):
        row[f"AP__{k}"] = ap(s_att[kinds == k], d["att_label"][kinds == k])
    row.update(evaluate_v3(d, meta, s_att, s_held, tau, no_prefix=frozenset(d["_no_prefix"])))
    return row
