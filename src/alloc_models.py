"""Stage 2 for the allocation study: split a target budget B into profile P, training F and calibration C.

For vehicle v, offset o (s from the first frame of the budget capture), profile length Lp and training length Lf:
  P = [o, o + Lp), F = [o + Lp, o + Lp + Lf), C = [o + Lp + Lf, o + B)        (Lc = B - Lp - Lf)
Each segment is windowed on its own (1 s windows, 0.5 s stride, only windows fully inside the segment).
Training set: target F windows, plus n_src source windows (equal per donor, source profile 250 s, windows from the
donor's [250, 430) s) when n_src > 0. Lf = 0 with n_src > 0 is the source-trained detector.
IF: 200 trees, max_samples = min(256, n_fit). Threshold tau = max calibration score, alarm = score > tau.
Evaluation windows come from alloc_features.py (identical for every split with the same (o, Lp)).
"""
from __future__ import annotations

import functools
import json
import os
import sys
import zlib

import numpy as np
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(__file__))
from canio import load_ctt_csv  # noqa: E402
from experiment import CTT, ROOT, load_roles  # noqa: E402
from features import build_profile, slice_time, window_features  # noqa: E402
from models_v2 import EXCLUDE_EVAL, EXCLUDE_PREFIX, VEHICLES, ap, evaluate  # noqa: E402

FA = os.path.join(ROOT, "data/features_alloc")
FV2 = os.path.join(ROOT, "data/features_v2")


@functools.lru_cache(maxsize=4)
def budget_capture(v):
    r = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))[v]
    return load_ctt_csv(os.path.join(CTT, r["budget"]))


@functools.lru_cache(maxsize=64)
def target_profile(v, o, Lp):
    b = budget_capture(v)
    t0 = b["t"][0]
    return build_profile(slice_time(b, t0 + o, t0 + o + Lp))


def seg_windows(v, o, Lp, a, z):
    """Windows of the budget capture segment [o + a, o + z) relative to the profile (v, o, Lp)."""
    if z - a < 1.0:
        return np.zeros((0, 18))
    b = budget_capture(v)
    t0 = b["t"][0]
    A, Z = t0 + o + a, t0 + o + z
    return window_features(slice_time(b, A, Z), target_profile(v, o, Lp), 1.0, 0.5, t_origin=A, t_stop=Z)["X"]


@functools.lru_cache(maxsize=64)
def eval_data(v, o, Lp):
    d = dict(np.load(os.path.join(FA, v, f"o{o:g}_P{Lp:g}.npz")))
    meta = json.load(open(os.path.join(FA, v, "meta.json")))
    files = [x["file"] for x in meta["attack_info"]]
    excl = [i for i, f in enumerate(files) if f in EXCLUDE_EVAL.get(v, set())]
    if excl:
        mk = ~np.isin(d["att_file"], excl)
        for k in ("Xatt", "att_label", "att_ts", "att_te", "att_file"):
            d[k] = d[k][mk]
    d["_no_prefix"] = EXCLUDE_PREFIX.get(v, set())
    return d, meta


@functools.lru_cache(maxsize=8)
def source_pool(v):
    return {u: np.load(os.path.join(FV2, "source", f"{u}_P250.npz"))["X"] for u in VEHICLES if u != v}


def source_sample(v, n, rng):
    pool = source_pool(v)
    donors = sorted(pool)
    per = [n // len(donors) + (1 if i < n % len(donors) else 0) for i in range(len(donors))]
    return np.vstack([pool[u][rng.choice(len(pool[u]), k, replace=False)] for u, k in zip(donors, per)])


def run(v, o, B, Lp, Lf, n_src, seed, tag=""):
    Lc = B - Lp - Lf
    assert Lc >= 1.0, "calibration segment shorter than one window"
    Xf = seg_windows(v, o, Lp, Lp, Lp + Lf)
    Xc = seg_windows(v, o, Lp, Lp + Lf, B)
    rng = np.random.default_rng(zlib.crc32(f"{v}|{o}|{n_src}|{seed}".encode()))  # same source sample across splits
    Xs = source_sample(v, n_src, rng) if n_src else np.zeros((0, Xf.shape[1] if Xf.size else 18))
    Xfit = np.vstack([Xf.reshape(-1, 18), Xs])
    clf = IsolationForest(n_estimators=200, max_samples=min(256, len(Xfit)), random_state=seed).fit(Xfit)
    sc = lambda X: -clf.score_samples(X)  # noqa: E731
    d, meta = eval_data(v, o, Lp)
    s_cal, s_att, s_held = sc(Xc), sc(d["Xatt"]), sc(d["Xheld"])
    tau = float(s_cal.max())
    row = {"tag": tag, "target": v, "o": o, "B": B, "Lp": Lp, "Lf": Lf, "Lc": Lc, "n_src": n_src, "seed": seed,
           "n_F": len(Xf), "n_C": len(Xc), "n_fit": len(Xfit), "tau": tau, "AP": ap(s_att, d["att_label"])}
    row.update(evaluate(d, meta, s_att, s_held, tau))
    return row
