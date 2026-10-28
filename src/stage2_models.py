"""Stage 2: fit Isolation Forests on cached features and evaluate.

Conditions per cached (target, budget, offset, profile):
  source_ratio r: target train windows + r x as many source windows (pooled from the other vehicles)
  control "jitter": target train windows + r x as many jittered copies of target windows (same count as source,
                    no information from other vehicles) -> separates "more samples" from "source information"
Thresholds are computed from calibration windows only; several rules are recorded side by side.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(__file__))
from metrics import auprc, events_per_hour, segment_detection  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(ROOT, "data/features")
THRESH_RULES = {"max": lambda c: c.max(), "q95": lambda c: np.quantile(c, 0.95)}


def load_source(exclude: str, plen: float) -> np.ndarray:
    xs = []
    for f in sorted(glob.glob(os.path.join(FEAT, "source", f"*_p{plen:g}.npz"))):
        v = os.path.basename(f).split("_p")[0]
        if v != exclude:
            xs.append(np.load(f)["X"])
    return np.vstack(xs)


def evaluate(scores_att, thr, d, kinds):
    lab = d["att_label"]; ts = d["att_ts"]; te = d["att_te"]; fid = d["att_file"]
    alarm = scores_att >= thr
    out = {}
    # clean normal prefixes: windows ending >= 1 s before the first attack segment of their file
    first = {}
    for i, a, b in d["segments"]:
        first[int(i)] = min(first.get(int(i), np.inf), a)
    pre_ev, pre_h, pre_al, pre_n = 0, 0.0, 0, 0
    det = []
    for i in np.unique(fid):
        m = fid == i
        keep = m & (te <= first.get(int(i), np.inf) - 1.0)
        if keep.sum() >= 2:
            ne, hh, _ = events_per_hour(ts[keep], te[keep], alarm[keep])
            pre_ev += ne; pre_h += hh; pre_al += int(alarm[keep].sum()); pre_n += int(keep.sum())
        segs = [(a, b) for j, a, b in d["segments"] if int(j) == i]
        for ok, dl in segment_detection(ts[m], te[m], alarm[m], segs):
            det.append((kinds[int(i)], ok, dl))
    out["prefix_events_per_hour"] = pre_ev / pre_h if pre_h else np.nan
    out["prefix_window_fpr"] = pre_al / pre_n if pre_n else np.nan
    out["prefix_hours"] = pre_h
    det = pd.DataFrame(det, columns=["kind", "ok", "delay"])
    out["segment_recall"] = float(det.ok.mean()) if len(det) else np.nan
    out["median_delay_s"] = float(det.delay.median()) if det.ok.any() else np.nan
    for k, g in det.groupby("kind"):
        out[f"recall__{k}"] = float(g.ok.mean())
    out["att_window_tpr"] = float(alarm[lab == 1].mean()) if (lab == 1).any() else np.nan
    return out


def run(cfg, out_csv):
    rows = []
    for tdir in sorted(glob.glob(os.path.join(FEAT, "*"))):
        tv = os.path.basename(tdir)
        if tv == "source" or not os.path.isdir(tdir):
            continue
        if cfg.get("targets") and tv not in cfg["targets"]:
            continue
        if not os.path.exists(os.path.join(tdir, "files.json")):
            continue  # stage 1 not finished for this target
        files = json.load(open(os.path.join(tdir, "files.json")))
        kinds = files["kinds"]
        for f in sorted(glob.glob(os.path.join(tdir, "B*_o*_*.npz"))):
            B, o, pcond = re.match(r"B(\d+)_o(\d+)_(\w+)\.npz", os.path.basename(f)).groups()
            B, o = int(B), int(o)
            if cfg.get("budgets") and B not in cfg["budgets"]:
                continue
            d = np.load(f)
            Xtr, Xcal, Xatt = d["Xtr"], d["Xcal"], d["Xatt"]
            if len(Xtr) < 4 or len(Xcal) < 2:
                continue
            src = load_source(tv, round(B * cfg["split"][0], 3)) if cfg["source_profile"] == "matched" \
                else load_source(tv, float(cfg["source_profile_long_s"]))
            sd = Xtr.std(0) + 1e-9
            for mix, ratio, seed in [(m, r, s) for m in cfg["mixes"] for r in cfg["source_ratios"] for s in cfg["seeds"]]:
                if ratio == 0 and mix != "source":
                    continue  # ratio 0 is the same target-only model for every mix
                rng = np.random.default_rng(1000 * seed + 7)
                n_add = int(round(ratio * len(Xtr)))
                if mix == "source":
                    if n_add > len(src):
                        continue
                    Xadd = src[rng.choice(len(src), n_add, replace=False)] if n_add else np.zeros((0, Xtr.shape[1]))
                else:  # jitter control: resampled target windows + Gaussian noise (0.1 sd per feature)
                    base = Xtr[rng.integers(0, len(Xtr), n_add)]
                    Xadd = base + rng.normal(0, 0.1, base.shape) * sd
                Xfit = np.vstack([Xtr, Xadd])
                ms = min(cfg["max_samples"], len(Xfit))
                clf = IsolationForest(n_estimators=cfg["n_estimators"], max_samples=ms, random_state=seed).fit(Xfit)
                sc = lambda X: -clf.score_samples(X)  # noqa: E731
                s_cal, s_att = sc(Xcal), sc(Xatt)
                base_row = {"target": tv, "budget_s": B, "offset_s": o, "profile": pcond,
                            "mix": "target_only" if ratio == 0 else mix, "ratio": ratio, "seed": seed,
                            "n_train_target": len(Xtr), "n_added": n_add, "n_cal": len(Xcal),
                            "auprc": auprc(s_att, d["att_label"])}
                for rule, fn in THRESH_RULES.items():
                    thr = fn(s_cal)
                    row = dict(base_row, thr_rule=rule, threshold=thr)
                    row.update(evaluate(s_att, thr, d, kinds))
                    if "Xheld" in d.files:
                        n_ev, hrs, eph = events_per_hour(d["held_ts"], d["held_te"], sc(d["Xheld"]) >= thr)
                        row["heldout_events_per_hour"] = eph
                    rows.append(row)
        print("done", tv, flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


DEFAULT2 = {"split": [0.5, 0.3, 0.2], "source_profile": "matched", "source_profile_long_s": 600,
            "mixes": ["source", "jitter"], "source_ratios": [0, 1, 4], "seeds": [0, 1, 2],
            "n_estimators": 200, "max_samples": 256, "budgets": None, "targets": None}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = dict(DEFAULT2)
    if a.config:
        cfg.update(json.load(open(a.config)))
    json.dump(cfg, open(a.out.replace(".csv", "_config.json"), "w", encoding="utf-8"), indent=1)
    run(cfg, a.out)
