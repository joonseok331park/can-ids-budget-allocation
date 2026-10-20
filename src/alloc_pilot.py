"""Exploratory pilot (before the protocol was fixed): budget allocation between detector training and threshold calibration.

Keeps the target profile segment P (short [250-B/2, 250) or reference [0, 250)) and the evaluation data fixed and
changes how the remaining target time [250, 250 + 0.5B) is used:
  A_main   F = [250, 250+0.3B) target windows (+ source if r>0), C = [250+0.3B, 250+0.5B)      (the main grid)
  A_cal    F = source windows only (n_src per donor), C = [250, 250+0.5B) target windows (all target time to C)
  A_split  F = first half of [250, 250+0.3B) target windows, C = second half of it + [250+0.3B, 250+0.5B)
  A_srcC   F = source windows only (same sample and model as A_cal), C = [250+0.3B, 250+0.5B) as in A_main
           -> A_srcC vs A_main isolates the training data; A_cal vs A_srcC isolates the calibration length
Windows straddling 250+0.3B are absent in every arm (segments were windowed separately), so A_cal's C has one
window fewer than a fresh segmentation of [250, 250+0.5B) would.
Output: results/runs/alloc_pilot.csv
"""
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(__file__))
from models_v2 import EXCLUDE_EVAL, EXCLUDE_PREFIX, FV2, ap, evaluate, meta_files, pick_source  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_SRC_PER_DONOR = 71


def load(target, B, pc):
    d = dict(np.load(os.path.join(FV2, target, f"B{B}_{pc}.npz")))
    excl = [i for i, a in enumerate(meta_files(target)) if a in EXCLUDE_EVAL.get(target, set())]
    if excl:
        mk = ~np.isin(d["att_file"], excl)
        for k in ("Xatt", "att_label", "att_ts", "att_te", "att_file"):
            d[k] = d[k][mk]
    d["_no_prefix"] = EXCLUDE_PREFIX.get(target, set())
    return d


def run(target, B, pc, arm, seed, r=0):
    d = load(target, B, pc)
    meta = json.load(open(os.path.join(FV2, target, "meta.json")))
    Xtr = d["Xtr"]
    Xc_main = d["Xcal"][d["cal_ts"] + 1.0 <= 0.2 * B + 1e-9]
    key = "src" if arm in ("A_cal", "A_srcC") else arm  # A_cal and A_srcC share the source sample and the model
    rng = np.random.default_rng(zlib.crc32(f"{target}|{B}|{key}|{r}|{seed}".encode()))
    if arm == "A_main":
        Xs = pick_source(target, 250.0, int(round(r * len(Xtr))), rng) if r else np.zeros((0, Xtr.shape[1]))
        Xfit, Xcal, ms = np.vstack([Xtr, Xs]), Xc_main, min(256, len(Xtr))
    elif arm == "A_cal":
        Xfit = pick_source(target, 250.0, 3 * N_SRC_PER_DONOR, rng)
        Xcal, ms = np.vstack([Xtr, Xc_main]), min(256, len(Xfit))
    elif arm == "A_srcC":
        Xfit = pick_source(target, 250.0, 3 * N_SRC_PER_DONOR, rng)
        Xcal, ms = Xc_main, min(256, len(Xfit))
    elif arm == "A_split":
        h = len(Xtr) // 2
        Xfit, Xcal, ms = Xtr[:h], np.vstack([Xtr[h + 1:], Xc_main]), min(256, h)
    clf = IsolationForest(n_estimators=200, max_samples=ms, random_state=seed).fit(Xfit)
    sc = lambda X: -clf.score_samples(X)  # noqa: E731
    s_cal, s_att, s_held = sc(Xcal), sc(d["Xatt"]), sc(d["Xheld"])
    tau = float(s_cal.max())
    row = {"target": target, "B": B, "profile": pc, "arm": arm, "r": r, "seed": seed, "n_fit": len(Xfit),
           "n_cal": len(Xcal), "max_samples": ms, "AP": ap(s_att, d["att_label"])}
    row.update(evaluate(d, meta, s_att, s_held, tau))
    return row


if __name__ == "__main__":
    rows = []
    for t in ("forester", "impala", "silverado", "traverse"):
        for B in (30, 120):
            for pc in ("short", "ref"):
                for s in (0, 1, 2):
                    rows.append(run(t, B, pc, "A_main", s, 0))
                    rows.append(run(t, B, pc, "A_main", s, 1))
                    rows.append(run(t, B, pc, "A_cal", s))
                    rows.append(run(t, B, pc, "A_split", s))
                    rows.append(run(t, B, pc, "A_srcC", s))
        print("done", t, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT + "/results/runs/alloc_pilot.csv", index=False)
    g = out.groupby(["target", "B", "arm", "r"]).agg(AP=("AP", "mean"), held_pos=("held_pos_frac", "mean"),
                                                     prefix_pos=("prefix_pos_frac", "mean"),
                                                     detect=("capture_detect_rate", "mean"), n_cal=("n_cal", "first")).round(3)
    print(g.to_string())
