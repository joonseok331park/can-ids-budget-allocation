"""Source-vs-no-source comparison under a limited target normal budget (CTT vehicles as targets).

For each target vehicle, budget B, budget offset o and profile condition:
  budget window [o, o+B) of the target's budget capture is split into profile / train / calibration segments
  target profile  : from the profile segment ("budget") or from a long separate stretch ("reference", diagnostic only)
  target windows  : residual features of the train and calibration segments relative to the target profile
  source windows  : other vehicles' normal residuals, each relative to its own profile built from the same
                    profile length as the target ("matched") or from a long stretch ("long")
  detector        : IsolationForest on target train windows, optionally + source windows (ratio r x target count)
  threshold       : from calibration-segment scores only
  evaluation      : held-out normal capture (alarm events/hour) and the target's attack captures (AUPRC, recall, delay)
Only the IF training data differs between the target-only and source+target conditions.
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(__file__))
from canio import ensure_sorted, load_ctt_csv  # noqa: E402
from features import build_profile, slice_time, window_features  # noqa: E402
from manifest import attack_segments  # noqa: E402
from metrics import auprc, events_per_hour, segment_detection, window_rates  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTT = os.path.join(ROOT, "data/raw/ctt")


def load_roles(manifest_path: str) -> dict:
    m = pd.read_csv(manifest_path)
    m = m.drop_duplicates("sha256")  # identical files across sets count once
    roles = {}
    for v, g in m.groupby("vehicle"):
        af = g[g.kind == "attack-free"].sort_values("duration_s", ascending=False)  # longest = budget capture
        acc = g[g.kind == "accessory"].sort_values("file")
        att = g[~g.kind.isin(["attack-free", "accessory"]) & (g.attack_frames > 0)].sort_values(["kind", "file"])
        roles[v] = {"budget": af.path.iloc[0], "budget_dur": float(af.duration_s.iloc[0]),
                    "reference": af.path.iloc[1] if len(af) > 1 else None,
                    "accessory": acc.path.tolist(), "attacks": att.path.tolist(),
                    "attack_kinds": att.kind.tolist()}
    return roles


def prefix_windows(w: dict, segs, margin: float = 1.0):
    """Windows that end before the first attack segment starts (minus margin): clean normal prefix."""
    if not segs:
        keep = np.ones(w["t_start"].size, bool)
    else:
        keep = w["t_end"] <= segs[0][0] - margin
    return keep


def seg_split(t0, B, fr):
    a = t0
    p_end = a + B * fr[0]
    tr_end = p_end + B * fr[1]
    return (a, p_end), (p_end, tr_end), (tr_end, a + B)


def run(cfg: dict, out_csv: str):
    roles = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))
    targets = cfg["targets"] or sorted(roles)
    W, S = cfg["width"], cfg["stride"]
    fr = cfg["split"]
    rows = []
    src_cache: dict = {}

    def source_windows(sv, prof_len, o_s):
        key = (sv, round(prof_len, 3), round(o_s, 3))
        if key not in src_cache:
            cap = load_ctt_csv(os.path.join(CTT, roles[sv]["budget"]))
            t0 = cap["t"][0] + o_s
            plen = prof_len if cfg["source_profile"] == "matched" else cfg["source_profile_long_s"]
            prof = build_profile(slice_time(cap, t0, t0 + plen))
            seg = slice_time(cap, t0 + plen, t0 + plen + cfg["source_window_span_s"])
            src_cache[key] = window_features(seg, prof, W, S)["X"]
        return src_cache[key]

    for tv in targets:
        r = roles[tv]
        bcap = load_ctt_csv(os.path.join(CTT, r["budget"]))
        rcap = load_ctt_csv(os.path.join(CTT, r["reference"])) if r["reference"] else None
        acaps = [(p, k, ensure_sorted(load_ctt_csv(os.path.join(CTT, p)))) for p, k in zip(r["attacks"], r["attack_kinds"])]
        bdur = bcap["t"][-1] - bcap["t"][0]
        for B in cfg["budgets"]:
            step = max(B, cfg["offset_step_min_s"])
            offsets = [float(o) for o in np.arange(0, bdur - B, step)][: cfg["n_offsets"]]
            for o in offsets:
                t0 = bcap["t"][0] + o
                (pa, pb), (ta, tb), (ca, cb) = seg_split(t0, B, fr)
                for pcond in cfg["profile_conditions"]:
                    if pcond == "budget":
                        prof = build_profile(slice_time(bcap, pa, pb))
                    else:  # reference: long profile from the other attack-free capture (diagnostic, exceeds budget)
                        if rcap is None:
                            continue
                        prof = build_profile(slice_time(rcap, rcap["t"][0], rcap["t"][0] + cfg["reference_profile_s"]))
                    Xtr = window_features(slice_time(bcap, ta, tb), prof, W, S)["X"]
                    Xcal = window_features(slice_time(bcap, ca, cb), prof, W, S)["X"]
                    if len(Xtr) < 4 or len(Xcal) < 2:
                        continue
                    # secondary false-alarm set: the other attack-free capture (only when it is not the profile source)
                    hw = window_features(rcap, prof, W, S) if (rcap is not None and pcond == "budget") else None
                    aw = [(p, k, window_features(c, prof, W, S), attack_segments(c["t"], c["label"])) for p, k, c in acaps]
                    prof_len = B * fr[0]
                    srcX = np.vstack([source_windows(sv, prof_len, cfg["source_offset_s"])
                                      for sv in sorted(roles) if sv != tv])
                    for ratio, seed in itertools.product(cfg["source_ratios"], cfg["seeds"]):
                        rng = np.random.default_rng(1000 * seed + 7)
                        n_src = int(round(ratio * len(Xtr)))
                        if n_src > len(srcX):
                            continue
                        Xs = srcX[rng.choice(len(srcX), n_src, replace=False)] if n_src else np.zeros((0, Xtr.shape[1]))
                        Xfit = np.vstack([Xtr, Xs])
                        ms = min(cfg["max_samples"], len(Xfit))
                        clf = IsolationForest(n_estimators=cfg["n_estimators"], max_samples=ms, random_state=seed)
                        clf.fit(Xfit)
                        sc = lambda X: -clf.score_samples(X) if len(X) else np.zeros(0)  # noqa: E731
                        cal_s = sc(Xcal)
                        thr = cal_s.max() if cfg["threshold"] == "max_calib" else np.quantile(cal_s, cfg["threshold"])
                        row = {"target": tv, "budget_s": B, "offset_s": o, "profile": pcond, "source_ratio": ratio,
                               "seed": seed, "n_train_target": len(Xtr), "n_train_source": n_src, "n_cal": len(Xcal),
                               "max_samples": ms, "threshold": thr, "profile_ids": int(prof.vocab.size)}
                        if hw is not None and len(hw["X"]):
                            hs = sc(hw["X"])
                            n_ev, hrs, eph = events_per_hour(hw["t_start"], hw["t_end"], hs >= thr)
                            row.update({"heldout_events": n_ev, "heldout_hours": hrs, "heldout_events_per_hour": eph,
                                        "heldout_window_fpr": float((hs >= thr).mean())})
                        all_s, all_l, det_rows = [], [], []
                        pre_ev, pre_h, pre_alarm, pre_n = 0, 0.0, 0, 0
                        for p, k, w, segs in aw:
                            s_ = sc(w["X"])
                            all_s.append(s_); all_l.append(w["label"])
                            keep = prefix_windows(w, segs)
                            if keep.sum() >= 2:
                                ne, hh, _ = events_per_hour(w["t_start"][keep], w["t_end"][keep], (s_ >= thr)[keep])
                                pre_ev += ne; pre_h += hh
                                pre_alarm += int((s_ >= thr)[keep].sum()); pre_n += int(keep.sum())
                            for d, dl in segment_detection(w["t_start"], w["t_end"], s_ >= thr, segs):
                                det_rows.append((k, d, dl))
                        all_s = np.concatenate(all_s); all_l = np.concatenate(all_l)
                        row["attack_files_auprc"] = auprc(all_s, all_l)
                        row.update({"prefix_events": pre_ev, "prefix_hours": pre_h,
                                    "prefix_events_per_hour": pre_ev / pre_h if pre_h > 0 else np.nan,
                                    "prefix_window_fpr": pre_alarm / pre_n if pre_n else np.nan})
                        wr = window_rates(all_l, all_s >= thr)
                        row.update({"attack_files_window_tpr": wr["tpr"], "attack_files_window_fpr": wr["fpr"]})
                        det = pd.DataFrame(det_rows, columns=["kind", "detected", "delay"])
                        row["segments"] = len(det)
                        row["segment_recall"] = float(det.detected.mean()) if len(det) else np.nan
                        row["median_delay_s"] = float(det.delay.median()) if det.detected.any() else np.nan
                        for k, g in det.groupby("kind"):
                            row[f"recall__{k}"] = float(g.detected.mean())
                        rows.append(row)
                print(f"[{time.strftime('%H:%M:%S')}] {tv} B={B} o={o} done", flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return rows


DEFAULT = {
    "targets": None, "width": 1.0, "stride": 0.5, "split": [0.5, 0.3, 0.2],
    "budgets": [30, 120, 300], "n_offsets": 4, "offset_step_min_s": 120, "profile_conditions": ["budget", "reference"],
    "reference_profile_s": 600, "source_profile": "matched", "source_profile_long_s": 600,
    "source_offset_s": 0, "source_window_span_s": 300, "source_ratios": [0, 1, 4], "seeds": [0, 1, 2],
    "n_estimators": 200, "max_samples": 256, "threshold": "max_calib",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="JSON file overriding DEFAULT")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = dict(DEFAULT)
    if a.config:
        cfg.update(json.load(open(a.config)))
    json.dump(cfg, open(a.out.replace(".csv", "_config.json"), "w", encoding="utf-8"), indent=1)
    run(cfg, a.out)
