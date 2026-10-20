"""Stage 2 (v2): Isolation Forest conditions on nested-profile features, with fixed source selection.

Main grid: target x B x profile(short/ref) x r(0,1) x seed, source P = 250 s, max_samples = min(256, N_target_fit).
Contrasts are separate condition names (see CONTRASTS). Alarm rule: score > tau, tau = max calibration score.
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
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURES  # noqa: E402
from metrics import alarm_events  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FV2 = os.path.join(ROOT, "data/features_v2")
VEHICLES = ["forester", "impala", "silverado", "traverse"]
# results/reuse_check.csv and results/reuse_perid.csv: the normal frames of Impala standstill-1 are the budget
# capture itself (overlap with P/F/C) -> excluded from all evaluation; the normal frames of standstill-2 are the
# held-out capture -> kept for AP/detection but excluded from the prefix normal evaluation (no double counting).
EXCLUDE_EVAL = {"impala": {"set_01/train_01/standstill-1.csv"}}
EXCLUDE_PREFIX = {"impala": {"set_01/train_01/standstill-2.csv"}}


def ap(scores, labels):
    labels = np.asarray(labels).astype(int)
    if labels.sum() == 0 or labels.sum() == labels.size:
        return np.nan
    return float(average_precision_score(labels, scores))


def pick_source(target, plen, n_add, rng):
    donors = [v for v in VEHICLES if v != target]
    per = [n_add // len(donors) + (1 if i < n_add % len(donors) else 0) for i in range(len(donors))]
    xs = []
    for v, n in zip(donors, per):
        X = np.load(os.path.join(FV2, "source", f"{v}_P{plen:g}.npz"))["X"]
        if n > len(X):
            raise ValueError(f"not enough source windows from {v}: need {n}, have {len(X)}")
        xs.append(X[rng.choice(len(X), n, replace=False)] if n else np.zeros((0, X.shape[1])))
    return np.vstack(xs)


def normal_eval(ts, te, alarm):
    """Continuous normal stretch: hours, alarm events, events/hour, positive window fraction."""
    if ts.size == 0:
        return 0.0, 0, np.nan, np.nan
    hours = float(te[-1] - ts[0]) / 3600.0
    n = len(alarm_events(ts, te, alarm))
    return hours, n, n / hours if hours > 0 else np.nan, float(alarm.mean())


def evaluate(d, meta, s_att, s_held, tau):
    out = {}
    a_held = s_held > tau
    out["held_hours"], out["held_events"], out["held_fa_per_h"], out["held_pos_frac"] = normal_eval(d["held_ts"], d["held_te"], a_held)
    lab, ts, te, fid = d["att_label"], d["att_ts"], d["att_te"], d["att_file"]
    alarm = s_att > tau
    pre_h = pre_ev = pre_pos = pre_n = 0
    det_rows = []
    for i, info in enumerate(meta["attack_info"]):
        m = fid == i
        fa = float(d["att_first"][i]) if "att_first" in d else info["first_attack"]
        if m.sum() == 0:  # file excluded from evaluation
            continue
        pm = m & (te <= fa) & (info.get("file") not in d.get("_no_prefix", set()))
        if pm.sum() >= 2:
            h, n, _, _ = normal_eval(ts[pm], te[pm], alarm[pm])
            pre_h += h; pre_ev += n; pre_pos += int(alarm[pm].sum()); pre_n += int(pm.sum())
        hit = np.flatnonzero(m & (lab == 1) & alarm)
        if hit.size:
            j = hit[0]
            delay = float(te[j] - fa)
            # was the alarm already on before the attack? walk back through consecutive alarm windows
            idx_m = np.flatnonzero(m)
            pos = np.searchsorted(idx_m, j)
            while pos > 0 and alarm[idx_m[pos - 1]] and ts[idx_m[pos]] <= te[idx_m[pos - 1]] + 1e-9:
                pos -= 1
            pre_on = bool(te[idx_m[pos]] <= fa)
            det_rows.append((info["kind"], True, delay, pre_on))
        else:
            det_rows.append((info["kind"], False, np.nan, False))
    out["prefix_hours"] = pre_h
    out["prefix_events"] = pre_ev
    out["prefix_fa_per_h"] = pre_ev / pre_h if pre_h else np.nan
    out["prefix_pos_frac"] = pre_pos / pre_n if pre_n else np.nan
    det = pd.DataFrame(det_rows, columns=["kind", "detected", "delay", "pre_on"])
    out["capture_detect_rate"] = float(det.detected.mean())
    fresh = det[det.detected & ~det.pre_on]
    out["n_pre_on"] = int((det.detected & det.pre_on).sum())
    out["median_delay_s"] = float(fresh.delay.median()) if len(fresh) else np.nan
    for k, g in det.groupby("kind"):
        out[f"det__{k}"] = float(g.detected.mean())
    out["att_pos_window_tpr"] = float(alarm[lab == 1].mean())
    return out


def meta_files(target):
    return [a["file"] for a in json.load(open(os.path.join(FV2, target, "meta.json")))["attack_info"]]


def run_condition(target, B, pcond, r, seed, src_plen=250.0, ms_rule="target", c_len=None, tag="main", variant="",
                  drop=()):
    d = dict(np.load(os.path.join(FV2, target, f"B{B}_{pcond}.npz")))
    if variant:  # e.g. "clk": replace the evaluation arrays, keep training and calibration windows
        d.update(dict(np.load(os.path.join(FV2, target, f"B{B}_{pcond}_{variant}.npz"))))
    excl = [i for i, a in enumerate(meta_files(target)) if a in EXCLUDE_EVAL.get(target, set())]
    if excl:
        mk = ~np.isin(d["att_file"], excl)
        for k in ("Xatt", "att_label", "att_ts", "att_te", "att_file"):
            d[k] = d[k][mk]
    d["_no_prefix"] = EXCLUDE_PREFIX.get(target, set())
    keep = np.array([f not in set(drop) for f in FEATURES])
    if drop:  # feature-group ablation: remove columns everywhere (training, calibration, evaluation, source)
        for k in ("Xtr", "Xcal", "Xatt", "Xheld"):
            d[k] = d[k][:, keep]
    meta = json.load(open(os.path.join(FV2, target, "meta.json")))
    Xtr = d["Xtr"]
    c_len = 0.2 * B if c_len is None else c_len
    cm = d["cal_ts"] + 1.0 <= c_len + 1e-9
    Xcal = d["Xcal"][cm]
    # source selection depends only on (target, B, r, seed): identical across profile conditions
    rng = np.random.default_rng(zlib.crc32(f"{target}|{B}|{r}|{seed}|{src_plen}".encode()))
    n_add = int(round(r * len(Xtr)))
    Xs = pick_source(target, src_plen, n_add, rng)[:, keep] if n_add else np.zeros((0, Xtr.shape[1]))
    Xfit = np.vstack([Xtr, Xs])
    ms = min(256, len(Xtr)) if ms_rule == "target" else min(256, len(Xfit))
    clf = IsolationForest(n_estimators=200, max_samples=ms, random_state=seed).fit(Xfit)
    sc = lambda X: -clf.score_samples(X)  # noqa: E731  (higher = more anomalous)
    s_cal, s_att, s_held = sc(Xcal), sc(d["Xatt"]), sc(d["Xheld"])
    tau = float(s_cal.max())
    row = {"tag": tag, "variant": variant or "none", "dropped": "+".join(drop) or "none", "target": target, "B": B, "profile": pcond, "r": r, "seed": seed, "src_P": src_plen,
           "max_samples_rule": ms_rule, "max_samples": ms, "n_target_fit": len(Xtr), "n_source": len(Xs),
           "n_cal": int(cm.sum()), "c_len_s": c_len, "tau": tau, "profile_ids": int(d["profile_ids"]),
           "AP": ap(s_att, d["att_label"])}
    kinds = np.array([meta["attack_info"][i]["kind"] for i in d["att_file"]])
    for k in sorted(set(kinds)):
        m = kinds == k
        row[f"AP__{k}"] = ap(s_att[m], d["att_label"][m])
    row.update(evaluate(d, meta, s_att, s_held, tau))
    return row


def main_grid(targets, budgets, seeds, variant=""):
    rows = []
    for t in targets:
        for B in budgets:
            for p in ("short", "ref"):
                for r in (0, 1):
                    for s in seeds:
                        rows.append(run_condition(t, B, p, r, s, variant=variant, tag="main" if not variant else variant))
    return rows


def contrasts(targets, seeds):
    rows = []
    for t in targets:
        for s in seeds:
            rows.append(run_condition(t, 120, "short", 0.25, s, tag="c1_r025"))
            rows.append(run_condition(t, 120, "short", 1, s, ms_rule="merged", tag="c2_ms_merged"))
            rows.append(run_condition(t, 30, "short", 1, s, src_plen=15.0, tag="c3_matched_srcP15"))
    return rows


def c_length(targets, budgets, seeds):
    rows = []
    for t in targets:
        for B in budgets:
            for p in ("short", "ref"):
                for r in (0, 1):
                    for s in seeds:
                        for c in (6.0, 24.0, 60.0):
                            rows.append(run_condition(t, B, p, r, s, c_len=c, tag=f"clen_{int(c)}"))
    return rows


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--what", choices=["main", "contrasts", "clen", "clk"], required=True)
    ap_.add_argument("--targets", default="silverado,forester")
    ap_.add_argument("--budgets", default="120,30")
    ap_.add_argument("--seeds", default="0,1,2")
    ap_.add_argument("--out", required=True)
    a = ap_.parse_args()
    T = a.targets.split(","); Bs = [int(x) for x in a.budgets.split(",")]; Ss = [int(x) for x in a.seeds.split(",")]
    rows = {"main": lambda: main_grid(T, Bs, Ss), "contrasts": lambda: contrasts(T, Ss),
            "clen": lambda: c_length(T, Bs, Ss), "clk": lambda: main_grid(T, Bs, Ss, variant="clk")}[a.what]()
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("rows", len(rows))
