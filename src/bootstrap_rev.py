"""Bootstrap intervals for the B = 120 paired contrasts at the two fixed positions (scores from revision_runs.py).

Attack side: attack captures are resampled with replacement within each vehicle (the same resample for both
conditions of a contrast and for all seeds). AP is recomputed over the windows of the resampled captures, and
R_att,5 and R_nor,5 (prefix reference) from per-capture outcomes.
Held-out side: moving blocks of 20 s (40 windows) of the held-out capture, the same blocks for both conditions; the
held-out reference R_nor,5 resamples the virtual starts (every 5 s) in blocks of the same 20 s.
Each replicate gives the seed-mean value per cell and condition; the paired difference per cell and the median of
the 8 cell differences are reported with 2.5 and 97.5 % percentiles.
Usage: python3 src/bootstrap_rev.py [n_boot] -> results/rev_bootstrap.csv
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(__file__))
from alloc_models import eval_data  # noqa: E402
from eval_v3 import CURVE_T, LA_PAD, VSTEP, onsets  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.join(ROOT, "data/scores_rev")
VEH = ["forester", "impala", "silverado", "traverse"]
POS = [190.0, 0.0]
BLOCK = 40
T = 5.0
CONTRASTS = [("S_sameC", "T_main"), ("S_allC", "T_split"), ("T_split", "T_main"), ("S_allC", "S_sameC"),
             ("S_allC@m", "S_allC"), ("S_sameC@m", "S_sameC")]  # @m = donor profile matched to B/2


def load_scores(v, u, cond):
    if cond.endswith("@m"):
        pat = f"matched__{v}__u{u:g}__{cond[:-2]}__L60__s*.npz"
    else:
        pat = f"seeds10__{v}__u{u:g}__{cond}__L250__s*.npz"
    fs = sorted(glob.glob(os.path.join(SC, pat)), key=lambda f: int(f.rsplit("__s", 1)[1][:-4]))
    return [dict(np.load(f)) for f in fs]


def held_starts(d, alarm):
    """Indicator per virtual start of the held-out capture (as in eval_v3.virtual_starts): new alarm within T s."""
    ts, te = d["held_ts"], d["held_te"]
    ends = te[onsets(alarm)]
    vs = np.arange(ts[0] + VSTEP, te[-1] - max(CURVE_T), VSTEP)
    lo = np.searchsorted(ends, vs, side="right")
    hi = np.searchsorted(ends, vs + T, side="right")
    return hi > lo


def per_capture(d, meta, alarm, no_prefix):
    """Per attack capture: (window index array, new alarm within T s, prefix hits, prefix virtual starts)."""
    ts, te, fid = d["att_ts"], d["att_te"], d["att_file"]
    out = []
    for i, info in enumerate(meta["attack_info"]):
        m = np.flatnonzero(fid == i)
        if m.size == 0:
            continue
        m = m[np.argsort(ts[m], kind="stable")]
        fa, la = info["first_attack"], info["last_attack"]
        a = alarm[m]
        pm = te[m] <= fa
        hits = n = 0
        if info["file"] not in no_prefix and pm.sum() >= 3:
            e_on = te[m][pm][onsets(a[pm])]
            vs = np.arange(ts[m][0] + VSTEP, fa - T + 1e-9, VSTEP)
            if vs.size:
                lo = np.searchsorted(e_on, vs, side="right"); hi = np.searchsorted(e_on, vs + T, side="right")
                hits, n = int((hi > lo).sum()), int(vs.size)
        cand = np.flatnonzero(onsets(a) & (te[m] > fa) & (te[m] <= la + LA_PAD))
        new5 = bool(cand.size and te[m][cand[0]] - fa <= T)
        out.append((m, new5, hits, n))
    return out


def cell_stats(v, u, conds, n_boot, rng):
    d, meta = eval_data(v, u, 60.0)
    nop = frozenset(d["_no_prefix"])
    lab = d["att_label"]
    caps = None
    per = {}
    for c in conds:
        runs = load_scores(v, u, c)
        assert len(runs) == 10, (v, u, c, len(runs))
        per[c] = []
        for r in runs:
            pc = per_capture(d, meta, r["s_att"] > r["tau"], nop)
            caps = [x[0] for x in pc]
            a_held = r["s_held"] > r["tau"]
            per[c].append((r["s_att"], a_held, np.array([x[1] for x in pc]),
                           np.array([x[2] for x in pc]), np.array([x[3] for x in pc]), held_starts(d, a_held)))
    k = len(caps)
    nh = d["Xheld"].shape[0]
    starts = np.arange(0, nh - BLOCK + 1)
    nv = per[conds[0]][0][5].size
    vblock = int(BLOCK * 0.5 / VSTEP)  # the same 20 s blocks on the 5 s grid of virtual starts
    vstarts = np.arange(0, nv - vblock + 1)
    reps = []
    for b in range(n_boot + 1):
        if b == 0:
            ci, hb, vb = np.arange(k), np.arange(nh), np.arange(nv)
        else:
            ci = rng.integers(0, k, k)
            hb = np.concatenate([np.arange(s, s + BLOCK) for s in rng.choice(starts, int(np.ceil(nh / BLOCK)))])[:nh]
            vb = np.concatenate([np.arange(s, s + vblock) for s in rng.choice(vstarts, int(np.ceil(nv / vblock)))])[:nv]
        w = np.concatenate([caps[i] for i in ci])
        y = lab[w]
        rec = {"b": b}
        for c in conds:
            ap_, al, ratt, rnor, rheld = [], [], [], [], []
            for s_att, a_held, new5, hits, n, hv in per[c]:
                ap_.append(average_precision_score(y, s_att[w]))
                al.append(a_held[hb].mean())
                ratt.append(new5[ci].mean())
                rnor.append(hits[ci].sum() / max(n[ci].sum(), 1))
                rheld.append(hv[vb].mean())
            rec[(c, "AP")] = np.mean(ap_)
            rec[(c, "held_pos_frac")] = np.mean(al)
            rec[(c, "R_att5")] = np.mean(ratt)
            rec[(c, "E5")] = np.mean(np.array(ratt) - np.array(rnor))
            rec[(c, "E5_held")] = np.mean(np.array(ratt) - np.array(rheld))
        reps.append(rec)
    return reps


def main(n_boot=500):
    rng = np.random.default_rng(20261028)
    conds = sorted({c for p in CONTRASTS for c in p})
    allrep = {}
    for v in VEH:
        for u in POS:
            allrep[(v, u)] = cell_stats(v, u, conds, n_boot, rng)
            print(v, u, "done", flush=True)
    rows = []
    for a, b in CONTRASTS:
        for metric in ("AP", "held_pos_frac", "R_att5", "E5", "E5_held"):
            diffs = np.array([[r[(a, metric)] - r[(b, metric)] for r in allrep[cell]] for cell in allrep])  # cells x reps
            for j, cell in enumerate(allrep):
                x = diffs[j]
                rows.append({"contrast": f"{a} - {b}", "metric": metric, "cell": f"{cell[0]} u={cell[1]:g}",
                             "est": x[0], "lo": np.percentile(x[1:], 2.5), "hi": np.percentile(x[1:], 97.5)})
            med = np.median(diffs, axis=0)
            rows.append({"contrast": f"{a} - {b}", "metric": metric, "cell": "median of 8 cells",
                         "est": med[0], "lo": np.percentile(med[1:], 2.5), "hi": np.percentile(med[1:], 97.5)})
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results/rev_bootstrap.csv"), index=False, float_format="%.4f")
    print("saved results/rev_bootstrap.csv")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
