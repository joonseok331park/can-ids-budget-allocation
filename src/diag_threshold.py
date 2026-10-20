"""Threshold diagnostics at B = 120, both positions, seed 0.

For each vehicle and position:
  1. Nested-C invariance: S_sameC and S_allC use the same source forest; S_sameC's C [96, 120) s is inside S_allC's
     C [60, 120) s, so tau_allC >= tau_sameC and alarms(allC) must be a subset of alarms(sameC). Checked on all
     evaluation windows.
  2. The C windows that set tau for S_allC and T_split (top 3 scores): their time, score, and the residual features
     that are furthest above the C-window median (in C-window MAD units).
  3. Attack captures that had a new alarm under the shorter C but lose it under the longer C (S_sameC -> S_allC), and
     under T_main -> T_split.
Output: results/diag_threshold.md
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import alloc_v3 as A  # noqa: E402
from alloc_models import eval_data, seg_windows  # noqa: E402
from eval_v3 import onsets  # noqa: E402
from features import FEATURES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

B, P = 120.0, 0.5


def scores(v, u, cond, seed=0):
    sc, _ = A.scorer(v, u, B, P, cond, seed)
    _, C = A.segments(B, P, cond)
    Xc = seg_windows(v, u, P * B, *C)
    d, meta = eval_data(v, u, P * B)
    return sc, Xc, sc(Xc), sc(d["Xatt"]), sc(d["Xheld"]), d, meta


def new_alarm_files(d, meta, alarm):
    hit = set()
    for i, info in enumerate(meta["attack_info"]):
        m = np.flatnonzero(d["att_file"] == i)
        if m.size == 0:
            continue
        m = m[np.argsort(d["att_ts"][m], kind="stable")]
        te = d["att_te"][m]
        ok = onsets(alarm[m]) & (te > info["first_attack"]) & (te <= info["last_attack"] + 1.0)
        if ok.any():
            hit.add((info["file"], info["kind"]))
    return hit


def top_windows(Xc, s_cal, C0, k=3):
    med = np.median(Xc, 0)
    mad = np.median(np.abs(Xc - med), 0) * 1.4826 + 1e-9
    out = []
    for j in np.argsort(-s_cal)[:k]:
        z = (Xc[j] - med) / mad
        top = np.argsort(-z)[:3]
        out.append(f"t={C0 + 0.5 * j:.1f}s score={s_cal[j]:.3f} " + ", ".join(f"{FEATURES[q]} {Xc[j, q]:.3g} (+{z[q]:.1f} MAD)" for q in top))
    return out


lines = ["# Threshold diagnostics (B = 120, seed 0)", "",
         "Times are seconds from the start u of the budget segment. MAD = distance from the median of the C windows, in MAD units.", ""]
rows = []
for v in ("forester", "impala", "silverado", "traverse"):
    for u in (190.0, 0.0):
        r = {"vehicle": v, "u": u}
        _, Xc1, c1, a1, h1, d, meta = scores(v, u, "S_sameC")
        _, Xc2, c2, a2, h2, _, _ = scores(v, u, "S_allC")
        t1, t2 = c1.max(), c2.max()
        al1, al2 = a1 > t1, a2 > t2
        r["same_scores"] = bool(np.allclose(a1, a2))
        r["tau_sameC"], r["tau_allC"] = round(float(t1), 4), round(float(t2), 4)
        r["subset_attack"] = bool(not (al2 & ~al1).any())
        r["subset_held"] = bool(not ((h2 > t2) & ~(h1 > t1)).any())
        lost_s = new_alarm_files(d, meta, al1) - new_alarm_files(d, meta, al2)
        _, Xm, cm, am, hm, _, _ = scores(v, u, "T_main")
        _, Xs, cs, as_, hs, _, _ = scores(v, u, "T_split")
        lost_t = new_alarm_files(d, meta, am > cm.max()) - new_alarm_files(d, meta, as_ > cs.max())
        r["lost_new_S_sameC_to_S_allC"] = len(lost_s)
        r["lost_new_T_main_to_T_split"] = len(lost_t)
        rows.append(r)
        lines += [f"## {v}, u = {u:g}", "",
                  f"- same source model: identical evaluation scores {r['same_scores']}, tau S_sameC {t1:.4f} -> S_allC {t2:.4f}, "
                  f"S_allC alarms a subset of S_sameC alarms (attack captures {r['subset_attack']}, held-out normal {r['subset_held']})",
                  "- C windows that set the S_allC threshold (top 3): " + " | ".join(top_windows(Xc2, c2, 60.0)),
                  "- C windows that set the T_split threshold (top 3): " + " | ".join(top_windows(Xs, cs, 78.0)),
                  f"- attack captures that lost their new alarm: S_sameC -> S_allC {len(lost_s)} "
                  + (", ".join(sorted(k for _, k in lost_s)) if lost_s else "")
                  + f" / T_main -> T_split {len(lost_t)} " + (", ".join(sorted(k for _, k in lost_t)) if lost_t else ""), ""]
    print(v, "done", flush=True)
summary = pd.DataFrame(rows)
lines = lines[:4] + [summary.to_markdown(index=False), ""] + lines[4:]
open(ROOT + "/results/diag_threshold.md", "w").write("\n".join(lines))
print(summary.to_string(index=False))
