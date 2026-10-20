"""Evaluation v3: separates alarms that were already on at attack start from new alarms.

Per attack capture (windows of that capture in time order, decision time = window end):
  on_at_start  the last window ending at or before the first attack frame fa is alarming
  new alarm    window j alarms, the previous window of the capture does not (or j is the first window),
               and fa < te[j] <= la + 1 s (la = last attack frame); delay = te[j] - fa (first such j)
  curve        share of attack captures with a new alarm within t s of fa, t in CURVE_T
Chance check on the held-out normal capture: virtual starts every VSTEP s; for each, did a new 0->1 alarm occur with
window end in (v, v + t]? Reported as the share of virtual starts, same t grid.
The same chance check is run inside each attack capture's normal prefix (starts v with v + t <= first attack frame),
pooled over captures (prefix_virtual_new_*), since attack captures carry their own normal-timing differences.
Also reports the v2 quantities (held/prefix alarm burden, events) so one call gives the whole row.
"""
from __future__ import annotations

import numpy as np

from metrics import alarm_events

CURVE_T = (1, 2, 5, 10, 30, 60)
VSTEP = 5.0
LA_PAD = 1.0


def onsets(alarm: np.ndarray) -> np.ndarray:
    """Boolean mask of 0->1 transitions (first window counts as a transition if it alarms)."""
    prev = np.r_[False, alarm[:-1]]
    return alarm & ~prev


def normal_burden(ts, te, alarm):
    if ts.size == 0:
        return 0.0, 0, np.nan, np.nan
    hours = float(te[-1] - ts[0]) / 3600.0
    n = len(alarm_events(ts, te, alarm))
    return hours, n, (n / hours if hours > 0 else np.nan), float(alarm.mean())


def virtual_starts(ts, te, alarm, step=VSTEP, horizon=max(CURVE_T)):
    """Share of virtual start points v (every `step` s) followed by a new alarm onset with end in (v, v+t]."""
    if ts.size < 3:
        return {t: np.nan for t in CURVE_T}
    on = onsets(alarm)
    ends = te[on]
    vs = np.arange(ts[0] + step, te[-1] - horizon, step)
    if vs.size == 0:
        return {t: np.nan for t in CURVE_T}
    out = {}
    for t in CURVE_T:
        lo = np.searchsorted(ends, vs, side="right")       # first onset end > v
        hi = np.searchsorted(ends, vs + t, side="right")   # onset ends <= v + t
        out[t] = float(np.mean(hi > lo))
    return out


def evaluate_v3(d, meta, s_att, s_held, tau, no_prefix=frozenset()):
    out = {}
    a_held = s_held > tau
    h, n, fa_h, pos = normal_burden(d["held_ts"], d["held_te"], a_held)
    out.update(held_hours=h, held_events=n, held_fa_per_h=fa_h, held_pos_frac=pos)
    for t, v in virtual_starts(d["held_ts"], d["held_te"], a_held).items():
        out[f"held_virtual_new_{t}s"] = v
    lab, ts, te, fid = d["att_label"], d["att_ts"], d["att_te"], d["att_file"]
    alarm = s_att > tau
    pre_h = pre_ev = pre_pos = pre_n = 0
    rows = []
    pv_hit = {t: 0 for t in CURVE_T}; pv_n = {t: 0 for t in CURVE_T}
    for i, info in enumerate(meta["attack_info"]):
        m = np.flatnonzero(fid == i)
        if m.size == 0:
            continue
        m = m[np.argsort(ts[m], kind="stable")]
        fa, la = info["first_attack"], info["last_attack"]
        a = alarm[m]
        pm = te[m] <= fa
        if pm.sum() >= 2 and info["file"] not in no_prefix:
            hh, nn, _, _ = normal_burden(ts[m][pm], te[m][pm], a[pm])
            pre_h += hh; pre_ev += nn; pre_pos += int(a[pm].sum()); pre_n += int(pm.sum())
        if info["file"] not in no_prefix and pm.sum() >= 3:
            # chance check inside the attack capture's own normal prefix: starts v with v + t <= fa
            e_on = te[m][pm][onsets(a[pm])]
            for t in CURVE_T:
                vs = np.arange(ts[m][0] + VSTEP, fa - t + 1e-9, VSTEP)
                if vs.size:
                    lo = np.searchsorted(e_on, vs, side="right"); hi = np.searchsorted(e_on, vs + t, side="right")
                    pv_hit[t] += int((hi > lo).sum()); pv_n[t] += int(vs.size)
        before = np.flatnonzero(pm)
        on_at_start = bool(before.size and a[before[-1]])
        cand = np.flatnonzero(onsets(a) & (te[m] > fa) & (te[m] <= la + LA_PAD))
        delay = float(te[m][cand[0]] - fa) if cand.size else np.nan
        any_pos_alarm = bool((a & (lab[m] == 1)).any())
        rows.append((info["kind"], on_at_start, cand.size > 0, delay, any_pos_alarm))
    k = np.array([r[0] for r in rows]); on0 = np.array([r[1] for r in rows]); new = np.array([r[2] for r in rows])
    dl = np.array([r[3] for r in rows]); anyp = np.array([r[4] for r in rows])
    out.update(prefix_hours=pre_h, prefix_events=pre_ev, prefix_fa_per_h=(pre_ev / pre_h if pre_h else np.nan),
               prefix_pos_frac=(pre_pos / pre_n if pre_n else np.nan),
               n_attack_captures=int(len(rows)), on_at_start_frac=float(on0.mean()),
               new_alarm_frac=float(new.mean()), new_alarm_misses=int((~new).sum()),
               new_alarm_median_delay_s=float(np.nanmedian(dl)) if new.any() else np.nan,
               any_positive_alarm_frac=float(anyp.mean()),
               att_pos_window_tpr=float(alarm[lab == 1].mean()))
    for t in CURVE_T:
        out[f"new_within_{t}s"] = float(np.mean(new & (dl <= t)))
        out[f"prefix_virtual_new_{t}s"] = pv_hit[t] / pv_n[t] if pv_n[t] else np.nan
    for kind in sorted(set(k)):
        out[f"new__{kind}"] = float(new[k == kind].mean())
    return out
