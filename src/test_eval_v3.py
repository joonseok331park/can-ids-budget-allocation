"""Hand-checkable examples for eval_v3. Run: python3 src/test_eval_v3.py

Attack capture windows: start 0, 0.5, ..., 9.5 (1 s long), first attack frame fa = 5.2 s, last la = 7.0 s.
  Pattern A: alarm on windows starting 2.0-3.0, off, on again 5.0-6.0.
    last window ending <= 5.2 ends at 5.0 (start 4.0) -> off -> on_at_start = False
    first onset with end in (5.2, 8.0]: window start 5.0, end 6.0 -> delay 0.8 s
  Pattern B: alarm on continuously from start 3.0 to 8.0
    window ending at 5.0 alarms -> on_at_start = True; no 0->1 onset inside (5.2, 8.0] -> no new alarm (1 miss)
Held-out normal windows: start 0..99.5, single alarming windows ending at 20.0 and 50.0.
  virtual starts every 5 s from 5 to 40.5 (= last end 100.5 - 60): v = 5, 10, ..., 40 (8 points)
  within 5 s : onset end 20 in (v, v+5] for v = 15 -> 1/8 = 0.125
  within 10 s: v = 10, 15 (end 20) and v = 40 (end 50) -> 3/8 = 0.375
  within 30 s: every v -> 1.0
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_v3 import evaluate_v3  # noqa: E402


def capture(pattern):
    ts = np.arange(0, 10, 0.5); te = ts + 1.0
    lab = ((te > 5.2) & (ts <= 7.0)).astype(int)
    if pattern == "A":
        al = ((ts >= 2.0) & (ts <= 3.0)) | ((ts >= 5.0) & (ts <= 6.0))
    else:
        al = (ts >= 3.0) & (ts <= 8.0)
    return ts, te, lab, al


def run(pattern):
    ts, te, lab, al = capture(pattern)
    hts = np.arange(0, 100, 0.5); hte = hts + 1.0
    hal = np.isin(hte, [20.0, 50.0])
    d = {"att_label": lab, "att_ts": ts, "att_te": te, "att_file": np.zeros(ts.size, int),
         "held_ts": hts, "held_te": hte}
    meta = {"attack_info": [{"file": "x", "kind": "k", "first_attack": 5.2, "last_attack": 7.0}]}
    # scores: alarm windows get 1.0, others 0.0, tau 0.5
    return evaluate_v3(d, meta, al.astype(float), hal.astype(float), 0.5)


a = run("A")
assert a["on_at_start_frac"] == 0.0 and a["new_alarm_frac"] == 1.0, a
assert abs(a["new_alarm_median_delay_s"] - 0.8) < 1e-9, a["new_alarm_median_delay_s"]
assert a["new_within_1s"] == 1.0 and a["new_within_2s"] == 1.0
b = run("B")
assert b["on_at_start_frac"] == 1.0 and b["new_alarm_frac"] == 0.0 and b["new_alarm_misses"] == 1, b
assert b["any_positive_alarm_frac"] == 1.0
assert abs(a["held_virtual_new_5s"] - 0.125) < 1e-9, a["held_virtual_new_5s"]
assert abs(a["held_virtual_new_10s"] - 0.375) < 1e-9, a["held_virtual_new_10s"]
assert a["held_virtual_new_30s"] == 1.0
print("all checks passed")
