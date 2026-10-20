"""Hand-checkable example for models_v2.evaluate. Run: python3 src/test_eval_v2.py

One attack capture, windows of 1 s every 0.5 s starting at t=100: starts 100.0 .. 104.5 (10 windows).
First attack frame at 102.3, last at 103.2 -> positive windows: those containing an attack frame:
  [101.5,102.5) [102.0,103.0) [102.5,103.5) [103.0,104.0) -> indices 3,4,5,6
Prefix windows (end <= 102.3): indices 0,1,2 (ends 101.0, 101.5, 102.0). Prefix hours = (102.0-100.0)/3600.
Case A: alarms at 5 only -> detected, delay = 103.5 - 102.3 = 1.2 s, not pre-on, prefix events 0.
Case B: alarms at 2,3,4 -> event starts at window 2 (ends 102.0 <= 102.3) -> detected but pre-on;
        prefix events 1 (window 2), prefix positive fraction 1/3.
Held-out: 6 windows, alarms at 0 and 3 -> 2 events over (3.5)/3600 h.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from models_v2 import evaluate  # noqa: E402

ts = 100.0 + np.arange(10) * 0.5
te = ts + 1.0
lab = np.zeros(10, int); lab[[3, 4, 5, 6]] = 1
d = {"att_label": lab, "att_ts": ts, "att_te": te, "att_file": np.zeros(10, int),
     "held_ts": np.arange(6) * 0.5, "held_te": np.arange(6) * 0.5 + 1.0}
meta = {"attack_info": [{"kind": "x", "first_attack": 102.3, "last_attack": 103.2}]}
held_scores = np.array([2, 0, 0, 2, 0, 0], float)

sA = np.zeros(10); sA[5] = 2
oA = evaluate(d, meta, sA, held_scores, tau=1.0)
assert oA["capture_detect_rate"] == 1.0 and abs(oA["median_delay_s"] - 1.2) < 1e-9 and oA["n_pre_on"] == 0, oA
assert oA["prefix_events"] == 0 and abs(oA["prefix_hours"] - 2.0 / 3600) < 1e-12, oA
assert oA["held_events"] == 2 and abs(oA["held_hours"] - 3.5 / 3600) < 1e-12, oA

sB = np.zeros(10); sB[[2, 3, 4]] = 2
oB = evaluate(d, meta, sB, held_scores, tau=1.0)
assert oB["capture_detect_rate"] == 1.0 and oB["n_pre_on"] == 1 and np.isnan(oB["median_delay_s"]), oB
assert oB["prefix_events"] == 1 and abs(oB["prefix_pos_frac"] - 1 / 3) < 1e-12, oB

sC = np.zeros(10); sC[5] = 1.0  # equal to tau -> no alarm under strict ">"
oC = evaluate(d, meta, sC, held_scores, tau=1.0)
assert oC["capture_detect_rate"] == 0.0, oC
print("evaluate v2 hand example: all checks passed")
