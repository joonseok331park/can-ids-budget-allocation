"""Small hand-checkable example for metrics. Run: python3 src/test_metrics.py

Ten windows of width 1 s, stride 0.5 s: starts 0.0, 0.5, ..., 4.5; ends 1.0, 1.5, ..., 5.5.
Attack segment: [2.2, 3.0]. Windows overlapping it: starts 1.5, 2.0, 2.5 (indices 3, 4, 5).
Alarms at indices 1, 2, 5, 8:
  - events: {1,2} merge (0.5-1.5 touches 1.0-2.0) -> event (0.5, 2.0); {5} -> (2.5, 3.5); {8} -> (4.0, 5.0)  => 3 events
  - window TP = index 5 only (1); FP = 1, 2, 8 (3); FN = 3, 4 (2); TN = 0, 6, 7, 9 (4)
  - segment detected by window 5 (start 2.5 < 3.0, end 3.5 > 2.2); delay = 3.5 - 2.2 = 1.3 s
  - observed span 0.0 .. 5.5 s = 5.5/3600 h -> events/hour = 3 / (5.5/3600) = 1963.6...
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from metrics import alarm_events, events_per_hour, segment_detection, window_rates  # noqa: E402

ts = np.arange(10) * 0.5
te = ts + 1.0
lab = np.zeros(10, int); lab[[3, 4, 5]] = 1
alarm = np.zeros(10, bool); alarm[[1, 2, 5, 8]] = True

ev = alarm_events(ts, te, alarm)
assert ev == [(0.5, 2.0), (2.5, 3.5), (4.0, 5.0)], ev
n, h, rate = events_per_hour(ts, te, alarm)
assert n == 3 and abs(h - 5.5 / 3600) < 1e-12 and abs(rate - 3 / (5.5 / 3600)) < 1e-6, (n, h, rate)
r = window_rates(lab, alarm)
assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 3, 2, 4), r
det = segment_detection(ts, te, alarm, [(2.2, 3.0)])
assert det[0][0] and abs(det[0][1] - 1.3) < 1e-9, det
print("metrics hand example: all checks passed")
