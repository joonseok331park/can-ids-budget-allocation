"""Where do timestamp reversals occur? For every unique CTT capture: reversals (t[i+1] < t[i]) split by whether the
two frames involved are both normal, both attack-labelled or mixed, whether they lie before the first attack frame, and the
size of the backward step (median per capture in microseconds, largest per capture in milliseconds).
Output: results/sort_check_all.csv and a one-line summary on stdout.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from canio import load_ctt_csv  # noqa: E402
from experiment import CTT, ROOT  # noqa: E402

m = pd.read_csv(os.path.join(ROOT, "results/manifest_ctt.csv")).drop_duplicates("sha256")
rows = []
for _, r in m.iterrows():
    c = load_ctt_csv(os.path.join(CTT, r.path))
    i = np.flatnonzero(np.diff(c["t"]) < 0)
    if i.size == 0:
        continue
    lab = c["label"]
    a, b = lab[i] == 1, lab[i + 1] == 1
    att = np.flatnonzero(lab == 1)
    pre = i < att[0] if att.size else np.ones(i.size, bool)
    rows.append({"path": r.path, "vehicle": r.vehicle, "kind": r.kind, "reversals": i.size,
                 "both_normal": int((~a & ~b).sum()), "both_attack": int((a & b).sum()), "mixed": int((a ^ b).sum()),
                 "before_first_attack": int(pre.sum()), "median_step_us": float(np.median(-np.diff(c["t"])[i]) * 1e6),
                 "max_step_ms": float((-np.diff(c["t"])[i]).max() * 1000)})
d = pd.DataFrame(rows)
d.to_csv(os.path.join(ROOT, "results/sort_check_all.csv"), index=False)
print(f"captures with reversals {len(d)}; reversals {d.reversals.sum()}; both frames normal {d.both_normal.sum()}; "
      f"both attack {d.both_attack.sum()}; mixed {d.mixed.sum()}; before first attack frame {d.before_first_attack.sum()}; "
      f"largest step {d.max_step_ms.max():.2f} ms")
