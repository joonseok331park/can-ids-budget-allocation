"""Strengthens the two confirmed reuse cases: per informative ID, the share of the attack
capture's pre-attack 16-payload grams found in the reference capture and the longest run of consecutive matched
grams, in seconds of the attack capture. Output: results/reuse_cases.md
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from canio import ensure_sorted, load_ctt_csv  # noqa: E402
from check_reuse_perid import kgrams, payload_hash  # noqa: E402
from experiment import CTT, ROOT, load_roles  # noqa: E402

CASES = [("set_01/train_01/standstill-1.csv", "budget"), ("set_01/train_01/standstill-2.csv", "reference")]
r = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))["impala"]
b = load_ctt_csv(os.path.join(CTT, r["budget"]))
ids, cnt = np.unique(b["aid"], return_counts=True)
cand = ids[cnt >= 1000]
uniq = []
for a in cand:
    g = kgrams(payload_hash(b["data"][b["aid"] == a]))
    _, c = np.unique(g, return_counts=True)
    uniq.append((c == 1).sum() / max(g.size, 1))
top = cand[np.argsort(-np.array(uniq))[:5]]
rows = []
for path, role in CASES:
    ref = load_ctt_csv(os.path.join(CTT, r[role]))
    c = ensure_sorted(load_ctt_csv(os.path.join(CTT, path)))
    n = np.flatnonzero(c["label"] == 1)[0]
    for a in top:
        m = c["aid"][:n] == a
        q = kgrams(payload_hash(c["data"][:n][m]))
        tq = c["t"][:n][m][: q.size]
        refg = kgrams(payload_hash(ref["data"][ref["aid"] == a]))
        hit = np.isin(q, refg)
        # longest run of consecutive matched grams, measured in seconds of the attack capture
        best, start = 0.0, None
        for i, h in enumerate(hit):
            if h and start is None:
                start = i
            if (not h or i == hit.size - 1) and start is not None:
                end = i if h else i - 1
                best = max(best, float(tq[end] - tq[start]))
                start = None
        rows.append({"capture": path.split("/")[-1], "reference": role, "id": hex(int(a)), "grams": int(q.size),
                     "matched_share": round(float(hit.mean()), 4), "longest_matched_run_s": round(best, 1),
                     "prefix_s": round(float(c["t"][n - 1] - c["t"][0]), 1)})
t = pd.DataFrame(rows)
open(os.path.join(ROOT, "results/reuse_cases.md"), "w").write(
    "# Per-ID matches for the two confirmed reuse cases (Impala)\n\n" + t.to_markdown(index=False) + "\n")
print(t.to_string(index=False))
