"""Data and segment table (per target vehicle), written to results/table_data_segments.md."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from models_v2 import EXCLUDE_EVAL, EXCLUDE_PREFIX, FV2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = pd.read_csv(os.path.join(ROOT, "results/manifest_ctt.csv")).drop_duplicates("sha256").set_index("path")
rows = []
for v in ("forester", "impala", "silverado", "traverse"):
    meta = json.load(open(os.path.join(FV2, v, "meta.json")))
    d = np.load(os.path.join(FV2, v, "B120_short.npz"))
    files = [a["file"] for a in meta["attack_info"]]
    keep = [i for i, f in enumerate(files) if f not in EXCLUDE_EVAL.get(v, set())]
    mk = np.isin(d["att_file"], keep)
    lab, ts, te, fid = d["att_label"][mk], d["att_ts"][mk], d["att_te"][mk], d["att_file"][mk]
    pre_s = 0.0
    for i in keep:
        if files[i] in EXCLUDE_PREFIX.get(v, set()):
            continue
        sel = (fid == i) & (te <= meta["attack_info"][i]["first_attack"])
        if sel.sum() >= 2:
            pre_s += te[sel][-1] - ts[sel][0]
    rows.append({
        "vehicle": v,
        "budget capture (s)": f'{meta["budget_capture"].split("/")[-1]} ({m.loc[meta["budget_capture"], "duration_s"]:.0f})',
        "held-out capture (s)": f'{meta["heldout_capture"].split("/")[-1]} ({m.loc[meta["heldout_capture"], "duration_s"]:.0f})',
        "held-out windows": len(d["Xheld"]),
        "attack captures": len(keep),
        "attack kinds": len({meta["attack_info"][i]["kind"] for i in keep}),
        "attack-capture windows": int(mk.sum()),
        "attack-positive share": round(float(lab.mean()), 3),
        "prefix normal (h)": round(pre_s / 3600, 2),
    })
t = pd.DataFrame(rows)
seg = pd.DataFrame([
    {"B (s)": B, "short P": f"[{250 - B // 2}, 250)", "reference P": "[0, 250)", "F": f"[250, {250 + 0.3 * B:g})",
     "C": f"[{250 + 0.3 * B:g}, {250 + 0.5 * B:g})", "F windows": int((0.3 * B - 1) / 0.5 + 1),
     "C windows": int((0.2 * B - 1) / 0.5 + 1)} for B in (30, 120)])
out = ["# Data and segments", "",
       "Times are seconds from the first frame of the budget capture. Windows: 1 s long, 0.5 s stride.", "",
       seg.to_markdown(index=False), "", t.to_markdown(index=False), "",
       "Impala: standstill-1 is excluded from all evaluation and standstill-2 from the prefix normal evaluation.",
       "Traverse: force-neutral-2 of set_04/test_04 has no attack label and is excluded."]
open(os.path.join(ROOT, "results/table_data_segments.md"), "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
