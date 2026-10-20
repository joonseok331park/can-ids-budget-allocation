"""Feature-group ablation (exploratory): which residual groups carry ranking and which carry the alarm burden?

Leave-one-group-out over the 18 features, r = 0 and r = 1, both profile conditions, B = 30/120, seeds 0-2.
Output: results/runs/v2_ablation.csv
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from models_v2 import run_condition  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GROUPS = {
    "rate_id": ("rate_logratio", "oov_frac", "missing_frac", "id_js", "id_rate_dev"),
    "timing": ("iat_z_mean", "iat_z_gt3", "iat_short_frac", "id_max_iat_z", "id_max_short"),
    "payload_range": ("range_viol_frac", "id_max_range_viol"),
    "payload_change": ("byte_change_dev", "ham_z_mean", "dlc_mismatch_frac", "id_max_change_dev", "id_max_ham_z"),
    "transition": ("new_transition_frac",),
    "id_max_all": ("id_max_range_viol", "id_max_iat_z", "id_max_short", "id_max_change_dev", "id_max_ham_z"),
}

if __name__ == "__main__":
    rows = []
    for t in ("forester", "impala", "silverado", "traverse"):
        for B in (30, 120):
            for p in ("short", "ref"):
                for r in (0, 1):
                    for s in (0, 1, 2):
                        for g, cols in GROUPS.items():
                            rows.append(run_condition(t, B, p, r, s, tag=f"drop_{g}", drop=cols))
        print("done", t, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results/runs/v2_ablation.csv"), index=False)
    print("rows", len(rows))
