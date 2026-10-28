"""Check: effect of stable time-sorting on features, on the CTT file with the most timestamp reversals
(set_02/train_01/interval-1.csv, Traverse). Profile: Traverse budget capture [0, 250) s."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from canio import ensure_sorted, load_ctt_csv
from experiment import CTT, ROOT, load_roles
from features import FEATURES, build_profile, slice_time, window_features
roles = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))
b = load_ctt_csv(os.path.join(CTT, roles["traverse"]["budget"]))
prof = build_profile(slice_time(b, b["t"][0], b["t"][0] + 250))
raw = load_ctt_csv(os.path.join(CTT, "set_02/train_01/interval-1.csv"))
srt = ensure_sorted(raw)
neg = np.flatnonzero(np.diff(raw["t"]) < 0)
t0, t1 = raw["t"].min(), raw["t"].max()
wr = window_features(raw, prof, 1.0, 0.5, t_origin=t0, t_stop=t1)
ws = window_features(srt, prof, 1.0, 0.5, t_origin=t0, t_stop=t1)
D = np.abs(wr["X"] - ws["X"])
changed = D.max(1) > 1e-9
lab = ws["label"]
out = [f"file set_02/train_01/interval-1.csv: frames {raw['t'].size}, reversals {neg.size}, windows {len(lab)}",
       f"windows whose features change after sorting: {int(changed.sum())} ({changed.mean():.3%})",
       f"  of those, attack-positive windows: {int((changed & (lab==1)).sum())} / positive windows total {int(lab.sum())}",
       f"  changed among negative windows: {int((changed & (lab==0)).sum())} / {int((lab==0).sum())}"]
per = pd.DataFrame({"feature": FEATURES, "max_abs_diff": D.max(0), "mean_abs_diff_changed": D[changed].mean(0) if changed.any() else 0})
out.append(per.round(4).to_string(index=False))
open(os.path.join(ROOT, "results/sort_check.txt"), "w", encoding="utf-8").write("\n".join(out)); print("\n".join(out))
