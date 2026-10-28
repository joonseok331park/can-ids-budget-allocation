"""Public manifest: every CTT file with path, SHA256, documented vehicle (README), normalised
group (ID fingerprint), duplicate group (common origin by identical SHA256), reuse finding and usage role in this
study. Also writes results/provenance.json with the SHA-256 of the manifest, the feature caches and the run files.
"""
import glob
import hashlib
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = pd.read_csv(os.path.join(ROOT, "results/manifest_ctt.csv"))
sha = pd.read_csv(os.path.join(ROOT, "results/ctt_download_sha256.csv"))[["path", "sha256"]]
m = m.drop(columns=[c for c in ("sha256",) if c in m.columns]).merge(sha, on="path", how="left")
first = m.sort_values("path").drop_duplicates("sha256").set_index("sha256")["path"]
m["duplicate_of"] = [first[s] if first[s] != p else "" for p, s in zip(m.path, m.sha256)]
roles = {}
for v, g in m.drop_duplicates("sha256").groupby("vehicle"):
    af = g[g.kind == "attack-free"].sort_values("duration_s", ascending=False)
    roles[af.path.iloc[0]] = "budget capture (profile, training, calibration)"
    roles[af.path.iloc[1]] = "held-out normal evaluation"
REUSE = {"set_01/train_01/standstill-1.csv": "normal part = budget capture (excluded from evaluation)",
         "set_01/train_01/standstill-2.csv": "normal part = held-out capture (attack evaluation only)"}


def role(r):
    canon = r.duplicate_of or r.path
    if canon in roles:
        return roles[canon]
    if r.kind == "accessory":
        return "not used (accessory)"
    if r.attack_frames == 0:
        return "excluded (no attack label)"
    if canon == "set_01/train_01/standstill-1.csv":
        return "excluded (reuse of budget capture)"
    if canon == "set_01/train_01/standstill-2.csv":
        return "attack evaluation; prefix excluded from normal evaluation"
    return "attack evaluation"


m["role"] = [role(r) for r in m.itertuples()]
m["reuse_finding"] = [REUSE.get(r.duplicate_of or r.path, "") for r in m.itertuples()]
cols = ["path", "sha256", "readme_vehicle", "vehicle", "id_jaccard_to_vehicle", "duplicate_of", "kind",
        "duration_s", "frames", "attack_frames", "reuse_finding", "role"]
out = m[cols].rename(columns={"vehicle": "normalised_vehicle_group"}).sort_values("path")
out.to_csv(os.path.join(ROOT, "results/manifest_public.csv"), index=False)
print(out.role.value_counts().to_string())


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


prov = {"manifest_ctt_sha256": sha_file(os.path.join(ROOT, "results/manifest_ctt.csv")),
        "vehicle_mapping": "ID-set fingerprint; README mapping kept as readme_vehicle",
        "feature_caches": {os.path.relpath(p, ROOT): sha_file(p) for p in sorted(glob.glob(os.path.join(ROOT, "data/features_alloc/*/*.npz")) + glob.glob(os.path.join(ROOT, "data/features_road/*.npz")) + glob.glob(os.path.join(ROOT, "data/features_v2/source/*.npz")))},
        "results": {os.path.relpath(p, ROOT): sha_file(p) for p in sorted(glob.glob(os.path.join(ROOT, "results/runs/v3_*.csv")) + [os.path.join(ROOT, "results/runs/road_v3.csv")])}}
json.dump(prov, open(os.path.join(ROOT, "results/provenance.json"), "w", encoding="utf-8"), indent=1)
print("provenance entries", len(prov["feature_caches"]), len(prov["results"]))
