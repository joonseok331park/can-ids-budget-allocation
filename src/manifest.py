"""Build a per-file manifest for can-train-and-test (CTT) and ROAD.

Outputs (results/):
  manifest_ctt.csv   one row per CTT file: vehicle, kind, duration, frames, IDs, attack segments, duplicate group
  vehicle_fingerprint.csv  README vehicle and ID-set fingerprint per CTT file
  manifest_road.csv  one row per ROAD log: duration, frames, IDs, injection interval, masquerade flag
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from canio import load_ctt_csv, load_road_log, road_metadata  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTT = os.path.join(ROOT, "data/raw/ctt")
ROAD = os.path.join(ROOT, "data/raw/road_zenodo/road")

KNOWN = {"set_01": "impala", "set_02": "traverse", "set_03": "silverado", "set_04": "forester"}
UNKNOWN = {"set_01": "silverado", "set_02": "forester", "set_03": "forester", "set_04": "traverse"}


def vehicle_of(set_name: str, subset: str) -> str:
    if subset.startswith("train_01") or subset.startswith("test_01") or subset.startswith("test_03"):
        return KNOWN[set_name]
    return UNKNOWN[set_name]


def attack_segments(t: np.ndarray, label: np.ndarray, merge_gap: float = 1.0):
    """Contiguous attack periods from frame labels; gaps shorter than merge_gap seconds are merged."""
    idx = np.flatnonzero(label == 1)
    if idx.size == 0:
        return []
    ta = t[idx]
    breaks = np.flatnonzero(np.diff(ta) > merge_gap)
    starts = np.r_[ta[0], ta[breaks + 1]]
    ends = np.r_[ta[breaks], ta[-1]]
    return list(zip(starts, ends))


def id_fingerprint(m: pd.DataFrame, normal_ids: dict) -> tuple[list, list]:
    """Vehicle label from the IDs themselves rather than from the README.

    Reference per vehicle: union of the normal-frame IDs of its attack-free captures (README label). Each file gets
    the vehicle whose reference set has the highest Jaccard similarity with the file's own normal-frame IDs. The
    12 files of set_01/test_04 are documented as Silverado but match Traverse (Jaccard 1.0).
    """
    ref = {}
    for path, v in m.loc[m.kind == "attack-free", ["path", "readme_vehicle"]].itertuples(index=False):
        ref.setdefault(v, set()).update(normal_ids[path])
    best, jac = [], []
    for path in m.path:
        s = normal_ids[path]
        j = {v: len(s & r) / len(s | r) for v, r in ref.items()}
        b = max(j, key=j.get)
        best.append(b)
        jac.append(round(j[b], 3))
    return best, jac


def ctt_manifest() -> pd.DataFrame:
    sha = pd.read_csv(os.path.join(ROOT, "results/ctt_download_sha256.csv"))
    sha = sha[sha.ok == True]  # noqa: E712
    rows, normal_ids = [], {}
    for _, r in sha.iterrows():
        set_name, subset, fname = r.path.split("/")
        d = load_ctt_csv(os.path.join(CTT, r.path))
        t, aid, lab = d["t"], d["aid"], d["label"]
        normal_ids[r.path] = set(np.unique(aid[lab == 0]).tolist())
        dur = float(t[-1] - t[0]) if t.size else 0.0
        segs = attack_segments(t, lab)
        stem = fname[:-4]
        kind = stem.rsplit("-", 1)[0]
        rows.append({
            "path": r.path, "set": set_name, "subset": subset, "file": fname, "kind": kind,
            "vehicle": vehicle_of(set_name, subset), "sha256": r.sha256,
            "frames": int(t.size), "duration_s": round(dur, 3), "rate_fps": round(t.size / dur, 1) if dur > 0 else np.nan,
            "n_ids": int(np.unique(aid).size),
            "attack_frames": int((lab == 1).sum()), "attack_frac": round(float((lab == 1).mean()), 5) if t.size else np.nan,
            "n_attack_segments": len(segs),
            "first_attack_s": round(float(segs[0][0] - t[0]), 3) if segs else np.nan,
            "last_attack_end_s": round(float(segs[-1][1] - t[0]), 3) if segs else np.nan,
            "normal_prefix_s": round(float(segs[0][0] - t[0]), 3) if segs else round(dur, 3),
            "normal_suffix_s": round(float(t[-1] - segs[-1][1]), 3) if segs else 0.0,
            "t_monotonic": bool(np.all(np.diff(t) >= 0)),
            "dlc_gt8": int((d["dlc"] > 8).sum()),
        })
    m = pd.DataFrame(rows)
    grp = m.groupby("sha256").path.transform(lambda s: "|".join(sorted(s)) if len(s) > 1 else "")
    m["duplicate_of"] = grp
    m["dup_group_size"] = m.groupby("sha256").path.transform("count")
    m["readme_vehicle"] = m["vehicle"]
    m["vehicle"], m["id_jaccard_to_vehicle"] = id_fingerprint(m, normal_ids)
    m = m[[c for c in m.columns if c != "readme_vehicle"] + ["readme_vehicle"]]
    return m.sort_values(["vehicle", "kind", "file", "set", "subset"])


def road_manifest() -> pd.DataFrame:
    meta = road_metadata(ROAD)
    rows = []
    for name, v in meta.items():
        p = os.path.join(ROAD, v["folder"], name + ".log")
        if not os.path.exists(p):
            continue
        d = load_road_log(p)
        t, aid = d["t"], d["aid"]
        dur = float(t[-1] - t[0]) if t.size else 0.0
        inj = v.get("injection_interval")
        rows.append({
            "capture": name, "folder": v["folder"], "on_dyno": v.get("on_dyno"),
            "elapsed_sec_meta": v.get("elapsed_sec"), "duration_s": round(dur, 3), "frames": int(t.size),
            "rate_fps": round(t.size / dur, 1) if dur > 0 else np.nan, "n_ids": int(np.unique(aid).size),
            "fff_frames": int((aid == 0xFFF).sum()),
            "masquerade_file": name.endswith("_masquerade"),
            "message_confliction_present": v.get("message_confliction_present"),
            "injection_id": v.get("injection_id"), "injection_data_str": v.get("injection_data_str"), "modified": v.get("modified"),
            "injection_start_s": inj[0] if inj else np.nan, "injection_end_s": inj[1] if inj else np.nan,
            "bad_lines": int(d["n_bad_lines"]),
            "t_monotonic": bool(np.all(np.diff(t) >= 0)),
        })
    return pd.DataFrame(rows).sort_values(["folder", "capture"])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("ctt", "both"):
        m = ctt_manifest()
        m.to_csv(os.path.join(ROOT, "results/manifest_ctt.csv"), index=False)
        f = m.sort_values(["readme_vehicle", "kind", "file", "set", "subset"])
        f = f[["path", "set", "subset", "readme_vehicle", "vehicle", "id_jaccard_to_vehicle", "n_ids"]]
        f.columns = ["path", "set", "subset", "assigned", "fingerprint", "jaccard", "n_ids"]
        f.to_csv(os.path.join(ROOT, "results/vehicle_fingerprint.csv"), index=False)
        print("ctt rows", len(m))
    if which in ("road", "both"):
        r = road_manifest()
        r.to_csv(os.path.join(ROOT, "results/manifest_road.csv"), index=False)
        print("road rows", len(r))
