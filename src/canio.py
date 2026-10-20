"""Loading CAN captures into numpy arrays.

Every loader returns a dict with
    t     float64 [N]  timestamp in seconds (as stored in the file)
    aid   int32   [N]  arbitration ID
    dlc   int8    [N]  data length (bytes)
    data  uint8   [N,8] payload, zero padded to 8 bytes
    label int8    [N]  1 = attack frame, 0 = attack-free, -1 = unknown (ROAD, labels come from metadata windows)
Parsed arrays are cached as .npz files in data/cache/ so each raw file is parsed once.
"""
from __future__ import annotations
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE = ROOT + "/data/cache"


def _cache_path(src: str) -> str:
    h = hashlib.sha1(os.path.abspath(src).encode()).hexdigest()[:16]
    return os.path.join(CACHE, h + ".npz")


def _hex_payload_to_array(hex_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    s = hex_series.fillna("").astype(str).str.strip()
    dlc = (s.str.len() // 2).to_numpy().astype(np.int8)
    if (dlc > 8).any():
        raise ValueError("payload longer than 8 bytes found")
    padded = s.str.pad(16, side="right", fillchar="0")
    buf = bytes.fromhex("".join(padded.tolist()))
    data = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 8).copy()
    return dlc, data


def load_ctt_csv(path: str, use_cache: bool = True) -> dict:
    """can-train-and-test CSV: timestamp,arbitration_id,data_field,attack"""
    cp = _cache_path(path)
    if use_cache and os.path.exists(cp):
        z = np.load(cp)
        return {k: z[k] for k in z.files}
    df = pd.read_csv(path, dtype={"timestamp": np.float64, "arbitration_id": str, "data_field": str, "attack": np.int8})
    t = df["timestamp"].to_numpy(np.float64)
    aid = np.array([int(x, 16) for x in df["arbitration_id"].tolist()], dtype=np.int32)
    dlc, data = _hex_payload_to_array(df["data_field"])
    label = df["attack"].to_numpy(np.int8)
    out = {"t": t, "aid": aid, "dlc": dlc, "data": data, "label": label}
    os.makedirs(CACHE, exist_ok=True)
    np.savez(cp, **out)
    return out


_ROAD_RE = re.compile(r"^\((?P<t>[0-9.]+)\)\s+\S+\s+(?P<aid>[0-9A-Fa-f]+)#(?P<data>[0-9A-Fa-f]*)\s*$")


def load_road_log(path: str, use_cache: bool = True) -> dict:
    """ROAD candump-style log: (timestamp) can0 AID#DATA. Frame labels are -1; use metadata for windows."""
    cp = _cache_path(path)
    if use_cache and os.path.exists(cp):
        z = np.load(cp)
        return {k: z[k] for k in z.files}
    ts, aids, datas = [], [], []
    bad = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = _ROAD_RE.match(line)
            if not m:
                bad += 1
                continue
            ts.append(float(m.group("t")))
            aids.append(int(m.group("aid"), 16))
            datas.append(m.group("data"))
    dlc, data = _hex_payload_to_array(pd.Series(datas))
    out = {
        "t": np.asarray(ts, dtype=np.float64),
        "aid": np.asarray(aids, dtype=np.int32),
        "dlc": dlc,
        "data": data,
        "label": np.full(len(ts), -1, dtype=np.int8),
        "n_bad_lines": np.int64(bad),
    }
    os.makedirs(CACHE, exist_ok=True)
    np.savez(cp, **out)
    return out


def road_metadata(road_root: str) -> dict:
    meta = {}
    for sub in ("ambient", "attacks"):
        with open(os.path.join(road_root, sub, "capture_metadata.json")) as f:
            for k, v in json.load(f).items():
                v = dict(v)
                v["folder"] = sub
                meta[k] = v
    return meta


def ensure_sorted(cap: dict) -> dict:
    """Stable sort by timestamp. Some CTT attack captures have tiny backward steps (median ~20 us, max ~8 ms),
    mostly at attack frames; windowing and per-ID inter-arrival times assume time order."""
    t = cap["t"]
    if t.size < 2 or np.all(np.diff(t) >= 0):
        return cap
    o = np.argsort(t, kind="stable")
    return {k: (v[o] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == t.shape[0] else v) for k, v in cap.items()}
