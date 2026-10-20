"""Vehicle-relative ("residual") window features for CAN traffic.

A Profile summarises one vehicle's normal traffic. Window features measure how a stretch of traffic deviates
from a profile, so features from different vehicles live on the same scale and can be pooled for training.
All per-frame context (previous frame of the same ID, previous ID) is computed inside the array passed in,
so slicing a capture to a budget never leaks frames from outside the budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FEATURES = [
    "rate_logratio",      # log((n+1)/(expected+1)) total frames in window vs profile
    "oov_frac",           # frames whose ID is not in the profile vocabulary
    "missing_frac",       # profile IDs expected >=2 times in the window but absent
    "id_js",              # Jensen-Shannon divergence of window ID histogram vs profile
    "id_rate_dev",        # mean |log ratio| of per-ID counts vs expected
    "iat_z_mean",         # mean robust z of inter-arrival time (known IDs)
    "iat_z_gt3",          # fraction of known-ID frames with IAT z > 3
    "iat_short_frac",     # fraction of known-ID frames arriving < 0.5 x typical period
    "range_viol_frac",    # fraction of known-ID frames with a byte outside the profile range
    "byte_change_dev",    # mean |byte changed - profile change probability|
    "ham_z_mean",         # mean z of Hamming distance to previous payload of same ID
    "dlc_mismatch_frac",  # DLC differs from the profile's usual DLC for that ID
    "new_transition_frac",  # (previous ID, ID) pairs never seen in the profile
    # per-ID maxima (an attack on one ID is diluted in window-wide averages)
    "id_max_range_viol",  # max over IDs (>=3 frames in window) of range-violation fraction
    "id_max_iat_z",       # max over IDs of mean IAT z
    "id_max_short",       # max over IDs of short-IAT fraction
    "id_max_change_dev",  # max over IDs of mean byte-change deviation
    "id_max_ham_z",       # max over IDs of mean Hamming z
]


@dataclass
class Profile:
    vocab: np.ndarray                 # sorted known IDs
    rate: np.ndarray                  # frames/s per ID
    total_rate: float
    iat_med: np.ndarray
    iat_scale: np.ndarray
    dlc_mode: np.ndarray
    byte_min: np.ndarray              # [V,8]
    byte_max: np.ndarray              # [V,8]
    change_p: np.ndarray              # [V,8]
    ham_mean: np.ndarray
    ham_std: np.ndarray
    bigrams: set = field(default_factory=set)
    duration: float = 0.0
    n_frames: int = 0


def _prev_same_id(aid: np.ndarray) -> np.ndarray:
    """Index of the previous frame with the same ID (-1 if none)."""
    order = np.lexsort((np.arange(aid.size), aid))
    prev = np.full(aid.size, -1, dtype=np.int64)
    same = aid[order[1:]] == aid[order[:-1]]
    prev[order[1:][same]] = order[:-1][same]
    return prev


def _popcount8(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.astype(np.uint8)[..., None], axis=-1).sum(-1)


def _per_frame(cap: dict):
    aid, t, data, dlc = cap["aid"], cap["t"], cap["data"], cap["dlc"]
    prev = _prev_same_id(aid)
    has_prev = prev >= 0
    iat = np.full(aid.size, np.nan)
    iat[has_prev] = t[has_prev] - t[prev[has_prev]]
    changed = np.zeros((aid.size, 8), dtype=bool)
    changed[has_prev] = data[has_prev] != data[prev[has_prev]]
    ham = np.full(aid.size, np.nan)
    ham[has_prev] = _popcount8(data[has_prev] ^ data[prev[has_prev]]).sum(1)
    return prev, has_prev, iat, changed, ham


def build_profile(cap: dict) -> Profile:
    aid, t, data, dlc = cap["aid"], cap["t"], cap["data"], cap["dlc"]
    dur = float(t[-1] - t[0]) if t.size > 1 else 1e-9
    prev, has_prev, iat, changed, ham = _per_frame(cap)
    vocab, inv, counts = np.unique(aid, return_inverse=True, return_counts=True)
    V = vocab.size
    iat_med = np.zeros(V); iat_scale = np.zeros(V); ham_mean = np.zeros(V); ham_std = np.zeros(V)
    dlc_mode = np.zeros(V, dtype=np.int8)
    bmin = np.zeros((V, 8), dtype=np.uint8); bmax = np.zeros((V, 8), dtype=np.uint8)
    chp = np.zeros((V, 8))
    order = np.argsort(inv, kind="stable")
    bounds = np.r_[0, np.cumsum(counts)]
    for k in range(V):
        idx = order[bounds[k]:bounds[k + 1]]
        it = iat[idx]; it = it[np.isfinite(it)]
        if it.size:
            med = np.median(it)
            mad = np.median(np.abs(it - med)) * 1.4826
            iat_med[k] = med
            iat_scale[k] = max(mad, 0.05 * med, 1e-4)
        else:  # seen once: no period information
            iat_med[k] = dur
            iat_scale[k] = max(dur, 1e-4)
        hm = ham[idx]; hm = hm[np.isfinite(hm)]
        ham_mean[k] = hm.mean() if hm.size else 0.0
        ham_std[k] = hm.std() if hm.size else 0.0
        vals, cnt = np.unique(dlc[idx], return_counts=True)
        dlc_mode[k] = vals[np.argmax(cnt)]
        bmin[k] = data[idx].min(0); bmax[k] = data[idx].max(0)
        hp = has_prev[idx]
        chp[k] = changed[idx][hp].mean(0) if hp.any() else 0.0
    bigrams = set(zip(aid[:-1].tolist(), aid[1:].tolist()))
    return Profile(vocab=vocab, rate=counts / dur, total_rate=aid.size / dur, iat_med=iat_med, iat_scale=iat_scale,
                   dlc_mode=dlc_mode, byte_min=bmin, byte_max=bmax, change_p=chp, ham_mean=ham_mean, ham_std=ham_std,
                   bigrams=bigrams, duration=dur, n_frames=int(aid.size))


def window_features(cap: dict, prof: Profile, width: float = 1.0, stride: float = 0.5,
                    attack_intervals: list | None = None, t_origin: float | None = None,
                    t_stop: float | None = None) -> dict:
    """Features for sliding windows over cap. Returns dict(X [W,F], t_start, t_end, n_frames, label).

    Labels: if attack_intervals (absolute times) are given, a window is attack when it overlaps one;
    otherwise a window is attack when it contains a frame with label == 1.
    Windows are [t, t + width) starting at t_origin (default: first frame time); windows ending after t_stop
    (default: last frame time) are dropped so every window lies inside the segment.
    """
    aid, t, data, dlc = cap["aid"], cap["t"], cap["data"], cap["dlc"]
    n = aid.size
    if n == 0:
        return {"X": np.zeros((0, len(FEATURES))), "t_start": np.zeros(0), "t_end": np.zeros(0),
                "n_frames": np.zeros(0, int), "label": np.zeros(0, np.int8)}
    k_slots = int(round(width / stride))
    assert abs(k_slots * stride - width) < 1e-9, "width must be a multiple of stride"
    t0 = t[0] if t_origin is None else t_origin
    slot = np.floor((t - t0) / stride).astype(np.int64)
    n_slots = int(slot[-1]) + 1
    n_win = max(n_slots - k_slots + 1, 0)
    if n_win == 0:
        return {"X": np.zeros((0, len(FEATURES))), "t_start": np.zeros(0), "t_end": np.zeros(0),
                "n_frames": np.zeros(0, int), "label": np.zeros(0, np.int8)}

    V = prof.vocab.size
    pos = np.searchsorted(prof.vocab, aid)
    pos_c = np.clip(pos, 0, V - 1)
    known = (pos < V) & (prof.vocab[pos_c] == aid)
    kidx = np.where(known, pos_c, -1)

    prev, has_prev, iat, changed, ham = _per_frame(cap)
    kp = known & has_prev
    iat_z = np.zeros(n); short = np.zeros(n); ham_z = np.zeros(n); chdev = np.zeros(n)
    k = kidx[kp]
    iat_z[kp] = np.minimum(np.abs(iat[kp] - prof.iat_med[k]) / prof.iat_scale[k], 50.0)
    short[kp] = iat[kp] < 0.5 * prof.iat_med[k]
    ham_z[kp] = np.minimum(np.abs(ham[kp] - prof.ham_mean[k]) / (prof.ham_std[k] + 0.5), 50.0)
    chdev[kp] = np.abs(changed[kp].astype(float) - prof.change_p[k]).mean(1)
    rv = np.zeros(n)
    kk = kidx[known]
    nb = np.arange(8)[None, :] < dlc[known][:, None]
    viol = ((data[known] < prof.byte_min[kk]) | (data[known] > prof.byte_max[kk])) & nb
    rv[known] = viol.any(1)
    dm = np.zeros(n)
    dm[known] = dlc[known] != prof.dlc_mode[kk]
    newtr = np.zeros(n)
    if n > 1:
        pairs = list(zip(aid[:-1].tolist(), aid[1:].tolist()))
        newtr[1:] = [p not in prof.bigrams for p in pairs]

    def slot_sum(v):
        return np.bincount(slot, weights=v, minlength=n_slots)

    def win(v_slot):
        c = np.r_[0.0, np.cumsum(v_slot)]
        return c[k_slots:k_slots + n_win] - c[:n_win]

    cnt_all = win(slot_sum(np.ones(n)))
    cnt_known = win(slot_sum(known.astype(float)))
    cnt_kp = win(slot_sum(kp.astype(float)))
    oov = cnt_all - cnt_known
    safe = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)  # noqa: E731

    # per-ID counts per window: [n_win, V]
    idc = np.zeros((n_slots, V + 1))
    np.add.at(idc, (slot, np.where(known, kidx, V)), 1.0)
    cum = np.vstack([np.zeros((1, V + 1)), np.cumsum(idc, 0)])
    wid = cum[k_slots:k_slots + n_win] - cum[:n_win]
    exp_id = prof.rate * width
    exp_tot = prof.total_rate * width
    rate_logratio = np.log((cnt_all + 1.0) / (exp_tot + 1.0))
    need = exp_id >= 2
    missing_frac = (((wid[:, :V] == 0) & need[None, :]).sum(1) / max(need.sum(), 1))
    p_ref = np.r_[prof.rate / prof.rate.sum(), 0.0]
    p_win = wid / np.maximum(wid.sum(1, keepdims=True), 1)
    m = 0.5 * (p_win + p_ref[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        kl1 = np.where(p_win > 0, p_win * np.log(p_win / m), 0).sum(1)
        kl2 = np.where(p_ref[None, :] > 0, p_ref[None, :] * np.log(p_ref[None, :] / m), 0).sum(1)
    id_js = 0.5 * (kl1 + kl2)
    use = exp_id >= 1
    id_rate_dev = np.abs(np.log((wid[:, :V][:, use] + 0.5) / (exp_id[use] + 0.5))).mean(1) if use.any() else np.zeros(n_win)

    def per_id(v, mask):
        a = np.zeros((n_slots, V))
        sel = mask & known
        np.add.at(a, (slot[sel], kidx[sel]), v[sel])
        c = np.vstack([np.zeros((1, V)), np.cumsum(a, 0)])
        return c[k_slots:k_slots + n_win] - c[:n_win]

    def id_max(v, mask):
        num = per_id(v, mask)
        den = per_id(np.ones(n), mask)
        r = np.divide(num, den, out=np.zeros_like(num), where=den >= 3)
        return r.max(1)

    X = np.column_stack([
        rate_logratio,
        safe(oov, cnt_all),
        missing_frac,
        id_js,
        id_rate_dev,
        safe(win(slot_sum(iat_z)), cnt_kp),
        safe(win(slot_sum((iat_z > 3).astype(float) * kp)), cnt_kp),
        safe(win(slot_sum(short)), cnt_kp),
        safe(win(slot_sum(rv)), cnt_known),
        safe(win(slot_sum(chdev)), cnt_kp),
        safe(win(slot_sum(ham_z)), cnt_kp),
        safe(win(slot_sum(dm)), cnt_known),
        safe(win(slot_sum(newtr)), cnt_all),
        id_max(rv, known),
        id_max(iat_z, kp),
        id_max(short, kp),
        id_max(chdev, kp),
        id_max(ham_z, kp),
    ])
    t_start = t0 + np.arange(n_win) * stride
    t_end = t_start + width
    stop = t[-1] if t_stop is None else t_stop
    inside = t_end <= stop + 1e-9
    if attack_intervals is not None:
        lab = np.zeros(n_win, np.int8)
        for a, b in attack_intervals:
            lab |= ((t_start < b) & (t_end > a)).astype(np.int8)
    else:
        att = win(slot_sum((cap["label"] == 1).astype(float)))
        lab = (att > 0).astype(np.int8)
    return {"X": X[inside], "t_start": t_start[inside], "t_end": t_end[inside],
            "n_frames": cnt_all.astype(int)[inside], "label": lab[inside]}


def slice_time(cap: dict, start: float, end: float) -> dict:
    """Frames with start <= t < end (absolute timestamps)."""
    lo, hi = np.searchsorted(cap["t"], [start, end], side="left")
    return {k: (v[lo:hi] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == cap["t"].shape[0] else v)
            for k, v in cap.items()}


def drop_ids(cap: dict, ids) -> dict:
    keep = ~np.isin(cap["aid"], np.asarray(list(ids)))
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == keep.shape[0] else v)
            for k, v in cap.items()}
