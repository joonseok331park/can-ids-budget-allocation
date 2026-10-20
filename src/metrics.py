"""Window- and event-level evaluation.

Windows have start/end times. An alarm event is a run of consecutive alarm windows that overlap or touch;
false-alarm burden is reported as alarm events per hour of normal traffic. An attack segment counts as
detected when any alarm window overlaps it; detection delay is the end time of the first such window
minus the segment start (a window can only raise an alarm once it has ended).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    return float(average_precision_score(labels, scores))


def alarm_events(t_start: np.ndarray, t_end: np.ndarray, alarm: np.ndarray) -> list[tuple[float, float]]:
    """Merge alarm windows that overlap or touch into events (assumes windows sorted by start)."""
    events: list[list[float]] = []
    for s, e in zip(t_start[alarm], t_end[alarm]):
        if events and s <= events[-1][1] + 1e-9:
            events[-1][1] = max(events[-1][1], e)
        else:
            events.append([s, e])
    return [(a, b) for a, b in events]


def events_per_hour(t_start, t_end, alarm) -> tuple[int, float, float]:
    """Returns (n_events, hours_observed, events_per_hour) for a stretch of normal windows."""
    if t_start.size == 0:
        return 0, 0.0, float("nan")
    hours = float(t_end[-1] - t_start[0]) / 3600.0
    n = len(alarm_events(t_start, t_end, alarm))
    return n, hours, n / hours if hours > 0 else float("nan")


def segment_detection(t_start, t_end, alarm, segments, tol: float = 0.0):
    """For each attack segment (a, b): detected flag and delay (s). tol extends the segment end."""
    out = []
    for a, b in segments:
        hit = alarm & (t_start < b + tol) & (t_end > a)
        if hit.any():
            first_end = t_end[np.flatnonzero(hit)[0]]
            out.append((True, max(0.0, float(first_end - a))))
        else:
            out.append((False, float("nan")))
    return out


def window_rates(labels, alarm) -> dict:
    labels = np.asarray(labels).astype(bool)
    alarm = np.asarray(alarm).astype(bool)
    tp = int((labels & alarm).sum()); fp = int((~labels & alarm).sum())
    fn = int((labels & ~alarm).sum()); tn = int((~labels & ~alarm).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "tpr": tp / (tp + fn) if tp + fn else float("nan"),
            "fpr": fp / (fp + tn) if fp + tn else float("nan")}
