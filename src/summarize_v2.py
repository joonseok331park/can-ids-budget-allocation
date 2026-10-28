"""Tables for the exploratory v2 grid and contrasts (per vehicle first; seeds are not independent drives).

Writes results/v2_summary.md with:
  1. AP by vehicle x B x profile x r (mean +- sd over seeds)
  2. Delta(P) = AP(r=1) - AP(r=0) paired by seed, and I = Delta(ref) - Delta(short)
  3. Operating results (alarm burden on held-out capture and on attack-capture prefixes, capture detection)
  4. Contrasts (r=0.25, IF max_samples rule, matched source P) next to the matching main condition
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}" if np.isfinite(s) else f"{m:.3f}"


def main(main_csv, contrast_csv=None, out_md=f"{ROOT}/results/v2_summary.md"):
    d = pd.read_csv(main_csv)
    lines = ["# v2 summary (per vehicle, mean ± SD over 3 seeds)", "",
             "Seeds only change the IF randomness on the same data; they are not independent drives.", ""]
    # 1. AP table
    g = d.groupby(["target", "B", "profile", "r"]).AP.agg(["mean", "std"]).reset_index()
    g["AP"] = [fmt(a, b) for a, b in zip(g["mean"], g["std"])]
    t1 = g.pivot_table(index=["target", "B", "profile"], columns="r", values="AP", aggfunc="first")
    t1.columns = [f"r={c:g}" for c in t1.columns]
    lines += ["## 1. AP (all windows of the attack captures)", "", t1.to_markdown(), ""]
    # 2. Delta and interaction, paired by seed
    p = d.pivot_table(index=["target", "B", "profile", "seed"], columns="r", values="AP").reset_index()
    p["delta"] = p[1] - p[0]
    dd = p.pivot_table(index=["target", "B", "seed"], columns="profile", values="delta").reset_index()
    dd["I"] = dd["ref"] - dd["short"]
    agg = dd.groupby(["target", "B"]).agg(delta_short=("short", "mean"), delta_short_sd=("short", "std"),
                                          delta_ref=("ref", "mean"), delta_ref_sd=("ref", "std"),
                                          I=("I", "mean"), I_sd=("I", "std")).reset_index()
    agg["Δ(short)"] = [fmt(a, b) for a, b in zip(agg.delta_short, agg.delta_short_sd)]
    agg["Δ(ref)"] = [fmt(a, b) for a, b in zip(agg.delta_ref, agg.delta_ref_sd)]
    agg["I = Δ(ref) − Δ(short)"] = [fmt(a, b) for a, b in zip(agg.I, agg.I_sd)]
    lines += ["## 2. Source effect Δ(P) = AP(r=1) − AP(r=0) and its difference I between profile conditions", "",
              agg[["target", "B", "Δ(short)", "Δ(ref)", "I = Δ(ref) − Δ(short)"]].to_markdown(index=False), ""]
    # 3. operating results
    ops = d.groupby(["target", "B", "profile", "r"]).agg(
        held_h=("held_hours", "first"), held_fa_h=("held_fa_per_h", "mean"), held_pos=("held_pos_frac", "mean"),
        prefix_h=("prefix_hours", "first"), prefix_fa_h=("prefix_fa_per_h", "mean"), prefix_pos=("prefix_pos_frac", "mean"),
        detect=("capture_detect_rate", "mean"), pre_on=("n_pre_on", "mean"), delay=("median_delay_s", "median"),
        n_cal=("n_cal", "first")).reset_index().round(3)
    lines += ["## 3. Operating results (threshold = max calibration score, alarm = score > threshold)", "",
              "held = other attack-free captures kept out entirely, prefix = normal part of an attack capture before its first attack. "
              "fa_h = alarm events per hour, pos = share of positive windows, detect = share of attack captures detected, pre_on = captures detected by an alarm already on before the attack (mean).", "",
              ops.to_markdown(index=False), ""]
    if contrast_csv:
        c = pd.read_csv(contrast_csv)
        cg = c.groupby(["target", "tag", "B", "profile", "r", "src_P", "max_samples_rule"]).agg(
            AP=("AP", "mean"), AP_sd=("AP", "std"), held_pos=("held_pos_frac", "mean"),
            prefix_pos=("prefix_pos_frac", "mean"), detect=("capture_detect_rate", "mean")).reset_index().round(3)
        lines += ["## 4. Small contrasts", "", cg.to_markdown(index=False), ""]
    open(out_md, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
