"""Final tables: no new training experiments, only re-aggregation of stored runs plus
deterministic re-scoring of the same forests for per-capture new-alarm sets.

Writes results/final_tables.md with
  1. per budget: all 8 vehicle x position cells for each condition (AP, held-out alarm share, alarm on at attack start,
     attack new-alarm rate within 5 s, normal-prefix chance rate within 5 s, their difference E5) and the planned
     paired differences per cell
  2. position effect vs method effect on the same scale (B = 120)
  3. sample-size controls per position, and how positions were averaged
  4. new-alarm gains and losses with totals for S_sameC -> S_allC (model fixed) and T_main -> T_split (model and C
     change), all seeds, B = 120
  5. denominators: attack captures per vehicle, held-out normal hours, prefix normal hours, virtual start counts
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import alloc_v3 as A  # noqa: E402
from alloc_models import eval_data, seg_windows  # noqa: E402
from eval_v3 import onsets  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["T_main", "S_sameC", "S_allC", "T_split", "U_allC"]
PAIRS = [("S_sameC", "T_main"), ("S_allC", "S_sameC"), ("S_allC", "T_split"), ("T_split", "T_main"), ("S_allC", "U_allC")]
MET = [("AP", "AP"), ("held_pos_frac", "held-out alarm share"), ("on_at_start_frac", "alarm on at attack start"),
       ("new_within_5s", "R_attack,5 (new alarm within 5 s)"), ("prefix_virtual_new_5s", "R_normal,5 (prefix virtual starts)"),
       ("E5", "E5 = R_attack,5 − R_normal,5")]


def cell_tables(d, B):
    x = d[d.B == B].copy()
    x["E5"] = x.new_within_5s - x.prefix_virtual_new_5s
    x["cell"] = x.target + " u=" + x.u.map(lambda v: f"{v:g}")
    out = [f"## B = {B:g} s: all cells (seed means) and seed-paired differences", ""]
    for m, name in MET:
        p = x.pivot_table(index="cell", columns="cond", values=m, aggfunc="mean")[ORDER]
        for a, b in PAIRS:
            q = x.pivot_table(index=["cell", "seed"], columns="cond", values=m)
            p[f"{a}−{b}"] = (q[a] - q[b]).groupby("cell").mean()
        out += [f"### {name}", "", p.round(3).to_markdown(), "",
                "8-cell summary of the paired differences (descriptive): median [min, max] = "
                + "; ".join(f"{a}−{b} {p[f'{a}−{b}'].median():+.3f} [{p[f'{a}−{b}'].min():+.3f}, {p[f'{a}−{b}'].max():+.3f}]" for a, b in PAIRS), ""]
    return out


def effect_scale(d):
    x = d[(d.B == 120) & (d.cond != "U_allC")]
    g = x.groupby(["target", "u", "cond"])[["AP", "held_pos_frac"]].mean().reset_index()
    rows = []
    for m in ("AP", "held_pos_frac"):
        pos = g.pivot_table(index=["target", "cond"], columns="u", values=m)
        pe = (pos[190.0] - pos[0.0]).abs()
        meth = g.pivot_table(index=["target", "u"], columns="cond", values=m)
        me = pd.concat([(meth[a] - meth[b]).abs() for a, b in PAIRS[:4] if b != "U_allC"])
        rows.append({"metric": m, "position effect |u=190 − u=0| (16 vehicle×condition)": f"median {pe.median():.3f}, max {pe.max():.3f}",
                     "method effect |condition differences| (8 cells × 4 contrasts)": f"median {me.median():.3f}, max {me.max():.3f}"})
    return ["## Position effect and method effect on the same scale (B = 120, forest conditions)", "",
            pd.DataFrame(rows).to_markdown(index=False), ""]


def controls():
    c = pd.read_csv(os.path.join(ROOT, "results/runs/v3_controls.csv"))
    g = c.groupby(["target", "u", "cond"])[["AP", "held_pos_frac", "new_within_5s", "n_fit", "max_samples_"]].mean().round(3)
    avg = c.groupby(["target", "u", "cond"]).AP.mean().groupby(["target", "cond"]).mean().unstack().round(3)
    return ["## Sample-size controls per position (B = 120)", "",
            "Each value is the mean over seeds 0-2 at one position; the position-averaged numbers quoted in the text are the "
            "mean of the two position means (seed mean first, then the two positions with equal weight).", "",
            g.to_markdown(), "", "Position-averaged AP:", "", avg.to_markdown(), ""]


def detected(v, u, cond, seed):
    sc, _ = A.scorer(v, u, 120.0, 0.5, cond, seed)
    _, C = A.segments(120.0, 0.5, cond)
    tau = sc(seg_windows(v, u, 60.0, *C)).max()
    d, meta = eval_data(v, u, 60.0)
    al = sc(d["Xatt"]) > tau
    hit, n = set(), 0
    for i, info in enumerate(meta["attack_info"]):
        m = np.flatnonzero(d["att_file"] == i)
        if m.size == 0:
            continue
        n += 1
        m = m[np.argsort(d["att_ts"][m], kind="stable")]
        te = d["att_te"][m]
        if (onsets(al[m]) & (te > info["first_attack"]) & (te <= info["last_attack"] + 1.0)).any():
            hit.add(info["file"])
    return hit, n


def gains_losses():
    rows = []
    for v in ("forester", "impala", "silverado", "traverse"):
        for u in (190.0, 0.0):
            for a, b in (("S_sameC", "S_allC"), ("T_main", "T_split")):
                for s in (0, 1, 2):
                    ha, n = detected(v, u, a, s)
                    hb, _ = detected(v, u, b, s)
                    rows.append({"cell": f"{v} u={u:g}", "transition": f"{a}→{b}", "seed": s, "attack captures": n,
                                 f"new alarm (from)": len(ha), "new alarm (to)": len(hb),
                                 "lost": len(ha - hb), "gained": len(hb - ha)})
    t = pd.DataFrame(rows)
    s = t.groupby(["cell", "transition"])[["attack captures", "new alarm (from)", "new alarm (to)", "lost", "gained"]].agg(
        lambda z: f"{z.min()}–{z.max()}" if z.min() != z.max() else f"{z.min()}")
    return ["## New-alarm gains and losses, B = 120, seeds 0-2 (range over seeds)", "",
            "S_sameC→S_allC keeps the forest and lengthens C; T_main→T_split changes both the forest (less training data) "
            "and C, so its losses are not threshold effects alone. A 0→1 new alarm can also appear when a higher threshold "
            "splits a continuous alarm, so losses alone do not measure net change.", "",
            s.to_markdown(), ""]


def denominators(d):
    x = d[(d.B == 120) & (d.cond == "T_main") & (d.seed == 0)]
    t = x.groupby(["target", "u"])[["n_attack_captures", "held_hours", "prefix_hours"]].first().round(3)
    return ["## Denominators (B = 120; identical for every condition of a cell)", "",
            "Attack-capture rates divide by the number of attack captures evaluated; held-out alarm share divides by all "
            "held-out windows; R_normal,5 pools virtual starts every 5 s over all attack-capture normal prefixes "
            "(longer prefixes contribute more starts; a start v counts only if v + t ≤ first attack frame).", "",
            t.to_markdown(), ""]


if __name__ == "__main__":
    d = pd.read_csv(os.path.join(ROOT, "results/runs/v3_main.csv"))
    lines = ["# Final tables", "",
             "Medians, minima and maxima summarise the 8 vehicle × position cells descriptively; the two positions of a "
             "vehicle share the evaluation captures and the source bank, so cells are not independent trials. Seed "
             "variation is algorithmic (same data). 'Maximum' means the observed maximum over cells.", ""]
    for B in (120.0, 30.0, 300.0):
        lines += cell_tables(d, B)
    lines += effect_scale(d) + controls() + denominators(d) + gains_losses()
    open(os.path.join(ROOT, "results/final_tables.md"), "w").write("\n".join(lines))
    print("written", len(lines))
