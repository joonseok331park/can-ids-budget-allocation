"""Tables for the analyses added in revision. Reads results/runs/rev_*.csv, road_rev.csv, rev_bootstrap.csv,
period_by_id.csv and the budget captures; writes results/rev_tables.md.
Cells are vehicle x position; a cell value is the mean over its seeds. Paired differences are taken per seed first.
Usage: python3 src/revision_tables.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from canio import load_ctt_csv  # noqa: E402
from experiment import CTT, load_roles  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results/runs")
ORDER = ["T_main", "S_sameC", "S_allC", "T_split", "U_allC"]
TOL = 0.01


def rd(name):
    p = os.path.join(RUNS, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def prep(d):
    if d is None:
        return None
    d = d.copy()
    d["E5"] = d.new_within_5s - d.prefix_virtual_new_5s
    d["E5_held"] = d.new_within_5s - d.held_virtual_new_5s
    return d


def mr(x, f=".3f"):
    x = pd.Series(x).dropna()
    return f"{np.median(x):{f}} [{x.min():{f}}, {x.max():{f}}]" if len(x) else "–"


def cells(d, metric, cond):
    return d[d.cond == cond].groupby(["target", "u"])[metric].mean()


def paired(d, a, b, metric, key=("target", "u")):
    q = d[d.cond.isin([a, b])].pivot_table(index=list(key) + ["seed"], columns="cond", values=metric)
    return (q[a] - q[b]).groupby(list(key)).mean()


def counts(x, tol=TOL):
    return f"{(x > tol).sum()} up / {(x.abs() <= tol).sum()} tie / {(x < -tol).sum()} down"


def md(df, **kw):
    return df.to_markdown(**kw)


def t_seeds(m10, m3):
    out = ["## 1. Main grid with 10 seeds next to the fixed-protocol 3 seeds", "",
           "Median [min, max] over the 8 cells.", ""]
    rows = []
    for B in (30, 120, 300):
        for c in ORDER:
            r = {"B": B, "cond": c}
            for lab, d in (("3 seeds", m3), ("10 seeds", m10)):
                x = d[d.B == B]
                for m in ("AP", "held_pos_frac", "E5"):
                    r[f"{m} ({lab})"] = mr(cells(x, m, c))
            rows.append(r)
    out += [md(pd.DataFrame(rows), index=False), ""]
    rows = []
    for B in (30, 120, 300):
        x = m10[m10.B == B]
        for a, b, m in (("S_sameC", "T_main", "AP"), ("T_split", "T_main", "held_pos_frac"),
                        ("S_allC", "T_split", "held_pos_frac"), ("S_allC", "T_split", "E5"),
                        ("T_split", "T_main", "E5")):
            p = paired(x, a, b, m)
            p3 = paired(m3[m3.B == B], a, b, m)
            rows.append({"B": B, "contrast": f"{a} − {b}", "metric": m, "3 seeds": counts(p3),
                         "10 seeds": counts(p), "median diff (10 seeds)": f"{p.median():+.3f}"})
    out += ["Paired differences per cell, counted with the ±0.01 tolerance.", "", md(pd.DataFrame(rows), index=False), ""]
    return out


def t_matched(m10, mt):
    out = ["## 2. Donor profile length: 250 s (protocol) versus B/2 (matched to the target profile)", ""]
    rows = []
    for B in (30, 120, 300):
        x10, xm = m10[m10.B == B], mt[mt.B == B]
        tm = cells(x10, "AP", "T_main")
        for c in ("S_sameC", "S_allC"):
            r = {"B": B, "cond": c}
            for lab, x in (("250 s", x10), ("B/2", xm)):
                r[f"AP ({lab})"] = mr(cells(x, "AP", c))
                r[f"alarm ({lab})"] = mr(cells(x, "held_pos_frac", c))
                r[f"E5 ({lab})"] = mr(cells(x, "E5", c))
                diff = cells(x, "AP", c) - tm
                r[f"AP − T_main ({lab})"] = counts(diff)
            rows.append(r)
    out += ["Median [min, max] over 8 cells, 10 seeds. The last columns count cells against T_main.", "",
            md(pd.DataFrame(rows), index=False), ""]
    return out


def t_reference(m10):
    out = ["## 3. Normal reference and ranking negatives", "",
           "E5 with the attack-capture prefix reference (protocol) and with the held-out capture as reference; AP "
           "over attack-capture windows (protocol) and AP / partial AUC (FPR ≤ 0.01) of attack-positive windows "
           "against held-out normal windows. Median [min, max] over 8 cells, 10 seeds.", ""]
    rows = []
    for B in (30, 120, 300):
        x = m10[m10.B == B]
        for c in ORDER:
            rows.append({"B": B, "cond": c, "E5 prefix": mr(cells(x, "E5", c)), "E5 held-out": mr(cells(x, "E5_held", c)),
                         "AP": mr(cells(x, "AP", c)), "AP vs held-out": mr(cells(x, "AP_held", c)),
                         "pAUC0.01 vs held-out": mr(cells(x, "pAUC01_held", c))})
    out += [md(pd.DataFrame(rows), index=False), ""]
    rows = []
    for B in (30, 120, 300):
        x = m10[m10.B == B]
        for a, b in (("S_sameC", "T_main"), ("T_split", "T_main"), ("S_allC", "T_split"), ("S_allC", "S_sameC")):
            r = {"B": B, "contrast": f"{a} − {b}"}
            for m in ("E5", "E5_held", "AP", "AP_held", "pAUC01_held"):
                p = paired(x, a, b, m)
                r[m] = f"{p.median():+.3f}; {counts(p)}"
            rows.append(r)
    out += ["Paired differences: median over cells; counts with the ±0.01 tolerance.", "",
            md(pd.DataFrame(rows), index=False), ""]
    return out


def t_nominal(m10):
    out = ["## 4. Held-out alarm share against the exchangeable level 1/(n_C + 1)", ""]
    rows = []
    for B in (30, 120, 300):
        x = m10[(m10.B == B) & (m10.cond != "U_allC")]
        for c in ORDER[:4]:
            y = x[x.cond == c]
            nom = 1.0 / (y.n_C + 1)
            rows.append({"B": B, "cond": c, "n_C": int(y.n_C.median()), "nominal": f"{nom.median():.3f}",
                         "median achieved": f"{y.held_pos_frac.median():.3f}",
                         "runs above nominal": f"{(y.held_pos_frac > nom).mean():.0%}",
                         "median achieved / nominal": f"{(y.held_pos_frac / nom).median():.1f}",
                         "windows per hour at nominal": f"{7200 * nom.median():.0f}"})
    out += ["Runs = 8 cells × 10 seeds. Adjacent calibration windows overlap by half, so the effective n_C is "
            "smaller than the window count.", "", md(pd.DataFrame(rows), index=False), ""]
    return out


def t_oracle(m10):
    out = ["## 5. Detection at a matched held-out alarm share (oracle threshold, diagnostic only)", "",
           "The threshold is set on the held-out capture so that 1 % or 5 % of its windows alarm, so differences "
           "come from the ranking alone. Median [min, max] over 8 cells, 10 seeds.", ""]
    rows = []
    for B in (30, 120, 300):
        x = m10[m10.B == B].copy()
        for a in (1, 5):
            x[f"E5o{a}"] = x[f"orc{a}_new_within_5s"] - x[f"orc{a}_held_virtual_new_5s"]
        for c in ORDER:
            r = {"B": B, "cond": c}
            for a in (1, 5):
                r[f"R_att,5 @{a}%"] = mr(cells(x, f"orc{a}_new_within_5s", c))
                r[f"E5 held @{a}%"] = mr(cells(x, f"E5o{a}", c))
                r[f"window TPR @{a}%"] = mr(cells(x, f"orc{a}_att_pos_window_tpr", c))
            rows.append(r)
    out += [md(pd.DataFrame(rows), index=False), ""]
    rows = []
    for B in (30, 120, 300):
        x = m10[m10.B == B]
        for a, b in (("S_sameC", "T_main"), ("T_split", "T_main")):
            p = paired(x, a, b, "orc1_att_pos_window_tpr")
            rows.append({"B": B, "contrast": f"{a} − {b}", "window TPR @1 %": f"{p.median():+.3f}; {counts(p)}"})
    out += [md(pd.DataFrame(rows), index=False), ""]
    return out


def t_decomp(m10, dc):
    out = ["## 6. Decomposing T_split − T_main", "",
           "T_splitF_mainC uses the T_split forest (F = 15 % of B) with the T_main calibration segment (C = 20 % of B). "
           "T_splitF_mainC − T_main is the forest change at fixed C; T_split − T_splitF_mainC is the longer C at a "
           "fixed forest. Median over 8 cells of the paired differences, with counts.", ""]
    x = pd.concat([m10[m10.cond.isin(["T_main", "T_split"])], dc])
    rows = []
    for B in (30, 120, 300):
        y = x[x.B == B]
        for a, b in (("T_splitF_mainC", "T_main"), ("T_split", "T_splitF_mainC"), ("T_split", "T_main")):
            r = {"B": B, "step": f"{a} − {b}"}
            for m in ("AP", "held_pos_frac", "E5"):
                p = paired(y, a, b, m)
                r[m] = f"{p.median():+.3f}; {counts(p)}"
            rows.append(r)
    out += [md(pd.DataFrame(rows), index=False), ""]
    return out


def t_featsets(m10, fs):
    out = ["## 7. Feature sets (B = 120)", "",
           "u13: the IF uses the 13 features of the fixed score U. notime: the five inter-arrival features are "
           "removed. Median [min, max] over 8 cells, 10 seeds.", ""]
    x = pd.concat([m10[m10.B == 120], fs])
    rows = []
    for c in ("U_allC", "T_main_u13", "S_allC_u13", "T_main", "S_allC", "T_main_notime", "S_sameC_notime",
              "S_allC_notime", "T_split_notime"):
        rows.append({"cond": c, "AP": mr(cells(x, "AP", c)), "AP vs held-out": mr(cells(x, "AP_held", c)),
                     "alarm": mr(cells(x, "held_pos_frac", c)), "E5": mr(cells(x, "E5", c))})
    out += [md(pd.DataFrame(rows), index=False), ""]
    rows = []
    for a, b in (("S_sameC", "T_main"), ("S_sameC_notime", "T_main_notime"), ("S_allC_u13", "U_allC"),
                 ("T_split_notime", "T_main_notime")):
        r = {"contrast": f"{a} − {b}"}
        for m in ("AP", "AP_held", "held_pos_frac"):
            p = paired(x, a, b, m)
            r[m] = f"{p.median():+.3f}; {counts(p)}"
        rows.append(r)
    out += [md(pd.DataFrame(rows), index=False), ""]
    return out


def t_oem(oem):
    out = ["## 8. Donor manufacturer (B = 120, GM targets, 71 donor windows, psi 71, T_main's C)", "",
           "S_small: windows from all three other vehicles; S_fo71: from the Forester only; S_gm71: from the two other "
           "GM vehicles. Seed means per cell.", ""]
    p = oem.pivot_table(index=["target", "u"], columns="cond", values=["AP", "held_pos_frac"], aggfunc="mean").round(3)
    out += [md(p), ""]
    return out


def t_srccal(m10, mt, sc):
    out = ["## 9. Thresholds from donor normal windows only", "",
           "S_srcC: the source forest with τ = max over donor windows from 430 s to the end of each donor's budget "
           "capture (never in the bank), so the target budget is used only for the profile. Donor profiles of 250 s "
           "or B/2. Median [min, max] over 8 cells, 10 seeds.", ""]
    rows = []
    for B in (30, 120, 300):
        parts = [(c, m10[(m10.B == B) & (m10.cond == c)], "–") for c in ("T_main", "T_split", "S_allC")]
        parts.append(("S_allC", mt[(mt.B == B) & (mt.cond == "S_allC")], "B/2"))
        for L in (250.0, B / 2):
            parts.append(("S_srcC", sc[(sc.B == B) & (sc.donor_len == L)], "250 s" if L == 250 else "B/2"))
        for c, x, lab in parts:
            rows.append({"B": B, "cond": c, "donor profile": lab, "alarm": mr(cells(x, "held_pos_frac", c)),
                         "events/h": mr(cells(x, "held_fa_per_h", c), ".0f"),
                         "R_att,5": mr(cells(x, "new_within_5s", c)), "E5": mr(cells(x, "E5", c)),
                         "E5 held-out": mr(cells(x, "E5_held", c))})
    out += [md(pd.DataFrame(rows), index=False), ""]
    p = sc[sc.B == 120].groupby(["target", "u", "donor_len"])[["held_pos_frac", "held_fa_per_h", "new_within_5s"]].mean()
    out += ["Per cell, B = 120.", "", md(p.round(3)), ""]
    return out


def t_sweep(sw):
    out = ["## 10. Positions every 30 s (B = 120, seeds 0-2)", "",
           "Per vehicle: number of positions, and over positions the median [IQR] of the paired difference and the "
           "counts with the ±0.01 tolerance.", ""]
    rows = []
    for a, b, m in (("S_sameC", "T_main", "AP"), ("S_sameC", "T_main", "AP_held"), ("T_split", "T_main", "held_pos_frac"),
                    ("S_allC", "T_split", "held_pos_frac"), ("S_allC", "T_split", "E5"), ("T_split", "T_main", "E5"),
                    ("S_allC", "T_split", "E5_held"), ("T_split", "T_main", "E5_held"),
                    ("S_srcC", "T_split", "held_pos_frac"), ("S_srcC", "T_split", "E5"), ("S_srcC", "S_allC", "E5_held")):
        if a not in set(sw.cond):
            continue
        p = paired(sw, a, b, m)
        for v, g in p.groupby(level=0):
            q = g.quantile([0.25, 0.5, 0.75]).values
            rows.append({"contrast": f"{a} − {b}", "metric": m, "vehicle": v, "positions": len(g),
                         "median [IQR]": f"{q[1]:+.3f} [{q[0]:+.3f}, {q[2]:+.3f}]", "counts": counts(g)})
        q = p.quantile([0.25, 0.5, 0.75]).values
        rows.append({"contrast": f"{a} − {b}", "metric": m, "vehicle": "all", "positions": len(p),
                     "median [IQR]": f"{q[1]:+.3f} [{q[0]:+.3f}, {q[2]:+.3f}]", "counts": counts(p)})
    out += [md(pd.DataFrame(rows), index=False), ""]
    rows = []
    for c in [c for c in ORDER + ["S_srcC"] if c in set(sw.cond)]:
        for m in ("AP", "held_pos_frac", "E5"):
            g = sw[sw.cond == c].groupby(["target", "u"])[m].mean()
            rows.append({"cond": c, "metric": m, **{v: mr(g.loc[v]) for v in sorted(sw.target.unique())}})
    out += ["Distribution over positions of the seed-mean value, median [min, max].", "",
            md(pd.DataFrame(rows), index=False), ""]
    return out


def t_kinds(m10):
    x = m10[m10.B == 120]
    kc = [c for c in x.columns if c.startswith("AP__")]
    rows = []
    for k in kc:
        r = {"attack kind": k[4:]}
        for c in ("T_main", "S_sameC", "T_split"):
            r[c] = mr(cells(x, k, c), ".2f")
        p = paired(x, "S_sameC", "T_main", k)
        r["S_sameC − T_main"] = f"{p.median():+.2f} [{p.min():+.2f}, {p.max():+.2f}]"
        rows.append(r)
    mac = x.assign(macro=x[kc].mean(axis=1))
    rows.append({"attack kind": "macro mean over kinds", **{c: mr(cells(mac, "macro", c), ".2f") for c in ("T_main", "S_sameC", "T_split")},
                 "S_sameC − T_main": f"{paired(mac, 'S_sameC', 'T_main', 'macro').median():+.2f}"})
    return ["## 11. AP by attack kind (B = 120, 10 seeds)", "", "Median [min, max] over the cells that contain the kind.",
            "", md(pd.DataFrame(rows), index=False), ""]


def t_boot():
    p = os.path.join(ROOT, "results/rev_bootstrap.csv")
    if not os.path.exists(p):
        return []
    b = pd.read_csv(p)
    b["value"] = b.apply(lambda r: f"{r.est:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", axis=1)
    t = b.pivot_table(index=["contrast", "cell"], columns="metric", values="value", aggfunc="first")
    return ["## 12. Bootstrap intervals (B = 120, fixed positions, 10 seeds)", "",
            "Estimate [2.5 %, 97.5 %]. Attack captures resampled within vehicle; held-out capture in 20 s moving "
            "blocks. @m: donor profile matched to B/2.", "", md(t), ""]


def t_iat():
    d = pd.read_csv(os.path.join(ROOT, "results/period_by_id.csv"))
    d = d[d.bucket_ms <= 25]
    a = d[~d.kind.isin(["attack-free", "accessory"])]
    t = a.groupby(["vehicle", "kind"]).iqr_ratio.median().unstack(0).round(1)
    ref = d[d.kind.isin(["attack-free", "accessory"])].groupby(["vehicle"]).iqr_ratio.median().round(2)
    t.loc["(attack-free and accessory)"] = ref
    return ["## 13. Inter-arrival spread before the first attack frame, by attack kind", "",
            "Median over IDs with periods ≤ 25 ms of the IQR ratio (capture prefix / budget capture).", "", md(t), ""]


def t_start():
    """Needs the raw budget captures once; afterwards results/rev_budget_segments.csv is used."""
    cache = os.path.join(ROOT, "results/rev_budget_segments.csv")
    roles = load_roles(os.path.join(ROOT, "results/manifest_ctt.csv"))
    if not os.path.exists(os.path.join(CTT, roles["forester"]["budget"])):
        return ["## 14. Traffic in the budget-capture segments", "", md(pd.read_csv(cache), index=False), ""]
    rows = []
    for v in sorted(roles):
        c = load_ctt_csv(os.path.join(CTT, roles[v]["budget"]))
        t = c["t"] - c["t"][0]
        whole = set(np.unique(c["aid"]).tolist())
        for a, z in ((0, 60), (60, 120), (190, 250), (250, 310)):
            m = (t >= a) & (t < z)
            ids = set(np.unique(c["aid"][m]).tolist())
            rows.append({"vehicle": v, "segment (s)": f"[{a}, {z})", "frames/s": round(m.sum() / (z - a)),
                         "IDs seen": len(ids), "IDs of the capture not seen": len(whole - ids)})
        m = t >= 310
        rows.append({"vehicle": v, "segment (s)": f"[310, {t[-1]:.0f}]", "frames/s": round(m.sum() / (t[-1] - 310)),
                     "IDs seen": len(set(np.unique(c["aid"][m]).tolist())),
                     "IDs of the capture not seen": len(whole - set(np.unique(c["aid"][m]).tolist()))})
    pd.DataFrame(rows).to_csv(cache, index=False)
    return ["## 14. Traffic in the budget-capture segments", "", md(pd.DataFrame(rows), index=False), ""]


def t_road(rr):
    out = ["## 15. ROAD (B = 120, 10 seeds; mean of the two positions, range over positions)", ""]
    rows = []
    for (c, L), g in rr.groupby(["cond", "donor_len"], sort=False):
        pos = g.groupby("u").mean(numeric_only=True)
        f = lambda m: f"{pos[m].mean():.3f} [{pos[m].min():.3f}, {pos[m].max():.3f}]"  # noqa: E731
        rows.append({"cond": c, "donor profile": f"{L:g} s" if c.startswith("S_") else "–", "AP": f("AP"),
                     "AP fabrication": f("AP_fab"), "AP masquerade": f("AP_masq"), "AP vs held-out": f("AP_held"),
                     "alarm (11 captures)": f("held_pos_frac"), "alarm w/o 2 special": f("held_pos_excl"),
                     "alarm dyno": f("held_pos_dyno"), "alarm road": f("held_pos_road")})
    out += [md(pd.DataFrame(rows), index=False), ""]
    hc = [c for c in rr.columns if c.startswith("held_pos__")]
    t = rr[rr.donor_len == 250].groupby("cond")[hc].mean().T.round(3)
    t.index = [c[len("held_pos__"):] for c in t.index]
    out += ["Held-out alarm share per capture (donor profile 250 s).", "", md(t[ORDER]), ""]
    gc = sorted(c for c in rr.columns if c.startswith("APg__"))
    t = rr[rr.donor_len == 250].groupby("cond")[gc].mean().T.round(2)
    t.index = [c[len("APg__"):] for c in t.index]
    out += ["AP per attack and variant (donor profile 250 s).", "", md(t[ORDER]), ""]
    return out


def main():
    m3 = prep(rd("v3_main.csv"))
    m10 = prep(rd("rev_seeds10.csv"))
    parts = ["# Revision analyses", "",
             "All analyses here were added after the fixed-protocol results were known (post hoc).", ""]
    parts += t_seeds(m10, m3)
    for name, f in (("rev_matched.csv", lambda d: t_matched(m10, d)), (None, lambda d: t_reference(m10)),
                    (None, lambda d: t_nominal(m10)), (None, lambda d: t_oracle(m10)),
                    ("rev_decomp.csv", lambda d: t_decomp(m10, d)), ("rev_featsets.csv", lambda d: t_featsets(m10, d)),
                    ("rev_oem.csv", t_oem), ("rev_srccal.csv", lambda d: t_srccal(m10, prep(rd("rev_matched.csv")), d)),
                    ("rev_sweep.csv", lambda d: t_sweep(pd.concat([d, prep(rd("rev_sweepsrc.csv"))])
                                                        if rd("rev_sweepsrc.csv") is not None else d))):
        d = prep(rd(name)) if name else None
        if name is None or d is not None:
            parts += f(d)
    parts += t_kinds(m10) + t_boot() + t_iat() + t_start()
    rr = rd("road_rev.csv")
    if rr is not None:
        parts += t_road(rr)
    open(os.path.join(ROOT, "results/rev_tables.md"), "w").write("\n".join(parts) + "\n")
    print("saved results/rev_tables.md")


if __name__ == "__main__":
    main()
