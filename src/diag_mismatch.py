"""Residual-level diagnostic (exploratory): how far are source residual windows from the target's training windows?

For each target x B x profile condition:
  - per-feature KS statistic between target F windows and the pooled source windows (source P = 250 s, all windows)
  - separability: 5-fold cross-validated AUC of a logistic regression (standardised features) that tells target F
    windows from source windows (class-balanced by subsampling source to the target count, 20 repeats)
  - per-feature median of target F, source, target held-out normal and target attack-positive windows
Output: results/residual_mismatch.csv (per condition), results/residual_mismatch_features.csv (per feature)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FV2 = os.path.join(ROOT, "data/features_v2")
VEH = ["forester", "impala", "silverado", "traverse"]


def separability_blocked(A, B, rng, reps=20, k=5):
    """Time-block CV: both window sets are split into k contiguous blocks in time order and
    fold j tests on block j of both, so overlapping neighbouring windows never sit on both sides of a split."""
    from sklearn.metrics import roc_auc_score
    ia = np.array_split(np.arange(len(A)), k)
    ib = np.array_split(np.arange(len(B)), k)
    aucs = []
    for _ in range(reps):
        xs, ys, gs = [], [], []
        for j in range(k):
            sb = rng.choice(ib[j], min(len(ia[j]), len(ib[j])), replace=False)
            xs += [A[ia[j]], B[sb]]; ys += [np.zeros(len(ia[j])), np.ones(len(sb))]; gs += [np.full(len(ia[j]) + len(sb), j)]
        X, y, g = np.vstack(xs), np.concatenate(ys), np.concatenate(gs)
        fold = []
        for j in range(k):
            tr, te = g != j, g == j
            if len(np.unique(y[te])) < 2:
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(X[tr], y[tr])
            fold.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        aucs.append(np.mean(fold))
    return float(np.mean(aucs))


def separability(A, B, rng, reps=20):
    aucs = []
    for _ in range(reps):
        b = B[rng.choice(len(B), len(A), replace=False)]
        X = np.vstack([A, b]); y = np.r_[np.zeros(len(A)), np.ones(len(b))]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        aucs.append(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())
    return float(np.mean(aucs))


def main():
    rows, frows = [], []
    rng = np.random.default_rng(0)
    for t in VEH:
        S = np.vstack([np.load(os.path.join(FV2, "source", f"{v}_P250.npz"))["X"] for v in VEH if v != t])
        for B in (30, 120):
            for pc in ("short", "ref"):
                d = np.load(os.path.join(FV2, t, f"B{B}_{pc}.npz"))
                F, H = d["Xtr"], d["Xheld"]
                A = d["Xatt"][d["att_label"] == 1]
                ks = [ks_2samp(F[:, j], S[:, j]).statistic for j in range(F.shape[1])]
                ks_h = [ks_2samp(F[:, j], H[:, j]).statistic for j in range(F.shape[1])]
                rows.append({"target": t, "B": B, "profile": pc, "n_F": len(F),
                             "sep_auc_F_vs_source": separability(F, S, rng),
                             "sep_auc_F_vs_held": separability(F, H, rng) if len(H) >= len(F) else np.nan,
                             "sep_auc_F_vs_source_blockcv": separability_blocked(F, S, rng),
                             "sep_auc_F_vs_held_blockcv": separability_blocked(F, H, rng),
                             "mean_ks_F_vs_source": float(np.mean(ks)), "mean_ks_F_vs_held": float(np.mean(ks_h))})
                for j, f in enumerate(FEATURES):
                    frows.append({"target": t, "B": B, "profile": pc, "feature": f,
                                  "ks_F_vs_source": ks[j], "ks_F_vs_held": ks_h[j],
                                  "med_F": float(np.median(F[:, j])), "med_source": float(np.median(S[:, j])),
                                  "med_held": float(np.median(H[:, j])), "med_attack_pos": float(np.median(A[:, j]))})
        print("done", t, flush=True)
    r = pd.DataFrame(rows); fr = pd.DataFrame(frows)
    r.to_csv(os.path.join(ROOT, "results/residual_mismatch.csv"), index=False)
    fr.to_csv(os.path.join(ROOT, "results/residual_mismatch_features.csv"), index=False)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
