"""Human-readable summary of the CTT manifest (unique files only) -> results/manifest_ctt_summary.md"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

m = pd.read_csv(ROOT + "/results/manifest_ctt.csv")
u = m.drop_duplicates("sha256")
lines = ["# CTT manifest summary (unique files by sha256)", ""]
lines.append(f"files listed: {len(m)}, unique: {len(u)}, duplicate rows: {len(m) - len(u)}")
lines.append("")
lines.append("## Normal-only captures")
nm = u[u.kind.isin(["attack-free", "accessory"])][["vehicle", "path", "duration_s", "frames", "rate_fps", "n_ids", "attack_frames"]]
lines.append(nm.sort_values(["vehicle", "path"]).to_markdown(index=False))
lines.append("")
lines.append("## Attack captures per vehicle")
at = u[~u.kind.isin(["attack-free", "accessory"])]
g = at.groupby("vehicle").agg(files=("path", "count"), kinds=("kind", lambda s: ", ".join(sorted(set(s)))),
                              median_dur_s=("duration_s", "median"), median_prefix_s=("normal_prefix_s", "median"),
                              min_prefix_s=("normal_prefix_s", "min"), total_prefix_s=("normal_prefix_s", "sum"),
                              median_segments=("n_attack_segments", "median"), median_attack_frac=("attack_frac", "median"))
lines.append(g.round(2).to_markdown())
lines.append("")
lines.append("## Rates and ID counts per vehicle (normal captures)")
lines.append(nm.groupby("vehicle").agg(rate_fps=("rate_fps", "median"), n_ids=("n_ids", "median")).to_markdown())
lines.append("")
lines.append("## Data checks")
lines.append(f"non-monotonic timestamps: {int((~u.t_monotonic).sum())} files; DLC>8 frames: {int(u.dlc_gt8.sum())}")
lines.append(f"attack frames in normal-only captures: {int(nm.attack_frames.sum())}")
lines.append(f"attack captures with zero attack frames: {int((at.attack_frames == 0).sum())}")
open(ROOT + "/results/manifest_ctt_summary.md", "w").write("\n".join(lines))
print("\n".join(lines))
