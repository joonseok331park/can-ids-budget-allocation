#!/bin/bash
# Main and profile-share grids, summary table and allocation figures.
# Needs the allocation features from README step 4.
set -e
cd "$(dirname "$0")"
python3 src/alloc_v3_grid.py --what main --workers 2
python3 src/alloc_v3_grid.py --what pratio --workers 2
python3 src/summarize_v3.py
for B in 30 120 300; do
    python3 src/fig_alloc.py results/runs/v3_main.csv "$B" "results/figures/fig_alloc_B$B.png"
done
