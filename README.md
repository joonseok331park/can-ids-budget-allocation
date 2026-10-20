# can-ids-budget-allocation

Code and result files for the paper "Local Training versus Calibration for CAN Intrusion Detection on New Vehicles under a Short Normal-Traffic Budget".

The setting: an intrusion detector is installed on a vehicle it has not seen, and only B seconds of that vehicle's attack-free CAN traffic are available. The first half of the budget builds the vehicle profile. The rest goes to local Isolation Forest training (F), threshold calibration (C), or both. The scripts compare a local IF, an IF trained only on other vehicles' normal residuals, a target-only shift of time from F to C, and a fixed score, on can-train-and-test (four vehicles) and ROAD (one vehicle).

File names with `v3` belong to the fixed protocol reported in the paper. `v2` and the pilot files are from the exploratory phase before it.

## Data

Neither dataset is redistributed here.

- can-train-and-test: official Bitbucket repository `brooke-lampe/can-train-and-test` (DTU data deposit, CC BY 4.0). `results/ctt_repo_listing.csv` lists its 236 files and `results/ctt_download_sha256.csv` the SHA-256 of every downloaded file.
- ROAD: Zenodo record 10462796 (CC BY 4.0). Unpack `road.zip` into `data/raw/road_zenodo/`.

Vehicle labels come from the ID sets, not from the dataset README: the 12 files of `set_01/test_04` are documented as Silverado but are byte-identical to Traverse captures (`results/ctt_readme_check.md`, `results/vehicle_fingerprint.csv`).

## Files

- `src/`: all code.
- `results/runs/`: one row per vehicle, position, budget, condition and seed (`v3_main.csv`, `v3_controls.csv`, `v3_caldiag.csv`, `v3_pratio.csv`, `road_v3.csv`).
- `results/final_tables.md`, `results/v3_headline.md`, `results/road_summary.md`: the tables behind the numbers in the paper.
- `results/new_alarm_sets_B120.csv`: whether each attack capture got a new alarm, per condition and seed, at B = 120.
- `results/manifest_ctt.csv`, `results/manifest_public.csv`, `results/provenance.json`: per-file statistics, roles and hashes. `provenance.json` also lists the feature caches of the exploratory allocation grid (offsets 150 and 300), which the steps below do not create.
- `results/figures/`: Figs. 2 and 3 of the paper and the same allocation plot for B = 30 and 300.
- Data checks and diagnostics: `check_period_shift.py` and `check_period_by_id.py` (recording-clock differences between captures, `period_shift.csv`, `period_by_id_*.csv`), `check_reuse.py`, `check_reuse_perid.py` and `reuse_cases.py` (normal driving reused inside attack captures, `reuse_*`), `check_sort.py` (`sort_check.txt`), `table_data.py` (`table_data_segments.md`, after step 3) and `diag_threshold.py` (`diag_threshold.md`, after step 4). `ctt_readme_check.md` is a written record of the README check.
- Exploratory phase: `stage1_features.py`, `stage2_models.py`, `alloc_pilot.py`, `alloc_grid.py`, `ablation.py`, `clock_norm.py`, `diag_profiles.py` and `diag_mismatch.py`, with `results/runs/v2_*.csv`, `results/runs/alloc_pilot.csv` and `results/v2_summary.md`. `experiment.py`, `nested_features.py`, `models_v2.py` and `alloc_models.py` also date from that phase; the final pipeline reuses their loaders and helpers.

## Reproducing the results

Tested with Python 3.13, numpy 2.5, pandas 3.0, scikit-learn 1.9, matplotlib 3.11 and tabulate 0.10 (for the Markdown tables); `diag_mismatch.py` also needs scipy. Isolation Forest scores can differ slightly between scikit-learn versions. The download step needs `curl`. Figure text is set with pdflatex when it is installed and with the STIX fonts otherwise; PNG output through pdflatex also needs `pdftocairo` or Ghostscript.

1. `python3 src/download_ctt.py` (reads `results/ctt_repo_listing.csv`; `python3 src/list_ctt.py > results/ctt_repo_listing.csv` rebuilds that list), then unpack ROAD.
2. `python3 src/manifest.py` (capture statistics, duplicate groups and the ID-set vehicle labels).
3. `python3 src/nested_features.py --targets forester,impala,silverado,traverse` (per-vehicle metadata and the donor windows for the source bank).
4. `python3 src/alloc_features.py --offsets 190,0 --plens 60`, and the same for B = 30 (`--offsets 235,0 --plens 15`), B = 300 (`--offsets 100,0 --plens 150`) and the profile-share runs (`--offsets 190,0 --plens 30,90`).
5. `bash run_v3.sh` (main and profile-share grids, summary, allocation figures), then `python3 src/alloc_v3_grid.py --what controls` and `python3 src/alloc_v3_grid.py --what caldiag`.
6. `python3 src/road_v3.py --stage features && python3 src/road_v3.py --stage models`.
7. `python3 src/final_tables.py`, `python3 src/summary_tables.py`, `python3 src/export_new_alarm_sets.py`, `python3 src/fig_detect_curve.py results/runs/v3_main.csv 120 results/figures/fig_detect_curve_B120.png` and `python3 src/manifest_public.py`.

The metric code is checked against small hand-computed cases: `python3 src/test_metrics.py`, `python3 src/test_eval_v2.py`, `python3 src/test_eval_v3.py`.

## License

Code: MIT (see `LICENSE`). Result files derived from can-train-and-test and ROAD follow the CC BY 4.0 terms of those datasets.
