# ROAD external check (B = 120, P = 50 %, ROAD rules fixed before the results, mean over 3 seeds)

The source is all four CTT vehicles (213 bank rows: 54 Forester, 53 from each of the other three). Held-out normal data are 11 normal captures (2.63 h in total). The 4 accelerator captures have no injection interval, so they are left out of AP and new alarms and only their alarm share over all windows is reported.

|       u | cond    |    AP |   held pos |   held ev/h |   on@start |   new |   new≤5s |   chance≤5s prefix |   chance≤5s held |   excess5 |   accelerator pos |
|--------:|:--------|------:|-----------:|------------:|-----------:|------:|---------:|-------------------:|-----------------:|----------:|------------------:|
|   0.000 | S_allC  | 0.834 |      0.174 |     318.353 |      0.000 | 0.885 |    0.770 |              0.286 |            0.402 |     0.484 |             0.122 |
|   0.000 | S_sameC | 0.834 |      0.176 |     321.900 |      0.000 | 0.885 |    0.782 |              0.286 |            0.406 |     0.496 |             0.123 |
|   0.000 | T_main  | 0.831 |      0.233 |     291.876 |      0.000 | 0.828 |    0.713 |              0.408 |            0.398 |     0.304 |             0.219 |
|   0.000 | T_split | 0.843 |      0.154 |     258.052 |      0.046 | 0.793 |    0.678 |              0.313 |            0.324 |     0.365 |             0.124 |
|   0.000 | U_allC  | 0.667 |      0.809 |      54.347 |      0.310 | 0.448 |    0.172 |              0.245 |            0.074 |    -0.072 |             1.000 |
| 190.000 | S_allC  | 0.813 |      0.243 |     350.150 |      0.000 | 0.862 |    0.736 |              0.463 |            0.429 |     0.273 |             0.201 |
| 190.000 | S_sameC | 0.813 |      0.285 |     406.778 |      0.023 | 0.874 |    0.782 |              0.558 |            0.488 |     0.224 |             0.241 |
| 190.000 | T_main  | 0.718 |      0.210 |     309.105 |      0.000 | 0.885 |    0.782 |              0.367 |            0.401 |     0.414 |             0.160 |
| 190.000 | T_split | 0.713 |      0.197 |     282.375 |      0.000 | 0.862 |    0.724 |              0.340 |            0.368 |     0.384 |             0.143 |
| 190.000 | U_allC  | 0.366 |      0.848 |      66.508 |      0.862 | 0.207 |    0.069 |              0.000 |            0.091 |     0.069 |             1.000 |

## AP by attack type (mean of the two positions)

|                                |   S_allC |   S_sameC |   T_main |   T_split |   U_allC |
|:-------------------------------|---------:|----------:|---------:|----------:|---------:|
| correlated_signal_attack       |    0.924 |     0.924 |    0.948 |     0.946 |    0.807 |
| fuzzing_attack                 |    0.569 |     0.569 |    0.402 |     0.38  |    0.155 |
| max_engine_coolant_temp_attack |    0.661 |     0.661 |    0.504 |     0.529 |    0.131 |
| max_speedometer_attack         |    0.87  |     0.87  |    0.832 |     0.836 |    0.57  |
| reverse_light_off_attack       |    0.77  |     0.77  |    0.677 |     0.692 |    0.365 |
| reverse_light_on_attack        |    0.786 |     0.786 |    0.723 |     0.726 |    0.578 |

## Positive-window share per held-out normal capture (mean of the two positions)

|                                            |   S_allC |   S_sameC |   T_main |   T_split |   U_allC |
|:-------------------------------------------|---------:|----------:|---------:|----------:|---------:|
| ambient_dyno_drive_basic_short             |    0.108 |     0.124 |    0.152 |     0.108 |    0.7   |
| ambient_dyno_drive_benign_anomaly          |    0.393 |     0.408 |    0.417 |     0.288 |    0.887 |
| ambient_dyno_drive_extended_long           |    0.238 |     0.256 |    0.266 |     0.217 |    0.606 |
| ambient_dyno_drive_extended_short          |    0.432 |     0.465 |    0.546 |     0.445 |    0.713 |
| ambient_dyno_drive_radio_infotainment      |    0.079 |     0.107 |    0.067 |     0.044 |    0.719 |
| ambient_dyno_drive_winter                  |    0.073 |     0.094 |    0.048 |     0.043 |    0.378 |
| ambient_dyno_exercise_all_bits             |    0.201 |     0.213 |    0.197 |     0.17  |    0.825 |
| ambient_dyno_idle_radio_infotainment       |    0.105 |     0.139 |    0.068 |     0.047 |    0.765 |
| ambient_dyno_reverse                       |    0.073 |     0.096 |    0.058 |     0.046 |    0.589 |
| ambient_highway_street_driving_diagnostics |    0.385 |     0.415 |    0.374 |     0.318 |    0.988 |
| ambient_highway_street_driving_long        |    0.19  |     0.214 |    0.211 |     0.162 |    0.9   |
