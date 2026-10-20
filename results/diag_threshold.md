# Threshold diagnostics (B = 120, seed 0)

Times are seconds from the start u of the budget segment. MAD = distance from the median of the C windows, in MAD units.

| vehicle   |   u | same_scores   |   tau_sameC |   tau_allC | subset_attack   | subset_held   |   lost_new_S_sameC_to_S_allC |   lost_new_T_main_to_T_split |
|:----------|----:|:--------------|------------:|-----------:|:----------------|:--------------|-----------------------------:|-----------------------------:|
| forester  | 190 | True          |      0.5989 |     0.6658 | True            | True          |                            0 |                            0 |
| forester  |   0 | True          |      0.6846 |     0.6846 | True            | True          |                            0 |                            2 |
| impala    | 190 | True          |      0.649  |     0.649  | True            | True          |                            0 |                            1 |
| impala    |   0 | True          |      0.6695 |     0.6695 | True            | True          |                            0 |                            6 |
| silverado | 190 | True          |      0.5809 |     0.602  | True            | True          |                            0 |                            0 |
| silverado |   0 | True          |      0.6707 |     0.6744 | True            | True          |                            0 |                            0 |
| traverse  | 190 | True          |      0.5807 |     0.6384 | True            | True          |                            1 |                            5 |
| traverse  |   0 | True          |      0.5742 |     0.5742 | True            | True          |                            0 |                            0 |

## forester, u = 190

- same source model: identical evaluation scores True, tau S_sameC 0.5989 -> S_allC 0.6658, S_allC alarms a subset of S_sameC alarms (attack captures True, held-out normal True)
- C windows that set the S_allC threshold (top 3): t=80.5s score=0.666 iat_z_gt3 0.0675 (+67492260.1 MAD), iat_short_frac 0.000619 (+619195.0 MAD), id_rate_dev 0.0109 (+33.9 MAD) | t=79.5s score=0.652 iat_z_gt3 0.103 (+103405572.8 MAD), iat_short_frac 0.000619 (+619195.0 MAD), id_rate_dev 0.0114 (+35.5 MAD) | t=81.0s score=0.648 iat_z_gt3 0.0217 (+21685254.0 MAD), iat_short_frac 0.00062 (+619578.7 MAD), id_js 0.00023 (+94.0 MAD)
- C windows that set the T_split threshold (top 3): t=80.5s score=0.662 iat_z_gt3 0.0675 (+67492260.1 MAD), iat_short_frac 0.000619 (+619195.0 MAD), id_js 5.37e-05 (+19.7 MAD) | t=107.5s score=0.661 iat_z_gt3 0.0385 (+38461538.5 MAD), id_js 0.000359 (+147.9 MAD), id_max_iat_z 2.25 (+10.3 MAD) | t=80.0s score=0.654 iat_z_gt3 0.108 (+107940446.7 MAD), id_js 0.000184 (+74.3 MAD), id_max_iat_z 3.7 (+19.6 MAD)
- attack captures that lost their new alarm: S_sameC -> S_allC 0  / T_main -> T_split 0 

## forester, u = 0

- same source model: identical evaluation scores True, tau S_sameC 0.6846 -> S_allC 0.6846, S_allC alarms a subset of S_sameC alarms (attack captures True, held-out normal True)
- C windows that set the S_allC threshold (top 3): t=101.0s score=0.685 id_rate_dev 0.0107 (+8830182.7 MAD), iat_short_frac 0.000619 (+619195.0 MAD), iat_z_gt3 0.115 (+27.6 MAD) | t=72.0s score=0.667 id_rate_dev 0.0023 (+424733.9 MAD), iat_z_gt3 0.136 (+33.3 MAD), iat_z_mean 0.632 (+9.2 MAD) | t=78.0s score=0.665 id_rate_dev 0.0209 (+18981192.7 MAD), iat_short_frac 0.00124 (+1238390.1 MAD), id_js 0.000129 (+39.0 MAD)
- C windows that set the T_split threshold (top 3): t=101.0s score=0.678 id_rate_dev 0.0107 (+8830182.7 MAD), iat_short_frac 0.000619 (+619195.0 MAD), id_js 7.85e-05 (+17.9 MAD) | t=84.5s score=0.623 id_rate_dev 0.0204 (+18556458.8 MAD), iat_short_frac 0.00124 (+1237623.8 MAD), id_js 0.000127 (+38.2 MAD) | t=101.5s score=0.611 iat_short_frac 0.00062 (+619578.7 MAD), iat_z_gt3 0.0979 (+11.6 MAD), iat_z_mean 0.55 (+10.0 MAD)
- attack captures that lost their new alarm: S_sameC -> S_allC 0  / T_main -> T_split 2 DoS, rpm-accessory

## impala, u = 190

- same source model: identical evaluation scores True, tau S_sameC 0.6490 -> S_allC 0.6490, S_allC alarms a subset of S_sameC alarms (attack captures True, held-out normal True)
- C windows that set the S_allC threshold (top 3): t=112.0s score=0.649 id_rate_dev 0.00895 (+3.7 MAD), id_js 8.1e-05 (+1.7 MAD), iat_z_gt3 0.00278 (+1.4 MAD) | t=92.0s score=0.633 id_rate_dev 0.00556 (+1.1 MAD), ham_z_mean 1.24 (+0.7 MAD), id_max_change_dev 0.749 (+0.7 MAD) | t=105.0s score=0.622 id_js 0.000122 (+5.1 MAD), iat_z_mean 0.435 (+1.7 MAD), id_max_iat_z 1.35 (+1.6 MAD)
- C windows that set the T_split threshold (top 3): t=105.0s score=0.682 id_js 0.000122 (+5.9 MAD), id_rate_dev 0.00619 (+2.9 MAD), iat_z_mean 0.435 (+2.1 MAD) | t=115.0s score=0.680 id_js 0.000106 (+4.4 MAD), ham_z_mean 1.27 (+2.3 MAD), new_transition_frac 0.00897 (+1.8 MAD) | t=114.0s score=0.680 new_transition_frac 0.0118 (+2.7 MAD), range_viol_frac 0.195 (+1.4 MAD), id_max_iat_z 1.23 (+0.7 MAD)
- attack captures that lost their new alarm: S_sameC -> S_allC 0  / T_main -> T_split 1 standstill

## impala, u = 0

- same source model: identical evaluation scores True, tau S_sameC 0.6695 -> S_allC 0.6695, S_allC alarms a subset of S_sameC alarms (attack captures True, held-out normal True)
- C windows that set the S_allC threshold (top 3): t=99.5s score=0.669 id_js 0.000347 (+24.2 MAD), id_rate_dev 0.0252 (+14.9 MAD), range_viol_frac 0.315 (+6.4 MAD) | t=98.5s score=0.660 range_viol_frac 0.334 (+7.1 MAD), ham_z_mean 1.16 (+2.3 MAD), byte_change_dev 0.185 (+2.2 MAD) | t=100.5s score=0.657 id_rate_dev 0.0138 (+7.0 MAD), range_viol_frac 0.308 (+6.1 MAD), id_js 0.000115 (+4.6 MAD)
- C windows that set the T_split threshold (top 3): t=115.0s score=0.607 id_js 0.000152 (+7.6 MAD), id_rate_dev 0.0129 (+4.2 MAD), id_max_iat_z 1.24 (+2.3 MAD) | t=98.5s score=0.604 range_viol_frac 0.334 (+7.5 MAD), byte_change_dev 0.185 (+2.6 MAD), ham_z_mean 1.16 (+1.9 MAD) | t=104.0s score=0.599 range_viol_frac 0.274 (+5.1 MAD), rate_logratio 0.00512 (+2.7 MAD), id_rate_dev 0.00789 (+1.7 MAD)
- attack captures that lost their new alarm: S_sameC -> S_allC 0  / T_main -> T_split 6 double, double, force-neutral, force-neutral, rpm, speed

## silverado, u = 190

- same source model: identical evaluation scores True, tau S_sameC 0.5809 -> S_allC 0.6020, S_allC alarms a subset of S_sameC alarms (attack captures True, held-out normal True)
- C windows that set the S_allC threshold (top 3): t=79.5s score=0.602 id_max_ham_z 2.25 (+3.4 MAD), id_max_iat_z 2.33 (+2.2 MAD), ham_z_mean 0.553 (+2.0 MAD) | t=84.5s score=0.594 iat_short_frac 0.000389 (+388651.4 MAD), id_rate_dev 0.00703 (+16.2 MAD), id_max_change_dev 0.373 (+4.0 MAD) | t=85.0s score=0.587 iat_short_frac 0.000389 (+388651.4 MAD), id_rate_dev 0.00703 (+16.2 MAD), byte_change_dev 0.0782 (+4.1 MAD)
- C windows that set the T_split threshold (top 3): t=84.5s score=0.650 iat_short_frac 0.000389 (+388651.4 MAD), id_rate_dev 0.00703 (+16.2 MAD), byte_change_dev 0.0776 (+4.2 MAD) | t=85.0s score=0.631 iat_short_frac 0.000389 (+388651.4 MAD), id_rate_dev 0.00703 (+16.2 MAD), byte_change_dev 0.0782 (+4.5 MAD) | t=114.0s score=0.611 id_max_short 0.01 (+10000000.0 MAD), iat_short_frac 0.000389 (+389105.1 MAD), id_max_iat_z 2.56 (+3.1 MAD)
- attack captures that lost their new alarm: S_sameC -> S_allC 0  / T_main -> T_split 0 

## silverado, u = 0

- same source model: identical evaluation scores True, tau S_sameC 0.6707 -> S_allC 0.6744, S_allC alarms a subset of S_sameC alarms (attack captures True, held-out normal True)
- C windows that set the S_allC threshold (top 3): t=83.5s score=0.674 id_max_short 0.667 (+666666666.7 MAD), iat_short_frac 0.000777 (+777302.8 MAD), id_max_iat_z 12.5 (+38.1 MAD) | t=117.5s score=0.671 id_max_short 0.02 (+20000000.0 MAD), iat_short_frac 0.000779 (+778816.2 MAD), id_max_ham_z 2.9 (+5.2 MAD) | t=119.0s score=0.659 id_max_short 0.01 (+10000000.0 MAD), iat_short_frac 0.000389 (+388651.4 MAD), id_max_ham_z 4.55 (+11.9 MAD)
- C windows that set the T_split threshold (top 3): t=83.5s score=0.634 id_max_short 0.667 (+666666666.7 MAD), iat_short_frac 0.000777 (+777302.8 MAD), id_max_iat_z 12.5 (+36.4 MAD) | t=83.0s score=0.626 id_max_short 0.333 (+333333333.3 MAD), iat_short_frac 0.000389 (+388651.4 MAD), id_rate_dev 0.0117 (+26.1 MAD) | t=115.5s score=0.625 iat_short_frac 0.000389 (+388953.7 MAD), id_rate_dev 0.00842 (+14.3 MAD), ham_z_mean 0.731 (+3.7 MAD)
- attack captures that lost their new alarm: S_sameC -> S_allC 0  / T_main -> T_split 0 

## traverse, u = 190

- same source model: identical evaluation scores True, tau S_sameC 0.5807 -> S_allC 0.6384, S_allC alarms a subset of S_sameC alarms (attack captures True, held-out normal True)
- C windows that set the S_allC threshold (top 3): t=81.0s score=0.638 id_max_change_dev 0.529 (+11.9 MAD), id_max_ham_z 3.49 (+11.0 MAD), byte_change_dev 0.125 (+10.1 MAD) | t=80.5s score=0.623 byte_change_dev 0.107 (+7.0 MAD), id_max_ham_z 2.62 (+6.8 MAD), id_max_iat_z 2.89 (+6.7 MAD) | t=82.5s score=0.622 id_max_change_dev 0.73 (+20.7 MAD), id_max_ham_z 4.71 (+16.9 MAD), byte_change_dev 0.158 (+16.1 MAD)
- C windows that set the T_split threshold (top 3): t=80.5s score=0.647 id_max_iat_z 2.89 (+7.7 MAD), byte_change_dev 0.107 (+7.1 MAD), id_max_ham_z 2.62 (+6.2 MAD) | t=92.0s score=0.642 id_js 7.91e-05 (+15.0 MAD), iat_z_gt3 0.07 (+14.7 MAD), id_max_iat_z 3 (+8.2 MAD) | t=81.0s score=0.633 byte_change_dev 0.125 (+10.4 MAD), id_max_change_dev 0.529 (+10.3 MAD), id_max_ham_z 3.49 (+10.1 MAD)
- attack captures that lost their new alarm: S_sameC -> S_allC 1 systematic / T_main -> T_split 5 force-neutral, force-neutral, rpm, standstill, standstill

## traverse, u = 0

- same source model: identical evaluation scores True, tau S_sameC 0.5742 -> S_allC 0.5742, S_allC alarms a subset of S_sameC alarms (attack captures True, held-out normal True)
- C windows that set the S_allC threshold (top 3): t=106.0s score=0.574 iat_z_gt3 0.0535 (+11.2 MAD), id_max_iat_z 2.87 (+7.0 MAD), iat_z_mean 0.678 (+4.5 MAD) | t=105.5s score=0.570 iat_z_gt3 0.0475 (+9.9 MAD), id_max_iat_z 2.8 (+6.8 MAD), iat_z_mean 0.652 (+4.1 MAD) | t=94.0s score=0.566 iat_z_gt3 0.039 (+8.0 MAD), id_max_iat_z 2.86 (+7.0 MAD), iat_z_mean 0.709 (+5.0 MAD)
- C windows that set the T_split threshold (top 3): t=106.0s score=0.645 iat_z_gt3 0.0535 (+9.5 MAD), id_max_iat_z 2.87 (+6.7 MAD), iat_z_mean 0.678 (+4.4 MAD) | t=82.0s score=0.639 id_js 7.4e-05 (+9.8 MAD), iat_z_gt3 0.045 (+7.8 MAD), id_max_iat_z 3.08 (+7.5 MAD) | t=94.0s score=0.635 iat_z_gt3 0.039 (+6.7 MAD), id_max_iat_z 2.86 (+6.6 MAD), iat_z_mean 0.709 (+4.8 MAD)
- attack captures that lost their new alarm: S_sameC -> S_allC 0  / T_main -> T_split 0 
