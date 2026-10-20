# Data and segments

Times are seconds from the first frame of the budget capture. Windows: 1 s long, 0.5 s stride.

|   B (s) | short P    | reference P   | F          | C          |   F windows |   C windows |
|--------:|:-----------|:--------------|:-----------|:-----------|------------:|------------:|
|      30 | [235, 250) | [0, 250)      | [250, 259) | [259, 265) |          17 |          11 |
|     120 | [190, 250) | [0, 250)      | [250, 286) | [286, 310) |          71 |          47 |

| vehicle   | budget capture (s)       | held-out capture (s)    |   held-out windows |   attack captures |   attack kinds |   attack-capture windows |   attack-positive share |   prefix normal (h) |
|:----------|:-------------------------|:------------------------|-------------------:|------------------:|---------------:|-------------------------:|------------------------:|--------------------:|
| forester  | attack-free-3.csv (572)  | attack-free-4.csv (208) |                415 |                48 |             12 |                    36547 |                   0.084 |                1.64 |
| impala    | attack-free-1.csv (1350) | attack-free-2.csv (875) |               1748 |                27 |             10 |                    26953 |                   0.069 |                1.01 |
| silverado | attack-free-3.csv (667)  | attack-free-4.csv (504) |               1006 |                38 |             12 |                    22178 |                   0.108 |                1.48 |
| traverse  | attack-free-1.csv (445)  | attack-free-2.csv (287) |                572 |                43 |             12 |                    51784 |                   0.045 |                2.58 |

Impala: standstill-1 is excluded from all evaluation and standstill-2 from the prefix normal evaluation.
Traverse: force-neutral-2 of set_04/test_04 has no attack label and is excluded.