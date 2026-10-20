# CTT README vs data (checked 2026-10-19)

README.md fetched from the Bitbucket API (brooke-lampe/can-train-and-test, branch master) on 2026-10-19,
sha256 49903d1ab6856cb0033d7f74b3aa97cd2dceda587468f41dd6b2fa730d9392a4 (2272 bytes).

README mapping (lines 23-34): set_01 known = Impala, unknown = Silverado; set_02 known = Traverse, unknown = Forester;
set_03 known = Silverado, unknown = Forester; set_04 known = Forester, unknown = Traverse.

Observed: all 12 files in set_01/test_04_unknown_vehicle_unknown_attack (double-3/4, fuzzing-3/4, interval-3/4,
speed-3/4, systematic-3/4, triple-3/4) are byte-identical (same sha256) to the same-named files in
set_02/test_01_known_vehicle_known_attack, and their ID sets match Traverse's attack-free captures (Jaccard 1.0),
not Silverado's (results/vehicle_fingerprint.csv). No other file disagrees with the README mapping.
This study labels vehicles by ID-set fingerprint.
