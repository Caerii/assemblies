# Exploratory probes, 2026-10-09: a logit lens on replay

Every probe here is EXPLORATORY: not registered, run on twenty brains with seeds
outside every registration's (980-999), at cells earlier registrations judged,
and reported only to explain why a registration was designed as it was, never
to support a claim. They are disclosed in
[PREREG_refraction_memory.md](../../PREREG_refraction_memory.md), Amendment 48.
Earlier probes: [../2026-10-08/](../2026-10-08/README.md).

The LOGIT LENS reads every replay step's net drive as a score for every stored
token (the mean drive over its k neurons, in units of the k-WTA threshold); a
word's logit is the max over its tokens. SLACK is the correct next word's mean
margin over the best other word, on the steps it is top-1.

| Probe | Question | Result | Informed |
|---|---|---|---|
| `probe_basin.py` -> `probe_basin.log` | How deep is a spurious attractor's basin, and what predicts the size of a read-time rescue? (10000, 75, 0.48), U = 30-60 | The attractor's strength does not (its self-stability predicts nothing; it rarely out-drives the correct word, which fades instead); slack does: collapsed brains, all 0.00 masked, are rescued in its order (Spearman 0.91), adapted reliability ~ 4.6 (slack - 0.14) | Amendment 48 |
| `probe_basin.py 8000 60 0.6 67 35 45 55` -> `probe_basin_8000.log` | Does the slack law transfer to a second cell? | Same shape (Spearman 0.89); the masked crossing moves with the cell's healthy slack | the relative margin |
| `probe_basin.py <cell> 10` -> `probe_basin_ref10000.log`, `probe_basin_ref8000.log` | A reference margin per cell (U = 10) | 0.382 and 0.337, every brain within 0.004; relative slack puts both cells on one logistic (masked midpoint 0.765, 0.760; adapted 0.632, 0.656), each predicting the other's rescue within 0.03-0.07. The logs end in a traceback from the final correlation (no failing brains at U = 10), since guarded | Amendment 48's constants |
| `probe_fan.py` -> `probe_fan.log` | Who is the correct word's rival? (10000, 75, 0.48), U = 10-50 | The FAN: at step 1 the best rival is a successor of another occurrence of the current word in 86-99% of steps (chance 3-50%); the correct logit stays ~1.13 while the rival's climbs 0.79 -> 0.96 with reuse, and the margin's spread grows (0.05 -> 0.12). Reliability equals the product of per-step win rates; the dangerous step is 2, not 1 (margin 0.09 at U = 40 for survivors) | the fan reading |
| `probe_drive.py` -> `probe_drive.log` | Exact drive decomposition (source on/off the correct track x baseline/learned weight; error < 2e-6) | Correct assembly: baseline 0.73-0.76 vs field 0.60-0.62 (write-time SELECTION of well-wired neurons) + learned 0.38. Rival's learned drive at step 1 is all on-track (the type trace): fan ratio 0.40 / 0.70 / 0.90 at U = 10 / 40 / 50. At step 2 the intruders drive their own learned successor (0.02 / 0.22 / 0.39): two chains race, and at U = 50 the intruder wins (rival 1.12 vs correct 0.90) | two-chain race; a two-variable reduction |
