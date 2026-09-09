# Not drift, not crowding — the flush schedule was the channel

**Task #147 (E18). Experiment: `research/experiments/drift_vs_crowding.py` (suspects, mechanisms, and falsifiable predictions registered before the data). {record, chunk} × zipf-{200, 400} × seeds 42–46.**

## Both registered suspects refuted

| bar | prediction | measured | verdict |
|---|---|---|---|
| D1 | drift ⇒ accuracy rises with LAST-training position | ρ = 0.03/0.09, CIs span 0 | **refuted** |
| D2 | drift ⇒ image moves more at 400 | drift 0.22 @200 vs 0.03 @400 — INVERTED | **refuted** |
| C1 | crowding ⇒ pool grows 200→400 | ever-fired ~210–230 at BOTH (saturated) | **refuted** |

The post-flush label image is nearly rigid (trajectory 0.97, 1.0, 1.0,
...): once formed it does not wander. Degradation is position-blind and
pool-blind.

## The comparability check became the finding

The chunk arm — identical corpus, identical episodes, the ONLY
difference being deferred-scaling flushes at 8 points instead of 1 —

| PL overlap accuracy | one flush/phase (E17 regime) | 8 flushes |
|---|---|---|
| zipf-200 | 0.540 ± 0.099 | **0.727 ± 0.054** |
| zipf-400 | 0.242 ± 0.048 | **0.512 ± 0.096** |

+0.19 and +0.27. The E17 "budget degradation" was never about the
budget: it was the flush interval growing with the phase (49 episodes
per flush at 200, 107 at 400) until within-interval Hebbian mass
concentration — rich-get-richer inside each column, which the eventual
column normalization CANNOT undo because scaling preserves
within-column ratios — swamped the low-share forms. Position
independence (D1 flat) is this mechanism's signature: the damage is
total accumulated concentration, not training order.

This is the timescale-separation law (E8/E9) recurring one level down,
now with its missing half: the slow loop must be slow relative to the
FAST dynamics (E9 — per-update flushing collided with repetition) but
fast relative to ACCUMULATED MASS (E18 — per-phase flushing lets
concentration run away). The right schedule is a RATE — flush every K
episodes — with an interior optimum in K, exactly like every other
timescale in this program.

## E19 (to register): the flush-rate sweep, and the bar

Arms: flush interval ∈ {per-update, every {13, 25, 50} episodes,
per-phase} at zipf-{200, 400}, scoring the FULL balanced exam (E18
measured PL only — chunk-200's PL 0.727 with SG unknown could put the
balanced number anywhere from 0.72 to 0.84, so the 0.75 bar is LIVE
again and must be scored honestly), plus tense, roles, and E9's R1
regression guard (per-update was worse than per-phase at 50 frames —
the optimum must reproduce that end of the curve too). Mechanism
instrumentation: pre-flush column-mass concentration (max/mean) per
interval — the quantity the schedule is supposed to bound; it should
track accuracy across arms or the account is wrong. If the optimum-K
arm clears 0.75, the arc closes at the bar with every level fixed
where it lives; either way the schedule law (slow loop paced by MASS,
not by phases) joins the E-series' permanent findings.
