# The latch holds exactly, and the attractor flattens — E16's dynamics were understood, its substrate change was not

**Task #145 (E16). Mechanism `9884615` + `3f61148` (competition modes; value-area recurrence trained); measurement `90e9278` (`competition_dynamics.py`, bars L/S/X); revert `f096548` after the paired substrate measurement. ZIPF × {200, 400} × seeds 42–51, 5 readout variants per trained parser.**

## Bar accounting

- **L PASSES EXACTLY — the dynamics are understood.** acc(latched) −
  acc(oneshot) = 0.0000 ± 0.0000 at every T and every seed, while the
  margin grows +0.545. The paper's latch HOLDS a decision through time;
  it cannot improve one, because MI silences the loser at step 1 and
  with it the only evidence that could have reversed the call. A
  falsifiable prediction, registered, confirmed to the fourth decimal.
- **S FAILS**: settled−oneshot = +0.030 ± 0.038 (CI spans zero).
  Settled-3 is the best variant (0.658 at 200) but does not reach E15's
  overlap reference.
- **X FAILS**: best variant at zipf-400 = 0.630 < 0.75, with the
  exposure arithmetic CONFIRMED (107 episodes; head form "men" at 14
  exposures/seed) — exposure delivered, accuracy did not follow. Again.

## The finding the bars did not register: the substrate change was the regression

E16's cells differ from E15's by exactly one thing — the #89 fix let
training write the value areas' self-recurrence (the prerequisite for
multi-step evidence). Paired same-seed, same-corpus:

| readout, zipf-200 | E15 (no recurrence) | E16 (recurrence trained) | paired Δ |
|---|---|---|---|
| overlap | **0.700 ± 0.037** | 0.610 ± 0.067 | **−0.090** (8/10 seeds negative) |
| MI oneshot | 0.637 ± 0.048 | 0.597 ± 0.036 | −0.040 |

And on the recurrence-trained substrate the exposure law FLATTENS
("men" 14 eps → 0.70, "women" 10 → 0.40, where clean-substrate
calibration put 4 eps at ~1.0). Mechanism: every episode feeds the
area's previous winners back in, deepening ONE generic label attractor
that answers the same regardless of which word drives it — the
[[distinctness-is-not-information]] tradeoff surfacing at the
feature-image level. A value area must encode WHICH words carry its
value; a deep attractor encodes only that it is itself.

REVERTED (`f096548`): feed-forward label training is the measured-best
form; the competition modes remain as tested instruments (no-ops
without trained recurrence, stated in their docstring); #89's
disagreement remains open as an engine question, now with a measured
warning attached to the obvious "fix".

## Standing after E16

- Best-known configuration: **E15's — split areas, feed-forward label
  training, within-area overlap readout: 0.700 ± 0.037 at zipf-200.**
- The paper's inter-area inhibition is now fully characterized in this
  setting: it is a HOLD/COMMIT device (L), not an accuracy device; the
  accuracy lives in within-area evidence.
- The 0.75 bar remains open with ONE unmeasured cell left in the whole
  program: **zipf-400 on the clean substrate** — E16 measured 400 only
  on the damaged one. E17 (to register): revert-substrate ZIPF ×
  {200, 400}, overlap + oneshot readouts; bars: (i) zipf-200 overlap
  replicates 0.700 (the revert verified by measurement, not by
  assumption); (ii) zipf-400 overlap vs the bar, scored BOTH on the
  attested exam and on zipf-200's 30 forms as the fixed probe (the
  exam grows 30 → 43 at 400; the growing-exam lesson applies).
