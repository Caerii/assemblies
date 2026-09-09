# The forms do not share the lemma — E11's premise refuted by its own controls, and the SG-pull relocated downstream

**Task #140 (E11). Experiment: `research/experiments/overlap_ceiling.py` (bars B0–B5 registered before any overlap was measured). SLOW-scaled R1, seeds 42–46, n=3000.**

## B0 fails — and refutes the premise, not just the power

The registered hypothesis: a plural form's core assembly substantially
overlaps its lemma's (shared GroundingContext, mostly-shared phonology),
so the PL→NUMBER probe drives through SG-trained synapses. Measured:

| control | value |
|---|---|
| self-overlap (word vs itself, two activations) | **1.000** everywhere — the probe is sound |
| form vs OWN lemma ("boys"↔"boy") | 0.000–0.033 |
| form vs UNRELATED lemma ("boys"↔"girl") | 0.000–0.067 |
| per-seed mean overlap, all 5 seeds | 0.013–0.040 |

Form↔lemma overlap is indistinguishable from the unrelated-word floor.
At phon_weight=6, distinct phonology recruits **disjoint** assemblies
despite verbatim-shared grounding — the semantic-drive-share fix works
exactly as designed, and it kills this hypothesis at the premise. All
correlation bars (B1–B5) are void per the registration; the ensembles
confirm nothing correlates with an ~all-zero predictor (B1 −0.04 ± 0.46).

## The forensics that survive the refutation

1. **B3, the error fingerprint, fires WITHOUT its hypothesized cause: 20
   of 24 PL failures answer SG** (4 ties, zero other) — yet the SG-answer
   is uncorrelated with core overlap (pooled ρ = 0.05). The SG-pull is
   real, directional, and lives **downstream of the core representation**
   — in the core→NUMBER fiber mass and/or NUMBER's own SG-dominated
   structure, not in shared core neurons.
2. **The per-item exposure gradient**: every PL form at the pinned ~1
   episode/seed (E7's invariant) sits at 0.20–0.60 accuracy; the single
   form that escaped the pinning ("girls", 4 episodes/seed) reads 1.00.
   Ten items — a lead, not a law — but it points the same way as E8's
   raw-baseline result: per-item writes are what PL accuracy is made of.

## The relocated model (E12's target)

Each PL form's assembly is disjoint from everything else (this note), so
its number signal comes only from its OWN ~1 training episode writing
core→NUMBER toward the PL image. When that marginal mass does not
suffice, the probe defaults to SG — the attractor that the area's total
training mass built. This unifies E7 (exposure pinned), E8 (repetition
lifts the raw baseline), B3 (failures default to SG, never scatter), and
the item gradient above. What it does NOT yet explain: why repetition
failed to lift the SCALED arm (E8/E9) — which is exactly what the next
measurement must decide.

**E12 (to register): per-item drive decomposition.** For each PL form,
read the summed synaptic weight from its core assembly into the PL-image
vs SG-image columns of NUMBER (pure weight readout, no dynamics — E10's
census made per-item). Bars: (a) the PL−SG mass difference separates
correct from failed items; (b) "girls" sits at the top of the mass
distribution; (c) under R4, per-item PL mass rises but — if the scaled
ceiling is the SG default, not item mass — the SG-image mass rises in
proportion, which would name the mechanism that ate E8's repetition gain.
If confirmed, the lever is corpus-side per-form exposure (the
already-queued Zipf + slow-scaling synthesis), not any learning rule.
