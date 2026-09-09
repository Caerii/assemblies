# PREREG: the theorem regime — homeostasis where its preconditions actually hold

Registered before running. Follows PREREG_substrate_c_homeostasis.md and its
smoke diagnostic (7a75f4a): substrate C at the REGISTERED protocol compresses
the trained margin 2.4x (block drive excess 62 -> 25 over a background tail
of sd ~4), because the registered protocol sits OUTSIDE the theorems' stated
preconditions on every axis:

    kp >= 3 ln n     ours 28.0 vs floors 29.7 (n=20k) / 31.8 (n=40k)
    T >= (1/b)ln(n/k)  ours 15  vs floors 56.6 (n=20k) / 63.5 (n=40k)
    no w_max         ours clips at 20x init; (1.1)^57 = 228 >> 20

The third is the deep one. At the theorems' own T floor, substrate A's
multiplicative Hebbian hits the w_max clip at T~31 and saturates -- and a
CLIPPED substrate C degenerates toward A, because the column's untrained
background dominates its mass, pinning the renorm scale at ~1 while the
trained synapses sit at the cap. The theorems have no w_max because
homeostasis IS their boundedness mechanism. Deep training may be reachable
only under homeostasis; that is precisely the CDS-scale question, where
Zipf head words receive thousands of presentations
([[hebbian-mass-follows-frequency]]).

## Arms (matched pair; everything else the registered S5 protocol)

    A'  norm_init=False, synaptic_scaling=False, organ_p=0.5, T=64, w_max=None
    C'  norm_init=False, synaptic_scaling=True,  organ_p=0.5, T=64, w_max=None

organ_p=0.5: kp=35 clears both floors. T=64 clears both T floors.
w_max=None: supported configuration (every clip site is None-guarded);
A' is then pure unbounded Hebbian -- the runaway that homeostasis exists
to prevent, included as the honest same-regime control, not a strawman.

beta stays 0.10, the registered organ value. The theorems' beta ceiling
(ln n / 2kL) does not map cleanly onto this organ (their L is memorized-
sequence length; our transitions are length-1 conjunctions trained in
parallel) -- noted as an open mapping, not silently tuned around.

## Readouts

The live soft census (the primary instrument, snapped inside the probe),
exact@{10,50,100,500} (descriptive; [[exact-tables-are-tie-fragile]]),
hard defects, and per-arm stored-weight anatomy: max stored synapse and
column-mass distribution on the arc->state fiber.

## Bars, stated now

* **TR1 (machines intact):** zero hard defects, both arms, all 40 organs.
  *Prediction: PASSES (~80%).*
* **TR2 (the absolute claim):** C' total soft pairs < 30, i.e. the
  theorems' substrate IN ITS REGIME beats registered substrate A at its
  own protocol. *Prediction: PASSES (~55%) -- genuinely open; per-fiber
  granularity and the excluded stimulus fibers are the registered
  suspects if it fails.*
* **TR3 (in-regime comparison):** C' total soft <= A' total soft.
  *Prediction: PASSES (~60%).*
* **TR4 (boundedness, mechanical):** A' max stored synapse grows ~(1.1)^64
  (>100x init) while C' stays bounded (<10x the A' figure's log --
  concretely: C' max stored synapse < 60). *Prediction: PASSES (~90%).*

## Interpretation, stated now

* TR2+TR3 pass: the theorems' substrate is validated at its own
  preconditions; homeostasis becomes the default candidate for deep
  training; register the M-ceiling re-measurement and the CDS training
  design on substrate C'.
* TR2 fails, TR3 passes: in-regime homeostasis is harmless but the
  registered protocol's shallow-training margin is doing the real work;
  substrate A remains default for shallow protocols, C' for deep.
* TR3 fails (C' worse in-regime): per-fiber homeostasis is refuted at
  this granularity; the joint formulation (the docstring's own diagnosis)
  is the registered next step before touching stimulus geometry.
* TR4 fails: the boundedness mechanics are misunderstood; stop and audit
  before interpreting TR2/TR3.
* A' fails TR1 (hard defects from runaway): reported as the runaway's
  cost, not hidden; C' is then judged against registered A alone.

## Committed in advance

1. Bars before data; smoke checks API only, numbers void.
2. Both arms run to completion and both full census tables are committed
   whatever they say; failed bars are evidence.
3. No default changes from this unit; that requires the M-ceiling
   follow-up plus a migration note.
4. Build threading (organ_p / presentations / w_max as build() kwargs
   defaulting to the registered values) must leave every existing call
   site byte-identical -- guarded by the registered-defaults contract
   (defaults unchanged) and the goldens.

---

## Result (2026-08-24, 80/80 cells, all four bars PASS)

    TR1 PASS  zero hard defects, both arms, all 40 organs
    TR2 PASS  C' soft 0 < registered A's 30
    TR3 PASS  C' soft 0 <= A' soft 0
    TR4 PASS  A' max stored 445.8-490.4 (>100), C' max stored 13.2-15.7 (<60)

Every one of the 80 cells is PERFECT: soft 0, hard 0, first_bad 500,
exact@500 True. Both arms, all four groups, all ten seeds.

### The finding is NOT "homeostasis works"

**TR3 passes through a CEILING and the claim it operationalises dies.**
It was registered as "C' soft <= A' soft" to test whether homeostasis helps
in-regime. It passes as 0 <= 0 -- because the UNBOUNDED HEBBIAN CONTROL is
equally perfect. A' needs no homeostasis at all. This is the E-series
pattern again (a bar passing while its motivating claim dies), now on this
registration, and it is reported as both per the commitment above.

**What actually fixed the organ was being IN REGIME.** The registered S5
protocol sits outside the theorems' preconditions on every axis (kp 28.0 vs
floor 29.7; T=15 vs floor 56.6) and produced 30 soft pairs. Clearing both
floors (organ_p=0.5 -> kp=35, T=64) produced ZERO -- with or without
homeostasis. The preconditions were the whole story.

### What homeostasis DID do, exactly as specified

TR4 is unambiguous and mechanical:

    A' (unbounded)   w_stored_max 445.8-490.4   ~ (1.1)^64 = 456, as predicted
                     colsum_med   34297-44189
    C' (homeostatic) w_stored_max  13.2- 15.7
                     colsum_med    9999-20000   = the setpoint, rows * p

C' holds column mass ON the setpoint to four figures. So homeostasis
delivers precisely the boundedness it promises -- boundedness simply was not
NEEDED for correctness at this depth on this task. That is a real result
about scope, not a null.

### The dissociation worth following

Separately measured this session ([[arc-training-is-not-batchable]]): the
arc's identical-assembly fraction decays to 0.12 by presentation 12, i.e.
the assemblies DRIFT continuously during training. Yet every trajectory here
is exact to 500 steps. **Training-time assembly instability does not imply
functional failure.** The machine is right while its internals keep moving.

That is a caution about the stability instrument, not a vindication of it:
drift still blocks batching and the low-rank fast path (both need stability),
but it does NOT predict dysfunction. Cf.
[[distinctness-is-not-information]].

### Registered interpretation, applied honestly

The note's rule for "TR2+TR3 pass" was *the theorems' substrate is validated
at its own preconditions; homeostasis becomes the default candidate for deep
training*. That rule ASSUMED C' would beat A'. It tied at ceiling instead, so
the rule does not apply as written and is not forced. The supported
conclusions are narrower:

1. In-regime training is sufficient for this organ; out-of-regime was the
   defect. Register the floors as a build-time precondition.
2. Homeostasis is mechanically correct and unnecessary here. Whether it is
   necessary where A' would run away -- deeper T, higher load, or the
   recurrent settings where w_max=None is genuinely dangerous -- is untested
   and is the natural follow-up.
3. No default changes, per commitment 3.

### Confound registered AFTER the fact, stated plainly

The arc carries refraction, whose bias accumulates GEOMETRICALLY and is never
cleared (see PREREG_refraction_stability.md). The theorems have no refraction
term. This run cannot separate "homeostasis is unnecessary" from "refraction
dominated whatever homeostasis did". Noted as a limitation of this design,
not repaired by it.
