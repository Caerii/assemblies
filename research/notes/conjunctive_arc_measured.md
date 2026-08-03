# The FSM arc forms. The readout that consumes it is confounded.

Task #92. Implemented Dabagia/Papadimitriou/Vempala (2025) Thm 4's **arc
assembly** in `reference/word_order_learner.py` (`conjunctive_arc=True`, off by
default) and measured it against the existing emergent baseline.

Headline: **18/24 -> 5/24.** The mechanism is right and the numbers are worse,
and the reason is a third thing neither arm controls.

---

## What was built, and the version that was tried first and rejected

Multi-mood word order is an FSM: state = last constituent emitted, symbol =
mood, transition = next constituent. Thm 4's architecture routes every
transition through an arc assembly `A_{q,sigma}` that is a conjunction of
(state, symbol).

**First attempt — reuse the HELPER areas as the arc**, co-firing MOOD into the
current constituent's helper at every step instead of only the first. Measured:
helper mood-separation stays at **1.00**, generation unchanged. The reason is
structural, not a tuning failure: `_activate_role` drives TPJ -> helper for ten
rounds, so a single MOOD co-fire is a few percent of the helper's drive.
[[mood-collapse-is-a-drive-ratio]] one layer down.

And that is exactly where the reference architecture departs from the theorem:
**Thm 4's arc receives the state and the symbol and nothing else.** Our helper
also receives the word, which dominates it. A conjunction cannot form in an area
whose winners are already decided by a third input.

**Second attempt — a dedicated `ARC` area** receiving only SYNTAX_prev and
MOOD. Transition learned as ARC -> helper[next]; generation picks the next
constituent by drive from ARC. The first word uses MOOD alone into ARC (the
start state q0), which routes the reference's special case through the same area
as every other transition. Fired for the same number of rounds as the direct
order synapse it replaces, so the two arms differ in the SOURCE of the order
signal and not in how much drive it gets.

## The arc itself works

    same-(state, mood) formed twice, overlap   1.00, 1.00, 1.00   (stable identity)
    across moods, same state                   q0 0.46-0.62   S 0.70-0.76
                                               V 0.64-0.78    O 0.54-0.66
    HELPER across moods (the thing it replaces)          1.00
    SYNTAX across moods                                  0.96-0.98

So the arc has a stable identity AND is mood-specific, which is more than any
previous intervention achieved -- five are on record as failing, and the
structural workaround `per_mood_syntax` succeeds only by adding an area per
mood. This is the first version where the mood distinction is both LEARNED and
actually present in an assembly.

## And it does not reach the output, because the readout is size-confounded

Drive from ARC to each candidate helper, after S, at `metric="pre_kwta"`:

                          agent    action   patient     picks   wanted
    mood0 (SVO)            8.26      5.72      9.34         O        V
    mood1 (SOV)            7.07      5.42      9.09         O        O

Patient wins in both. The mood-dependent part of the signal is ~15%; the
constant per-helper spread is ~63%. And the helpers are not the same size:

    HELPER recruited w:   S 432    V 658    O 484
    TPJ    recruited w:   S 579    V 689    O 609

`pre_kwta` already divides by the candidate count -- that normalization exists
in this file precisely because "two role areas measured here differed by 391 vs
449, enough to reverse a ranking on size alone." It is not enough. Dividing by
`w` trades one size artifact for another: the largest helper (V, 658) reads as
the *weakest* per neuron.

**So the constituent competition is decided by how much substrate each helper
recruited, not by what was learned.** That is upstream of the arc, upstream of
mood, and it applies to the 18/24 baseline as well as to the 5/24 arc arm. Any
mechanism evaluated through this readout is being scored through a ~4x larger
nuisance term.

This reframes #92. The arc is not refuted -- it is unmeasurable until the
readout is fixed, and "we changed the mechanism and the number went down" is not
evidence about the mechanism when the number is dominated by recruitment.

## Next, in order

1. **Fix the readout first.** Candidates: score a CONTRAST (drive under this
   cue minus mean drive over cues), which cancels any per-area constant; or
   use the paper's own selection rule -- the three ROLE areas in MUTUAL
   INHIBITION -- which is the mechanism [[mutual-inhibition-prefers-untrained]]
   found is dormant in this repo, never having fired in 1373 projections. Note
   MI resolves on total activation, so it may inherit the same bias; that is
   worth measuring rather than assuming.
2. Re-run both arms through the fixed readout. Only then is 18/24 vs 5/24 a
   comparison of mechanisms.
3. Equalize or report recruitment: `w` per helper is a state variable nobody
   was watching, and it differs by 1.5x after 60 sentences.

## Two probe defects found on the way, both reading exactly 1.000

Worth recording because both were introduced by me, in the same hour, while
building the diagnostic that was supposed to prevent this class of error.

1. `mood_separation` first drove SYNTAX from MOOD alone and read **1.000 at
   initialization**, where the true value is ~0.04. On an untrained fiber every
   weight is 1, every candidate ties, and the deterministic index tie-break
   returns identical winners for every mood -- failure mode 4 in
   `assembly_calculus.binding`. A probe that cannot distinguish "merged" from
   "never driven" is useless when those are the two hypotheses.
2. The second version used `read_only()` for isolation and ALSO read **1.000**,
   for the opposite reason: `read_only()` freezes winners, so the probe returned
   whatever was already in the area, identically for every mood.

Fixed by driving through the real chain under `frozen()`, against a **deepcopy**
-- because `frozen()` permits recruitment, and measured here, calling
`mood_separation` between `train` and `generate` CHANGED the generated order.
That contamination also produced a false positive I briefly believed: the arc
appeared to split `['SVO','SVO']` into `['SVO','VOS']`, and with a clean probe
it does not.

[[fake-perfect-probe-signatures]] lists "a 1.000 the exact model does not
produce" as a signature. Two more instances, from two different causes, in one
sitting.
