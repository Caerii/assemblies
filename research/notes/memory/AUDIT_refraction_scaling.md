# AUDIT: refraction + synaptic scaling on the same area

Prompted by the identity in `PREREG_refraction_capacity.md`: refraction charges
the bias against the RAW drive, so any mechanism that rescales raw between
wins denominates the bias ledger in a moving unit. Synaptic scaling (substrate
C) is such a mechanism. Question: do the two co-occur on a live surface, and
what happens when they do?

## Where they co-occur

`NemoArcFSM` and `SequenceTransducer` build their arc with
`refracted=True, refracted_strength=0.1`. Three experiments run that organ
with `Brain(synaptic_scaling=True)` UNSCOPED, so the refracted arc is also
scaled:

    research/experiments/seq_s5_substrate_c.py          (substrate C census)
    research/experiments/seq_s5_theorem_regime.py       (arm C')
    research/experiments/seq_refraction_stability.py    (SCALINGS=[False, True])

The arc is FEEDFORWARD (no arc->arc projection anywhere; `train_transition`
projects symbol+state -> arc, then arc -> state), so
`REFRACTION-CANCELS-CONVERGENCE`'s recurrent result does NOT bite the organ.
This interaction does.

## Measured (2026-09-03, `refraction_audit_scaling.log`)

Feedforward area driven by an AREA fiber with FIXED source winners -- the
arc's situation -- n=4000 k=100 p=0.5 beta=0.1, w_max=None (as S5), 240
rounds, 16 brains, consecutive-round stability:

    refraction s=beta, scaling OFF   stable to ~100 (0.970), departs 100-150
    refraction s=beta, scaling ON    departs by 10-20; late stability 0.120; fill 0.62
    no refraction,     scaling ON    16/16 converged at round 2; late 1.000

**Either alone holds. Together the arc loses its assemblies within ~10
presentations.** Mechanism: scaling renormalizes each winner column after the
Hebbian write, so raw grows by less than (1+beta) per win while the bias,
charged on raw, still compounds at (1+s) = (1+beta); the identity's margin
(net = D against raw ~ D(1+beta)^t) is geometric-small, and the mismatch flips
the winner set within tens of rounds -- faster than a first estimate from
beta*k/n, so the estimate is not quoted; the measurement is.

**A second, engine-level lifetime.** With NO scaling and NO clip the identity
still dies at ~100-150 wins: net = raw - bias is a float32 difference of two
numbers growing as 1.1^t, and at t ~ 130 the cancellation error exceeds the
tie gap between winners and runners-up ([[KWTA-TIE-FRAGILE]]). Both engines
keep the bias in float32. Irrelevant under a clip (which binds first, ~31
rounds at w_max=20), relevant to any w_max=None refracted run longer than ~100
presentations of one conjunction.

## Consequence

Arc-side results of the three scripts above were measured on an arc whose
assemblies cannot hold under the combination. Their STATE-area and census
conclusions are not automatically void -- the state area is neither refracted
nor recurrent-refracted -- but any claim that rests on arc stability under
substrate C needs re-reading with the arc EXCLUDED from scaling
(`synaptic_scaling={state_area}`), which the engine already supports.
`PREREG_substrate_c_homeostasis.md` is the registration to amend.

## Rule

Never scale a refracted area. `Brain(synaptic_scaling=...)` accepts a set of
area names; a refracted arc must not be in it.
