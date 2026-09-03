# PREREG: does refraction oppose formation-side interference?

Registered before running. Follows `PREREG_anchor_ratio.md` (PASS: capacity is
set at formation by the anchor-to-trained-recurrence ratio) and the reading of
`core/_refraction.py` against the reference `RefractedArea`.

## The identity this rests on

Refraction charges `bias[w] += raw_drive[w] * strength` at every win and
subtracts `bias` before k-WTA. Hebbian plasticity multiplies every active
synapse into a winner by `(1 + beta)`. For a neuron winning repeatedly on the
SAME input with base drive D, after t wins:

    raw_t  = D (1+beta)^t
    bias_t = strength * D * sum_{s<t} (1+beta)^s = D ((1+beta)^t - 1)   at strength = beta
    net_t  = raw_t - bias_t = D                                          exactly

At strength = beta -- which is every call site in this repo and the
reference's own `plasticity` -- refraction is the exact anti-Hebbian
counterweight on a neuron's own repeated input. For a DIFFERENT input that
shares only part of the pre-population, raw is only partially potentiated but
the bias is the neuron's whole history, so a used neuron is handicapped
relative to a fresh one. Refraction is an intrinsic homeostat AND an
orthogonalizer. Preconditions of the identity: no w_max clip binding, no
column rescaling between wins, strength == beta.

## Three predictions

**P0 -- partial-cue recall FAILS with the bias on (derived, so registered as a
prediction rather than discovered).** At the capacity operating point
(n=4000 k=100 p=0.5 beta=0.10 T=8 s=100), a stored member's raw drive from a
half cue is ~ (k/2) p (1+beta)^T ~ 54, its bias ~ (s+k) p ((1+beta)^T - 1)
~ 114, so net < 0 while a fresh neuron reads ~ 25. The reference never
partial-cues a refracted area; its arcs are always driven by the full
training input, where the identity makes net = base.

    PASS: rank-1 with bias ON at M=8 (well under the ceiling) <= 0.25
          (chance at M=8 is 0.125); the control arm (no refraction) reads
          >= 0.95 at M=8 as it always has.

**P1 -- refraction protects the SYNAPTIC memory at formation (the question
that matters).** Read out with the bias masked to zero -- the probe reads what
the synapses hold, with the intrinsic state that would veto them removed --
refraction should raise M* by handicapping the stored neurons that supply the
trained recurrent pull.

    PASS: M*(refracted, bias-masked readout) >= 1.3 x 23.5 = 30.6
    FAIL: <= 1.1 x 23.5 = 25.9
    Also recorded: distinctness and fill at M*, and F3-style early/late
    rank-1 at M=32 -- if refraction works by orthogonalizing, the early-first
    erosion of F3 should flatten.

**P2 -- a refracted assembly must WANDER once w_max binds.** With the clip
at w_max = 20, raw saturates at ~w_max D while bias keeps growing by
beta * raw per win, so net falls by ~beta w_max D per win and the whole
assembly loses to fresh neurons after about

    ln(w_max)/ln(1+beta) + (1 - 1/w_max)/beta  ~  31.4 + 9.5  ~  41 rounds

of continuous stimulus + recurrence. One item, 120 rounds, 16 brains, overlap
with the round-10 assembly tracked per round.

    PASS: refracted arm departs (overlap < 0.5) at a median round in [30, 60];
          control (no refraction, same everything) never departs in 120.
    FAIL: refracted never departs (the identity is wrong or the clip does not
          bind as computed) or departs before round 20 (something other than
          saturation is driving it).

## Engine parity first

The hashed path has no refraction. It gets a per-neuron bias on `HashedArea`
(subtract before k-WTA, charge `raw * strength` at winners, gated on
plasticity like the engine) and is verified against `numpy_sparse` with
`set_refracted(area, True, 0.1)` on the multi-episode capacity protocol at
< 5e-6 relative drive error BEFORE any number above is read.

## Interpretation

* P0 pass, P1 pass, P2 pass -> refraction ADOPTED as a formation-side
  orthogonalizer with a derived lifetime; a decaying bias becomes the design
  question.
* P1 fails -> refraction does not oppose the pull as computed; either the
  handicap is too small relative to the stimulus anchor or the erosion is not
  through the neurons refraction charges.
* P2 fails with no departure -> the identity's saturation term is wrong; the
  clip interacts with the read-time 1/d differently than derived.
