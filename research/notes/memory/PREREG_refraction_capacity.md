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

---

## Result (2026-09-03): P0 PASS, P1 FAIL, P2 FAIL-AS-REGISTERED -- and the failure is the finding

Parity gate first: `test_refracted_capacity_protocol_reproduces_numpy_sparse`
-- net drive and accumulated bias both < 5e-6 relative to `numpy_sparse`
with `set_refracted`. Then, at n=4000 k=100 p=0.5 beta=0.10 w_max=20, 16
brains (logs `refraction_P0_bias_on.log`, `refraction_P1_bias_masked.log`,
`refraction_P2_strength_sweep.log`):

**P0 -- PASS, for a stronger reason than derived.** Bias-on rank-1 at M=8 is
0.133 (chance 0.125, bar <= 0.25). But fill at M=8 is 0.989: the area had
been consumed. The derived readout veto was never the operative effect.

**P1 -- FAIL.** Bias-masked rank-1 at M=8 is 0.141, M* = 8 CENSORED at fill
0.989. There was no synaptic memory to protect: with refraction at strength
beta the assemblies never formed.

**P2 -- FAIL as registered, then explained.** The recurrent refracted arm
departs at ROUND 11 (every brain, to below-chance overlap), not ~41; the
feedforward refracted arm holds to ~20 and settles at 0.63 (1/16 ever below
0.5). So the recurrent churn is not the clip. The identity itself says why:
in the stable state net drive is CONSTANT under repetition, so refraction at
strength = beta removes the Hebbian convergence force. A feedforward area needs
none -- its input ranking is fixed, which is the reference's only use of
`RefractedArea` -- but a recurrent assembly converges ONLY through
rich-get-richer, and with the force cancelled the changing recurrent input
reshuffles the winners every round.

Strength sweep, consecutive-round stability, 240 rounds:

    s/beta   converged   conv round   late stab   fill
    0 (ctl)   16/16          4         1.000      0.034
    0.5       16/16         48         1.000      0.051
    0.7       16/16         45         1.000      0.078
    0.8        1/16        229         0.221      0.999   <- transition
    0.9        0/16         --         0.008      1.000
    0.95       0/16         --         0.008      1.000
    1.0        0/16         --         0.005      1.000
    FF, 1.0   14/16        182         0.993      0.094

A SHARP TRANSITION between 0.7 and 0.8 beta. Below it the assembly converges
~10x slower than control, and the registered saturation arithmetic (~41
rounds) reappears as a TRANSIENT re-ranking at rounds 44-48 that the assembly
survives (at 0.7 beta one event permanently swaps ~12% of the assembly, then it
re-locks). Above it the area churns through EVERY neuron: refraction at high
strength is a firing-rate equalizer, and firing-rate homeostasis is
incompatible with attractor memory in a recurrent k-WTA area. A rough estimate
from the transient handicap (s * rec against (beta - s) * (stim + rec), stim ~
rec here) puts the critical ratio near 2/3; measured in (0.7, 0.8). Approximate.

**On representational drift** (raised while this ran): this mechanism gives
EVENT-DRIVEN drift below the transition and total per-round turnover above it,
never slow diffusive drift. Graded drift would need a decaying bias or noise.

**Interpretation, per the registered clauses:** P1 fails -> "refraction does
not oppose the pull as computed"; the specific reason is that at strength beta
it also cancels formation. Whether refraction BELOW the transition protects
formation is Amendment 1, below.

---

## Amendment 1 (2026-09-03, post hoc, labelled): refraction BELOW the transition

Question: does refraction at a strength where the assembly still converges
protect formation? Same protocol, `--refracted-factor` 0.5 and 0.7, both
readouts (logs `refraction_amend1_*.log`).

    factor  readout   rank-1 M=8 / 16 / 24     pw/chance M=8..16   fill M=16 / 24   M*
    0.5     masked    0.984 / 0.992 / 0.258     0.00 - 0.05         0.887 / 0.977   19.6
    0.5     net       0.250 / 0.062 / 0.042     (same training)                      8.0
    0.7     masked    0.180 / 0.062 / 0.042     1.07 - 5.19         0.913 / 0.938    8.0
    0.7     net       0.125 / 0.062 / 0.042                                          8.0
    control (no refraction)                     ~1.5                0.47 / 0.60     23.5

**The orthogonalizer is real:** at 0.5 beta the stored assemblies' pairwise
overlap is 0.00-0.05x chance against the control's ~1.5x. **And it costs the
area:** convergence is ~10x slower, so each item's T=8 rounds visit many more
neurons, fill reaches 0.977 by M=24, and M* FALLS to 19.6 -- the ceiling turns
from interference-limited into FILL-limited at a lower M. At 0.7 beta the
item does not converge within T=8 at all. **P0's derived veto is confirmed
where assemblies actually form:** same training, masked 0.984 vs net 0.250.

**Conclusion, adopted:** refraction cannot protect formation in a RECURRENT
area at any strength -- above ~0.75 beta it cancels formation, below it spends
the substrate. The reference confining `RefractedArea` to FEEDFORWARD
conjunction areas, driven by their full input at recall, is the correct
design, not an omission. The protocol dependence (T=8) is noted: longer
episodes would let low-strength items converge but at a still higher fill cost
per item.

---

## CAVEAT (2026-09-04): the hashed selector mis-ranked NEGATIVE drives -- P1/P2 re-measurement pending

The sequence port's first width run found that `topk_select`'s key was the
raw float bits without a sign flip: a NEGATIVE net drive (raw - bias, which
refraction produces) ranked ABOVE every positive one, so a refracted area's
most-biased neurons kept winning and the bias grew without bound. Every
hashed-substrate number in this registration that SELECTED on a refracted
area (P1's bias-masked ranks, P2's wander/churn sweep, the fill-1.0 and
"churns through every neuron above s ~ 0.75 beta" readings, the refracted
capacity-scaling arm) was measured through that defect. The drive-parity
gate passed because it replayed the engine's winners and compared drives;
it never selected. P0 (no refraction) is unaffected. The identity
net_{t+1} - net_t = (beta - s) raw_t is algebra and stands.

Fixed in 1b475fc (order-preserving key; unit test with negatives). The
sweep is re-run with the fixed selector; until then
`REFRACTION-CANCELS-CONVERGENCE`'s empirical clauses are SUSPENDED.

## Re-measurement with the fixed selector (2026-09-04): P2's strength sweep

`seq_refraction_wander.py`, same protocol (n=4000, k=100, p=0.5, beta=0.1,
w_max=20, 240 rounds, 16 brains), `topk_select` order-preserving:

    s/beta     converged   conv round   late stab   fill      vs round-10 at 40 / 60
    0 (ctl)    16/16           4         1.000      0.034     1.000 / 1.000
    0.5        16/16         218         0.921      0.279     1.000 / 0.000   <- relocates ONCE near the clip, then holds
    0.7         6/16         237         0.821      0.911     0.009 / 0.002
    0.8         0/16          --         0.116      1.000
    0.9         0/16          --         0.008      1.000
    0.95        0/16          --         0.006      1.000
    1.0         0/16          --         0.005      1.000
    FF, 1.0     9/16         237         0.928      0.236     0.007 / 0.000   <- drifts, consecutive ~0.9-1.0

What survives: at s >= 0.8 beta a recurrent refracted assembly never
converges and churns through the whole area (fill 1.000) -- the churn was
not the defect. What changes: the intermediate rows. The old "converges
~10x slower at 0.5-0.7 beta with low fill" was the DEFECT locking the
assembly: recent winners carried the largest bias, ranked first, and won
again. With the selector fixed, 0.7 beta already fails to converge for
most brains (6/16, fill 0.91), and 0.5 beta converges, then RELOCATES once
around round 40-60 -- which is the registered P2 prediction (wander when
w_max binds at ~41 rounds) that the defective run had "refuted" at round
11. The feedforward arm drifts slowly rather than holding at 0.63.

Standing: the identity (algebra); the churn above ~0.75 beta (re-measured,
with the transition now somewhere in 0.5-0.7 beta rather than 0.7-0.8);
the P2 registered prediction, now SUPPORTED at 0.5 beta. Retracted: the
"~10x slower convergence, low fill" reading of the intermediate strengths
and the "below the transition it spends the substrate" clause -- Amendment
1's capacity table is being re-run (logs `refcap_*.log`) before any
conclusion there is restated. The theory register entry is revised to this
reading.

## Re-measurement of Amendment 1 with the fixed selector (2026-09-04, POST HOC, labelled)

Same protocol (`seq_capacity_scaling.py --refracted --refracted-factor f
--readout r`), cell G n = 4000, k = 60, T = 8:

    factor  readout   rank-1 M = 8 / 16 / 32     pw/chance     fill M = 16 / 32   M*
    1.0     masked    0.172 / 0.055 / 0.039      0.04 -> 0.74  1.000 / 1.000      8.0   (churn; as before)
    1.0     net       0.102 / 0.059 / 0.029                                        8.0
    0.5     masked    1.000 / 1.000 / 1.000      0.00          0.497 / 0.876      >= 256 CENSORED HIGH (never crossed 0.9)
    0.5     net       0.125 / 0.062 / 0.031      0.00                             8.0   (P0's veto: net cannot read it)
    0.7     masked    0.320 / 0.457 / 0.684      0.01 -> 0.11  0.789 / 1.000      below the bar throughout, RISING with M
    0.7     net       0.133 / 0.066 / 0.035                                        8.0
    control (no refraction, from the registration)     ~1.5 x chance   0.47 / 0.60     23.5

**Amendment 1's conclusion is OVERTURNED.** "Below the transition it spends
the substrate; M* falls to 19.6" was the selector defect. With the selector
fixed, refraction at 0.5 beta with the bias-masked readout stores EVERY
assembly of the grid -- rank-1 1.000 at M = 8, 16, 32 and above the bar to
M = 256 -- with pairwise overlap 0.00 x chance, against a Hebbian control
that ceilings at 23.5. The cost is fill (0.88 at M = 32, 1.0 by M*): the
ceiling is fill-limited far above where interference limits the control.
The net readout (bias subtracted) reads chance at every M: P0's derived
veto stands, and the masked readout -- what the synapses hold with the
intrinsic veto removed -- is the only way to read a refracted memory.
At 0.7 beta the items do not converge within T = 8 (rank-1 rises with M
because later items see a fuller, more refracted area), a regime of its
own.

**Status.** This is a re-measurement of a post-hoc amendment and is
labelled so. It reverses an ADOPTED conclusion ("refraction cannot protect
formation in a recurrent area at any strength"), so nothing is adopted
here; a fresh registration is owed: 20 brains, the M grid extended past
256 to find the fill-limited ceiling, T swept (the converge-within-T
question), and the control on the same run. The register entry's clause
is revised to "below the transition, with the masked readout, capacity is
fill-limited far above the Hebbian ceiling (post hoc; registration owed)".
