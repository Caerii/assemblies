# PREREG: are the S5 soft defects k-WTA BAR TIES?

Registered before running. Scoped substrate C fires on the state area (1800
calls, |dw| 1.08e6) and leaves the soft-defect census at exactly substrate A's
30 (`PREREG_substrate_c_homeostasis.md`, Amendment 2 result). Every soft defect
is ONE intruder of 70 (overlap 0.986). The hypothesis recorded there: the
intruder is decided at the k-WTA bar between cells of comparable mass, so no
MASS-based homeostat can move it.

## The discriminator: perturb the tie-break, not the mass

On S5 the substrate is `norm_init=False`, so drives are integer synapse
counts and exact ties at the bar are the common case ([[KWTA-TIE-FRAGILE]]).
Gaussian input noise with std 1e-3 on the STATE area reorders exact ties and
nothing else (the smallest non-tie gap is 1 count). Applied only at CENSUS
time, after training, under `probe()`, so the trained organ is identical
across readouts and only the tie resolution varies.

    S0      soft-pair set from the deterministic census (noise 0)
    S1..S3  soft-pair sets from three noisy censuses of the SAME organ

If the intruders are bar ties, WHICH pairs go soft depends on how the ties
break: the sets move while the count stays near the tie rate. If the
intruders strictly out-mass a block neuron, noise cannot change them and the
sets are identical.

## Bars

Organs: the four groups x seeds (42, 43, 44) -- 12 organs, same build as the
registered census. Jaccard between soft-pair sets.

    T1  TIE (adopt the hypothesis)
        PASS   mean Jaccard(S0, Si) <= 0.5 over organs with |S0| > 0, AND the
               noisy soft counts stay within [0.5x, 2x] of |S0| summed over
               organs, AND hard defects stay 0 under noise
    T2  MASS (refute it)
        PASS   mean Jaccard(S0, Si) >= 0.9 and counts unchanged
    Neither -> reported; a mixed population of ties and out-mass intruders,
    with the fraction estimated from the overlap of the three noisy sets.

Organs with |S0| = 0 contribute only to the hard-defect and count clauses.

## Interpretation, stated now

* T1 -> the census's soft defects are tie-break artifacts of integer drive,
  and no substrate A/B/C/G experiment on this organ should be judged on them
  again without a tie-aware readout; the registered "soft" counts measure
  the TIE RATE, which is a property of integer drive, not of the substrate.
* T2 -> the intruders carry real excess mass that scoped scaling somehow does
  not touch; substrate C's null then needs a different explanation and the
  bar-tie hypothesis is dropped.

---

## First run, and why it did not count (2026-09-03)

The registered script read T2 (MASS) with Jaccard 1.000 -- but its three
"independent" noisy censuses were ONE draw three times: `probe()` saves and
restores the engine rng state on exit, so every census inside it redrew the
same noise. Jaccard 1.00 between repeats was trivial. The comparison against
the deterministic set rested on one realization per organ, which under the
tie hypothesis has P ~ 1/2 of leaving a tied intruder in place. Not evidence.
The script now resets the rng state in place before each census.

**Positive control (the null needs one).** Z60/44 and S5/42, five
INDEPENDENT realizations per noise level, soft pairs compared to the
deterministic census:

    noise   Z60/44 (det: [('4','g1')])         S5/42 (det: [('11','g1')])
    1e-3    unchanged 5/5                      unchanged 5/5
    0.3     unchanged 5/5                      unchanged 5/5
    3.0     n = 3,3,2,0,5 (det pair kept 4/5)  n = 7,9,9,5,6 (det pair kept 5/5)

and a ladder on Z60/44: at std 1.0 the soft pair DISAPPEARS (the intruder is
displaced), at std 3 new pairs appear, at std 10 the census floods (117).
So the noise reaches the k-WTA, the realizations are independent, and at the
registered 1e-3 -- and at 300x that -- the intruder never moves. Under the tie
hypothesis ten unchanged draws per organ is P ~ 1e-3.

**The margin is ONE COUNT.** Surviving std 0.3 and losing at std 1.0 brackets
the intruder's excess drive over the displaced block neuron at about one
integer synapse count -- the smallest possible non-tie in integer drive. That
is why per-round column scaling cannot touch it: scaling equalizes each
neuron's TOTAL incoming mass, and a one-count advantage from the specific arc
assembly that drives this transition survives any renormalization of totals.
The soft defects are SOURCE-SPECIFIC margins, invisible to a total-mass
homeostat, and neither ties nor hubs.

The registered 12-organ test is re-run with independent draws below.

## Result (2026-09-03, independent draws): T2 PASS -- the soft defects are NOT bar ties

Twelve organs, three independent noisy censuses each (`seq_s5_bar_tie.log`,
`seq_s5_bar_tie_results.json`): four organs carry a soft defect, and in every
one of the twelve noisy readouts the soft-pair set is IDENTICAL to the
deterministic census -- mean Jaccard 1.000, count ratio 1.00, zero hard
defects under noise. With the positive control above (noise reaches the
k-WTA; at std 1.0 the intruder is displaced; at std 3 new pairs appear), T2
is met and T1 is not.

**Adopted: the intruders carry a real, SOURCE-SPECIFIC margin of about one
integer count** over the displaced block neuron, from the particular arc
assembly driving that transition. Consequences:

* The bar-tie reading of the substrate-C null is dropped. Scoped column
  scaling fires and moves mass and leaves these defects alone because it
  equalizes each neuron's TOTAL incoming mass, and a one-count advantage from
  one source survives any renormalization of totals. No total-mass homeostat
  (A, B, C, G) should be expected to move this census; the earlier
  "bar ties" hypothesis in `PREREG_substrate_c_homeostasis.md` is corrected
  to "one-count source-specific margins".
* A one-count margin is the resolution limit of integer drive. Per-neuron
  noise of std ~1 count erases the defect (and, at higher std, creates
  others): the soft census sits exactly at the edge of what integer drive can
  resolve, which is the quantitative content of [[KWTA-TIE-FRAGILE]].
* The soft count is therefore a property of the ARC -> STATE fiber's
  fine structure -- which arc neurons happen to wire to which block neurons
  -- and the mechanism that would address it is source-specific
  (fiber-level) normalization, not a homeostat on the postsynaptic neuron.
  Not registered here.
