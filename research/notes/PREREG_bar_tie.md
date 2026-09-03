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
