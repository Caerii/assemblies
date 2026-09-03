# PREREG: is M* a function of the ANCHOR-TO-RECURRENCE ratio?

Registered before running. Successor to `PREREG_formation_interference.md`
(F2 passed: anchor 2x -> M* 2.86x; F3 failed: erosion is diffuse, not capture).

## The constraint ledger a mechanism must fit

    C1  M* ~ (n/k)^2 at a fixed operating point     (CS1, verified)
    C2  M* ~ p^-0.6      over p in [0.3, 0.7]        (X1)
    C3  M* ~ beta^-1.3   over beta in [0.05, 0.2]    (X2; 0.05 censored)
    C4  M* ~ s^1.52      anchor size 100 -> 200      (F2)
    C5  erosion is retroactive, diffuse, early-first (F3)

## The hypothesis

C2-C4 are one fact: the forming assembly is a k-WTA fight between its own
stimulus anchor and the trained recurrent pull of everything stored before it.
Anchor strength (s), density (p) and gain (beta) enter only through that ratio,
so they TRADE OFF: an excursion in p or beta can be undone by a computed change
in s, and M* against x = s^1.52 p^-0.6 beta^-1.3 is a single curve.

## Held-out predictions (n=4000, k=100, T=8, w_max=20, arm B, 16 brains)

Baseline M* = 23.5 at (p=0.5, beta=0.10, s=100). Three excursions with
measured uncompensated ceilings, and the anchor that the ratio predicts
restores the baseline (s' = 100 * (M*_base / M*_exc)^(1/1.52)):

    A1  beta 0.10 -> 0.20   M* 8.0   (factor 2.94)   s' = 203 -> run s = 200
    A2  p    0.5  -> 0.7    M* 17.6  (factor 1.335)  s' = 121 -> run s = 121
    A3  p    0.5  -> 0.3    M* 28.8  (factor 0.816)  s' =  87 -> run s =  87

Grid --ms 8,12,16,20,24,28,32,40,48; M* read by `ceiling_from_curve`.

    PASS (ratio law):    each compensated M* within [17.6, 29.4) -- +/-25% of
                         23.5, i.e. closer to the baseline than to its own
                         uncompensated value. A1 is PRIMARY: it is the largest
                         excursion (2.94x) and the one a wrong law misses most.
    FAIL:                A1 stays within 25% of 8.0, or lands past 40.
    Partial:             A2/A3 pass, A1 fails -> p trades against the anchor but
                         beta does not; beta acts elsewhere (e.g. C5's erosion
                         term). Reported as such, not adopted.

The exponents are two- and three-point fits and are used ONLY to place the
compensated cells; no exponent is adopted from this test. What is adopted on
PASS is the mechanism CLASS: capacity is set by the anchor-to-recurrence drive
ratio at formation.
