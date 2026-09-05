# PREREG: refraction below the transition as an orthogonalizing memory

Registered before running. Owed by PREREG_refraction_capacity.md's
re-measurement (2026-09-04): with the substrate's selector fixed
(1b475fc), a recurrent k-WTA area refracted at 0.5 beta and read with the
bias MASKED held every assembly of a grid to M = 256 (rank-1 1.000 to
M = 32, pairwise 0.00 x chance) where the Hebbian control ceilings at
23.5. That was a post-hoc re-measurement of a post-hoc amendment; nothing
was adopted. This registration asks the question properly.

## Protocol

`seq_capacity_scaling.py` as registered in PREREG_capacity_scaling.md
(inhibit between assemblies; each assembly trained by its own stimulus
alongside recurrence for T rounds; rank-1 recovery of every stored
assembly from its stimulus; M* = the M at which the rank-1 curve crosses
0.9, read from the curve; censored cells reported as bounds), on the
hashed substrate, 20 brains, k = 60, p = 0.5, beta = 0.1, w_max = 20,
M checkpoints (8, 16, 32, 64, 128, 256, 512, 1024). Arm B (norm_init, no
column scaling) is the judged arm; arm G (with scaling) is reported.

    cells   REF   s = 0.5 beta, masked readout, T = 8, n in (2000, 4000, 8000)
            CTL   no refraction (masked = net), T = 8, the same n
            NET   s = 0.5 beta, NET readout, T = 8, n = 4000
            T4    s = 0.5 beta, masked, T = 4,  n = 4000
            T16   s = 0.5 beta, masked, T = 16, n = 4000
            S7    s = 0.7 beta, masked, T = 16, n = 4000

## Bars

    R1  CAPACITY.  M*(REF, n = 4000) >= 4 x M*(CTL, n = 4000).
        FAIL: < 2x.  Between: inconclusive.  A REF cell censored HIGH
        (rank-1 above 0.9 at M = 1024) passes R1 as a bound.
    R2  FILL LAW.  The REF ceiling is fill-limited and scales with n: fill
        at M* >= 0.9 in every REF cell, and M*(REF) k / n within +/- 25%
        across n = 2000, 4000, 8000 (a tiling ceiling, M* proportional to
        n at fixed k). FAIL: M* k / n varies by more than 2x across n.
    R3  ORTHOGONALITY.  Pairwise overlap of the stored assemblies at the
        largest uncensored M below M*(REF) <= 0.1 x chance, against CTL's
        >= 1.0 x chance at its own M*.
    R4  THE VETO.  The NET readout at n = 4000 has M* <= 8 (chance
        recovery): a refracted memory is readable only with the bias
        masked (P0 of PREREG_refraction_capacity.md, re-stated).
        FAIL: NET M* > 16.
    R5  ROUNDS.  Prediction, from the mechanism (each item's T rounds
        visit neurons; fill limits M*): M*(T4) > M*(T8) > M*(T16) at
        n = 4000, each step beyond the pooled CI, PROVIDED the items
        still converge (rank-1 at M = 8 >= 0.9 in every T cell). If T4
        fails to converge (rank-1 at M = 8 < 0.9) the prediction is void
        for that cell and reported.
    S7  reported, not judged: does 0.7 beta converge given T = 16 (rank-1
        at M = 8), and where is its ceiling?

## Interpretation, stated now

* R1-R4 pass -> a recurrent area under refraction at half beta is an
  orthogonalizing memory whose ceiling is TILING (n / k), not
  interference, read through the intrinsic veto: the homeostatic
  mechanism the reference confines to feedforward areas is the lever
  that lifts recurrent capacity by an order of magnitude. Adopt as a
  MEASURED register entry; the numpy engine's refracted mode is then the
  next thing to gate on this claim (its selection has no such defect,
  but its numbers were never read at these M).
* R1 fails -> the re-measurement was a T = 8 / n = 4000 accident; report,
  adopt nothing.
* R2 fails while R1 passes -> the ceiling is not tiling; fit and report.
* R5 inverted (M* rising with T) -> the ceiling is convergence-limited,
  not fill-limited; the mechanism claim is wrong as stated.
