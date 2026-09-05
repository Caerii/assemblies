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

## Result (2026-09-04, arm B on the organ fiber, 20 brains, M grid 8..1024)

    cell            n      M*                      fill@M*   M*k/n    distinct@1024   pw/chance near M*
    CTL  T=8     2000     11.3   [8, 16)            0.39      0.34     (0.63 at M=128 for n=4000)   1.8-2.6
    CTL  T=8     4000     83.4   [64, 128)          0.80      1.25                                  1.8 -> 15
    CTL  T=8     8000    306.8   [256, 384)         0.95      2.30                                  1.4 -> 9
    REF  T=8     2000    431.3   [384, 512)         1.00     12.94     1.000 (0.98 at 1024)         0.89 (M=384)
    REF  T=8     4000  >= 1024   censored high      1.00   >= 15.36    1.000 (rank-1 0.994 at 1024) 0.94 (M=1024)
    REF  T=8     8000  >= 1024   censored high      1.00   >=  7.68    1.000 (rank-1 1.000 at 1024) 0.94 (M=1024)
    NET  T=8     4000      8.0   chance (0.125 at M=8)
    T4   T=8->4  4000  does not converge: rank-1 0.225 at M=8, RISING to 1.000 at M >= 768
    T16  T=16    4000    203.2   [192, 256)         1.00      3.05     0.94
    S7   0.7b,16 4000    185.0   [128, 192)         1.00      2.78     0.98     converges: 0.97 at M=8

    R1  CAPACITY     REF(4000) >= 1024 against CTL(4000) 83.4: >= 12x       PASS (as a bound)
                     (n=2000: 431 / 11.3 = 38x, both resolved)
    R2  FILL LAW     fill@M* = 1.00 in every REF cell (first clause holds);
                     M*k/n across n: 12.9 / >=15.4 / >=7.7 -- two cells
                     censored, the ratio cannot be judged                 UNRESOLVED
    R3  ORTHOGONAL   pw/chance 0.89 at the largest uncensored M below
                     M*(2000); 0.94 at M=1024 for the censored cells     FAIL as written
    R4  THE VETO     NET M* = 8, chance at every M                          PASS
    R5  ROUNDS       M*(T8) >= 1024 > M*(T16) = 203, beyond CI              PASS for T8 > T16
                     T4 does not converge (0.225 at M=8): void, reported
    S7  0.7 beta converges given T=16 and holds ~185, above CTL's 83.

**Reading.** R1 and R4 are decisive, R5 holds where it can be judged, R2
is not yet judgeable, and R3 FAILS -- and the failure corrects the
mechanism claim. The stored assemblies are orthogonal (0.00 x chance)
only while the area still has unvisited neurons (M <= 64 at n = 4000);
past fill 1.0 their pairwise overlap returns to chance (0.9 x) -- and
recovery stays perfect anyway. What refraction preserves is not
orthogonality but DISTINCTNESS: `distinct` reads 1.000 through M = 1024
for every REF cell where the control collapses to 0.63 by M = 128 with
pairwise overlap 15 x chance (hub formation, rich-get-richer). Refraction
at half beta is an ANTI-MERGING force, not an orthogonalizer: it stops
repeat winners from becoming hubs, so items overlap at chance like
random subsets yet remain separately recoverable from their cues through
the intrinsic veto. The ceiling that remains is set by something other
than fill or overlap -- at n = 2000 it is 431 = 13 n/k -- and finding it
at n >= 4000 needs the grid extended. Fewer rounds per item raises the
ceiling (T8 > T16) as long as the items still converge; T = 4 does not,
and its curve rising with load says items formed in a fully-visited,
fully-refracted area converge where the first ones did not.

Per the interpretation stated above: R3 failed, so NOTHING IS ADOPTED
from this run; the mechanism claim is rewritten and re-registered below.

## Amendment 1 (2026-09-04, before the extension runs): the grid to 4096, and R3 restated

* R2 needs the censored cells resolved: n = 4000 and 8000 re-run with
  M checkpoints (1024, 1536, 2048, 3072, 4096), 20 brains, T = 8, arm B,
  masked. R2's ratio clause is judged on the resolved values.
* R3 is restated as DISTINCTNESS: `distinct` >= 0.99 at M*(REF) in every
  REF cell, against the control's < 0.7 at its own M*. The orthogonality
  clause is dropped as measured false past fill 1.0.
* Bars R1, R4, R5 stand as read.

### Extension result (2026-09-05, grid to 4096, 20 brains)

    REF  n=4000   M* = 1977.6  [1536, 2048)  resolved   M*k/n 29.7   distinct 1.000 at M*
    REF  n=8000   M* >= 4096   censored high (rank-1 1.000 at 4096)   M*k/n >= 30.7
    R1  at n=4000 resolved: 1978 / 83.4 = 23.7x                                  PASS
    R2  M*k/n = 12.9 / 29.7 / >= 30.7 across n = 2000 / 4000 / 8000: the 2000 -> 4000
        step is 2.3x, outside +/- 25%                                            FAIL as a ratio law at fixed k
    R3' (Amendment 1) distinct >= 0.99 at M*(REF) in every cell (1.000)            PASS
        against the control's 0.63 at its own M* (n=4000)

Neither ceiling is a ratio law at fixed k = 60: the control goes 11 / 83 /
307 and the refracted 431 / 1978 / >= 4096 with n doubling. Whether the
law is in n alone or in n/k is what Amendment 2 asks.

## Amendment 2 (2026-09-05, before running): the k sweep

Cells (n, k), REF (0.5 beta, masked, T = 8) and CTL, 20 brains, arm B,
M grid (8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096):

    (2000, 30)   n/k = 67, same ratio as (4000, 60)
    (4000, 30)   n/k = 133, same ratio as (8000, 60)
    (4000, 120)  n/k = 33, same ratio as (2000, 60)
    (8000, 120)  n/k = 67, same ratio as (4000, 60)

    R6  CONTROL LAW.  If M*(CTL) is a function of n/k alone
        ([[capacity-depends-on-n-over-k]] for assemblies), then
        CTL(8000,120) and CTL(2000,30) are within +/- 25% of CTL(4000,60) = 83,
        and CTL(4000,30) within +/- 25% of CTL(8000,60) = 307, CTL(4000,120)
        of CTL(2000,60) = 11. PREDICTION: PASSES (the assembly law).
        FAIL: any pair outside 2x -> the control ceiling depends on n and k
        separately; report the fit.
    R7  REFRACTED LAW.  The same four equalities for REF against REF(4000,60)
        = 1978, REF(8000,60) >= 4096, REF(2000,60) = 431.
        PREDICTION: uncertain. If R7 passes with R6, one law in n/k sizes
        both memories and refraction multiplies it by a constant (~24x at
        n/k = 67). If R7 fails while R6 passes, refraction changes the
        scaling itself.

## Amendment 3 (2026-09-05, before running): the claim gated on the NUMPY engine

The selector defect was the hashed substrate's; the numpy engine's k-WTA
(`torch.topk` / argpartition on the net drive) has none. Its refracted
mode has never been read at these loads. `refraction_memory_numpy.py`
mirrors the harness on `numpy_sparse`: area n = 2000, k = 60, p = 0.5,
beta = 0.1, w_max = 20, norm_init, no scaling, MATERIALIZED (the explicit
model); items trained by their stimulus alongside recurrence for T = 8
rounds from an inhibited area; the readout is the harness's HALF-CUE
recall -- the first k/2 neurons of the stored assembly set as winners,
T frozen recurrent rounds under `probe()` (no plasticity, no charge), the
bias zeroed for the masked readout and restored after -- ranked against
all stored assemblies; distinctness by exact duplicates. 5 brains, M grid
(8, 16, 32, 64, 128, 256, 512).

    N1  numpy REF M* >= 4 x numpy CTL M*.        FAIL: < 2x.
    N2  distinct >= 0.99 at M*(REF).
    N3  reported, not judged: numpy REF M* against the hashed 431. The two
        substrates differ in the stimulus model (the engine's stimuli into
        a materialized area are zero-or-size, the harness's are Binomial
        counts), so the number is not expected to match; the CLAIM is.

### Amendment 2 -- Result (2026-09-05, 20 brains, arm B, grid to 4096)

    n/k    cell          CTL M*              REF M*                 REF / CTL
     33    (2000,  60)   11.3  [8, 16)        431   [384, 512)        38x
     33    (4000, 120)   11.3  [8, 16)        383   [256, 384)        34x
     67    (4000,  60)   83.4  [64, 128)     1978   [1536, 2048)      24x
     67    (2000,  30)   64.3  [64, 128)     1589   [1536, 2048)      25x
     67    (8000, 120)   88.9  [64, 128)     2230   [2048, 3072)      25x
    133    (8000,  60)  306.8  [256, 384)  >= 4096  censored high   >= 13x
    133    (4000,  30)  262.8  [256, 384)  >= 4096  censored high   >= 16x

    R6  CONTROL LAW.   n/k-matched cells within +/- 25% of each other:
        64 / 83 = 0.77, 89 / 83 = 1.07, 263 / 307 = 0.86, 11.3 / 11.3      PASS
    R7  REFRACTED LAW. 1589 / 1978 = 0.80, 2230 / 1978 = 1.13,
        383 / 431 = 0.89; the n/k = 133 pair both censored at 4096,
        consistent                                                        PASS

**Reading.** BOTH ceilings are functions of n/k alone -- the assembly law
([[capacity-depends-on-n-over-k]]) holds for the refracted memory too --
and refraction multiplies it by a factor that is roughly constant in n/k:
~36x at n/k = 33, ~25x at 67, >= 13-16x at 133 (a bound; the multiplier
may fall slowly with n/k or not, the grid must reach past 4096 to say).
The "superlinearity in n" of the fixed-k sweep was n/k rising: over
n/k = 33 -> 67 -> 133 the control goes 11 -> 83 -> 307 (x7.4, x3.7) and
the refracted 431 -> 1978 -> >= 4096 (x4.6, >= x2.1). Fill is 1.000 at
every REF ceiling and 0.36-0.95 at the control's: the control dies of
merging before the area is full, the refracted memory fills the area and
goes on to a ceiling ~25x higher, distinct 1.000 throughout.

### Amendment 3 -- Result (2026-09-05, numpy_sparse, materialized, 5 brains, M grid to 512)

    numpy REF (0.5 beta, masked)   rank-1 1.000 at EVERY M to 512 on all 5 brains;
                                   distinct 1.000; fill 1.000 from M = 64; pw/chance 0.9-1.0 at 512
                                   M* >= 512 (censored high, all brains)
    numpy CTL                      M* = 34.3 [32, 64) on all 5 brains; distinct 0.55 at 128,
                                   0.14 at 512; pw/chance 10 -> 26 (hub collapse)
    N1  >= 512 / 34 = >= 15x                                                   PASS
    N2  distinct 1.000 at M*(REF)                                             PASS
    N3  reported: numpy REF >= 512 against the hashed 431 (consistent as a
        bound); numpy CTL 34 against the hashed 11 -- the mirror's summed
        stimulus (10 x 6, sd 9.5) is more graded than the harness's
        Binomial(60, 0.5) (sd 3.9), which helps the Hebbian control.

Two things the mirror taught before it passed, both recorded in the script:
the engine's ZERO-OR-SIZE stimulus collapses the control at p = 0.5 (one
tie for every item) and, at p = 0.05, makes the refracted item ROTATE
through its tied connected set (recall 0.5-0.75) -- refraction's benefit
requires GRADED stimulus drive; and a Brain-level projection reads its own
cached winners, so a half-cue recall must drive the engine directly.

## Adopted (2026-09-05)

R1, R4, R5 (where judgeable), R3 as restated (distinctness), R6, R7, N1
and N2 pass; R2 as a ratio law at fixed k fails and is superseded by R6/R7
(the law is in n/k). Register entry `REFRACTION-ANTI-MERGING` (MEASURED):
a recurrent k-WTA area refracted at half beta and read with the bias
masked holds ~25x the Hebbian ceiling; both ceilings are functions of
n/k alone; the mechanism is anti-merging (distinct 1.000 where the
control forms hubs), not orthogonalization (overlap returns to chance past
fill 1.0); the net readout reads chance; fewer rounds per item raise the
ceiling while the items still converge; graded stimulus drive is required.
