# PREREG: refraction below the transition as an orthogonalizing memory

> **Status (2026-09-09): adopted.**
> **Finding.** A recurrent k-WTA area refracted at 0.5 beta, read from a
> half cue with the bias masked, stores about 0.40 (n/k)² assemblies in
> regime, about 24 times the Hebbian control. Strength from 0.3 to 0.6 beta
> gives one plateau (Amendment 6). Ending each item's write when its
> winner set repeats raises the ceiling by 24 to 34 percent, and the
> ceiling sits where items stop converging within the round budget
> (Amendments 5 and 6).
> **What changed from the registration.** The title's "orthogonalizing"
> is inaccurate: stored items overlap at chance once the area is full, and
> what refraction does is prevent items from merging while they are
> written. The Hebbian control at T = 16 collapses to M* = 8, so its
> rounds window is attractor dominance rather than anything refraction
> adds (post hoc).
> **Read.** The Result section, then Amendments 4, 5 and 6.
> **Reproduce.** `python -m research.runner capacity-scaling
> --registration research/notes/memory/PREREG_refraction_memory.md --nk 4000:60
> --arms B --refracted --refracted-factor 0.5 --readout masked
> --ms 8,16,32,64,128,192,256,384,512,768,1024,1536,2048,3072,4096 --tag X`
> gives the 1961 cell in about 35 s on one GPU; add `--converge` for the
> gated 2645; drop `--refracted` for the control's 83. Results land in
> `research/results/runs/memory.capacity-scaling/TAG/` with a run record;
> the default seed list is 42..61 (20 brains). The fused kernels need the CUDA build
> environment (see `research/experiments/README.md`).
> **Cite.** `[[REFRACTION-ANTI-MERGING]]`.

![half-cue recall against stored items at n = 4000, k = 60: Hebbian control, refracted, refracted with the convergence gate](../figures/memory_recall_vs_M.png)

![ceiling M* against n/k for both arms, with the 0.40 (n/k)² and 0.017 (n/k)² lines; open markers are out-of-regime cells](../figures/memory_ceiling_vs_nk.png)

*Each point is one (n, k) cell's ceiling on log axes; slope two is a
square law; hollow points are cells below k p >= 3 ln n.*

![recall, the fraction of items converging within the write, and the rounds each write took, against stored items, with the convergence gate on](../figures/memory_gate_U.png)

*How to read it (Amendments 5 and 6): with the gate on, an item's write
ends as soon as its winners stop changing, within a budget of eight
rounds. Top: recall from a half cue. Middle: the fraction of items whose
winners did stop changing within the budget. Bottom: the mean number of
rounds a write took, eight meaning it never settled. Recall fails where
the middle curve falls: the ceiling is the load at which items no longer
converge within the budget.*

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

Artifacts of this amendment: [capacity_scaling_results_ksweep_ctl.json](../../results/memory/capacity_scaling_results_ksweep_ctl.json) (the k sweep, cells at n = 2000, 4000, 8000).

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

## Amendment 4 (2026-09-07, before running): the LAW in n/k, and the censored pair

The k sweep established that both ceilings are functions of n/k alone; it
did not say WHAT function. The resolved refracted cells give

    n/k   cell          REF M*    M* / (n/k)^2
     33   (2000,  60)     431       0.40
     33   (4000, 120)     383       0.35
     67   (4000,  60)    1978       0.44
     67   (2000,  30)    1589       0.35
     67   (8000, 120)    2230       0.50

i.e. an exponent of 2.05-2.2 over the doubling 33 -> 67, and a nearly
constant M* / (n/k)^2 of 0.35-0.50. A LINEAR law in n/k (constant
multiplier on a linear assembly law) would put the n/k = 133 cells at
~3,950; both already exceed 4,096, so linear is dead before this run. The
quadratic reading is a Willshaw-type store: a sparse associative memory
over n^2 synapses with k active per pattern holds ~c (n/k)^2 patterns.

Cells: REF (0.5 beta, masked, T = 8), 20 brains, arm B, (8000, 60) and
(4000, 30), M grid (..., 3072, 4096, 6144, 8192, 12288, 16384). CTL at
these cells is already resolved (307, 263).

    Q1  QUADRATIC.  M*(REF) at BOTH n/k = 133 cells lies in
        [0.35, 0.50] x 133^2 = [6,200, 8,850]; equivalently the exponent
        of the doubling 67 -> 133 lies in [1.65, 2.35].
        PREDICTION: PASSES.
        FAIL LOW  (M* < 6,200 but > 4,096): the exponent falls with n/k --
        the multiplier over the control decays, and the law is sub-quadratic
        at scale; report the exponent, do not fit a power law through three
        points as if it were one.
        FAIL HIGH (M* > 8,850): the exponent rises -- fill-limited effects
        at small n/k depressed the low cells; report.
        CENSORED at 16,384: report the bound; the grid was the limit.
    Q2  MULTIPLIER.  REF / CTL at n/k = 133, both cells, reported against
        the 34-38x at n/k = 33 and 24-25x at n/k = 67. If Q1 passes the
        control is the sub-quadratic one (11 -> 83 -> 307 is x7.4, x3.7) and
        the multiplier RISES with n/k; that is stated now as the reading,
        not as a bar.
    Q3  RESOLUTION.  Both ceilings resolved to a bracket <= 1.5x with an
        interior point; a cliff (no interior point) is reported as such.
    Q4  DISTINCTNESS AT THE CEILING (R3 restated) holds at both cells:
        distinct >= 0.99 at the last M below M*.

Nothing is adopted from this amendment alone: if Q1 passes, the register
entry's claim gains the sentence "the refracted ceiling is ~0.4 (n/k)^2"
and the constant is quoted with its range across the seven cells.

## Amendment 5 (2026-09-07, before running): convergence-gated rounds

R5 found the ceiling convergence-limited from above and below: T = 16
holds 203 where T = 8 holds 1978, and T = 4 does not converge at low load
(rank-1 0.225 at M = 8) yet does at high load (1.000 from M = 768). The
mechanism named there -- an item's rounds past convergence only charge
bias and potentiate what is already formed -- suggests the knob is not T
but WHEN TO STOP: run each item until its winner set repeats, T_max = 8.

Built as `stop_when_stable` on `HashedArea.project` (per brain; the
converged brain's rows go to the fibers as -1, the dead-brain convention;
a brain's rounds up to convergence are bit-identical to the ungated run's,
tested). Harness flag `--converge`; `rounds_used` reported per M.

Cells: (4000, 60), n/k = 67, 20 brains, arm B, grid to 4096: REF gated
(0.5 beta, masked) against REF T = 8 (1978, [1536, 2048)); CTL gated
against CTL T = 8 (83).

    G1  NOT WORSE.  M*(REF gated) >= 1536, the lower edge of the T = 8
        bracket, and rank-1 at M = 8 >= 0.9 (the items converge).
        PREDICTION: PASSES.
    G2  BETTER.  M*(REF gated) > 2048 (above the T = 8 bracket).
        PREDICTION: uncertain -- passes only if items converge in fewer
        than 8 rounds at high load; that is exactly what T = 4's late
        success says, and what G3 measures.
    G3  ROUNDS FALL WITH LOAD.  Mean rounds used per item at M <= 64
        exceeds the mean at M >= 1024. Reported with the fraction of items
        that converge before T_max at each M.
        NULL: if items rarely repeat a winner set inside 8 rounds the gate
        never fires, the gated run equals T = 8 EXACTLY, and this
        amendment is a null -- reported as such, nothing adopted.
    G4  CONTROL, reported: M*(CTL gated) against 83.

Adoption if G1 and G2 pass: the register entry's "fewer rounds per item
raise the ceiling" becomes "an item's rounds should end at convergence";
the harness default stays T = 8 (the registered protocol), gating opt-in.

### Amendment 4 -- Result (2026-09-07, 20 brains, arm B, grid to 16384, 299 s)

    cell          regime (k p vs 3 ln n)   REF M*   bracket          M*/(n/k)^2   CTL    REF/CTL
    (8000, 60)    IN  (30.0 vs 27.0)        6995    [6144, 8192)       0.395      307     23x
    (4000, 30)    OUT (15.0 vs 24.9)        4749    [4096, 6144)       0.27       263     18x

    Q1  QUADRATIC.   (8000, 60): 0.395 in [0.35, 0.50]                    PASS
                     (4000, 30): 0.27, below the window                   FAIL LOW
                     As registered ("BOTH cells"): FAIL LOW.
                     Doubling exponents 67 -> 133: 1.82 in regime
                     ((4000,60) -> (8000,60)), 1.58 out of regime
                     ((2000,30) -> (4000,30)); 33 -> 67 they were 2.05-2.2.
    Q2  MULTIPLIER.  23x and 18x at n/k = 133 against 24-25x at 67 and
                     34-38x at 33: the multiplier does NOT rise; the control
                     is ~quadratic too from 67 on (307 / 83 = x3.7, exponent
                     1.9; CTL / (n/k)^2 = 0.018, 0.017). The reading stated
                     in the amendment ("the control is the sub-quadratic
                     one") was wrong and is withdrawn.
    Q3  RESOLUTION.  [6144, 8192) 1.33x, interior 2; [4096, 6144) 1.50x,
                     interior 10 (at the 1.5x edge)                       PASS
    Q4  DISTINCT.    1.000 at the last M below M*, both cells             PASS

**Reading.** The law in n/k has a REGIME PRECONDITION that the k sweep's
+/- 25% criterion could not see while the pair was censored: the k = 30
cells sit below the harness's own in-degree floor (k p = 15 against
3 ln n = 23-25), and they are the low cells at both ratios ((2000, 30)
0.80 of its pair, (4000, 30) 0.68). In regime the refracted ceiling is
~0.40 (n/k)^2 at n/k = 67 and 133 with a doubling exponent of 1.8-2.2 --
Willshaw-like, and NOT adopted as a power law: the exponent is falling
slowly, and three ratios do not fix it. The out-of-regime cell shows the
T = 4 signature (rank-1 0.49 at M = 8 rising to 1.000 by M = 768): the
recurrent in-degree is too small to converge in 8 rounds at low load.
Nothing adopted; the register entry's "~25x" stands (23x in regime here)
and gains the precondition k p >= 3 ln n once G5 below has run.

R7 restated by this: the n/k = 133 pair, both now resolved, disagree by
0.68 -- R7 holds only within the regime; recorded against the Amendment 2
result, which counted the censored pair as consistent.

Artifacts of this amendment: [capacity_scaling_results_amend4_ref133.json](../../results/memory/capacity_scaling_results_amend4_ref133.json).

### Amendment 5, addendum (2026-09-07, before running): G5, the out-of-regime cell under the gate

If the (4000, 30) cell is convergence-limited from below, then rounds
gated on convergence with a higher ceiling should give its items the
rounds they need at low load without the T = 16 damage at high load.

    G5  (4000, 30), REF gated, T_max = 16, grid to 8192:
        rank-1 at M = 8 >= 0.9 (it converges)  AND  M* within +/- 25% of
        6995 (the in-regime pair) -> the n/k law holds with convergence
        as the precondition, stated as such in the register.
        rank-1 at M = 8 >= 0.9 but M* stays near 4749 -> the cell is
        below the law for a reason other than convergence (in-degree
        itself); k p >= 3 ln n becomes the precondition.
        rank-1 at M = 8 < 0.9 -> 16 rounds do not converge it either;
        report, nothing adopted.

### Amendment 5 -- Result (2026-09-07, (4000, 60) unless stated, 20 brains, arm B)

    arm                  M*      bracket           rank-1 @ M=8   rounds/item (M<=64 | M>=1024)   conv < T_max
    REF gated, T_max 8   2645    [2560, 2816) i4      1.000         7.1 | 7.7  (min 5.8 at M~400)   0.4 -> 0.99 -> 0.00 at M >= 2048
    REF T = 8 (R1)       1978    [1536, 2048)         0.994         8 | 8                            --
    CTL gated, T_max 8   (484)   see G4               0.319         4.4 | 4.0                        1.00 everywhere
    CTL T = 8            83      [64, 128)            ~1.0          8 | 8                            --
    G5 (4000, 30) gated, T_max 16
                          362    [256, 384)           0.956        10.1 | 9-12                       0.99 -> 0.6-0.8
    (4000, 30) T = 8     4749    [4096, 6144)         0.494         8 | 8                            --

    G1  NOT WORSE.   2645 >= 1536 and rank-1 1.000 at M = 8                PASS
    G2  BETTER.      2645 > 2048; bracket [2560, 2816) resolved (four
                     interior points, refined post hoc from [2048, 3072))  PASS  (+34%)
    G3  ROUNDS FALL WITH LOAD.  7.1 at M <= 64 against 7.7 at M >= 1024   FAIL
                     -- the shape is a U: rounds fall to 5.8 at M ~ 400
                     (99% of items converge inside 8) and RISE back to 8
                     past M ~ 1500, where the fraction converging inside
                     8 rounds falls to 0.25 at 1536, 0.01 at 2048 and 0
                     at 3072. The ceiling (2645) sits where items STOP
                     converging: past it an item wanders for all 8 rounds.
    G4  CONTROL.     reported: the gated control converges every item in
                     ~4.4 rounds and does NOT form a recallable memory --
                     rank-1 0.32 at M = 8, 0.20-0.25 to M = 128, rising to
                     0.92 at M = 384 and collapsing after; the "M* = 484"
                     is the downward crossing of a curve that only crossed
                     upward at 384, not a ceiling. Winner convergence is
                     not a formed memory: the control needs the rounds
                     after convergence to potentiate the recurrent weights
                     a half cue completes on.
    G5  (4000, 30), T_max = 16 gated: rank-1 0.956 at M = 8 (it converges,
                     10 rounds per item) and M* = 362 -- NEITHER
                     registered branch: converging the low-load items
                     costs the ceiling 13x (4749 -> 362), the T = 16
                     damage in a new form. The k = 30 cell is not
                     convergence-limited in a way rounds can repair;
                     k p >= 3 ln n stands as the precondition of the n/k
                     law. Nothing adopted for that cell.

**Reading.** Rounds per item are the memory's WRITE BUDGET. Under
refraction each round charges bias and potentiates; the rounds after an
item's winners settle buy nothing for that item and spend the area's
budget (the T = 16 result, and G5). Ending an item at convergence returns
that budget: +34% at n/k = 67, resolved. But convergence inside T_max is
itself load-dependent -- a U in load -- and the ceiling is where it is
lost, so the gate cannot move the ceiling past the point where items no
longer settle in 8 rounds. The control is the opposite regime: its
winners settle in 4 rounds and its memory is not yet written; gating
starves it. So "fewer rounds per item raise the ceiling" is replaced by:
end an item's rounds at convergence under a ceiling T_max the memory
still forms under (8 for the refracted memory; the control has no such
gate).

Artifacts of this amendment: [capacity_scaling_results_amend5_ref_gated.json](../../results/memory/capacity_scaling_results_amend5_ref_gated.json) (REF gated), [capacity_scaling_results_amend5_ref_gated_refine.json](../../results/memory/capacity_scaling_results_amend5_ref_gated_refine.json) (its bracket refinement), [capacity_scaling_results_amend5_ctl_gated.json](../../results/memory/capacity_scaling_results_amend5_ctl_gated.json) (CTL gated, G4), [capacity_scaling_results_amend5_g5.json](../../results/memory/capacity_scaling_results_amend5_g5.json) (G5).

## Adopted (2026-09-07), Amendments 4 and 5

Register entry `REFRACTION-ANTI-MERGING` gains: the regime precondition
k p >= 3 ln n for the n/k law (the k = 30 cells fall 20-32% below their
pairs); in regime the refracted ceiling is ~0.40 (n/k)^2 at n/k = 67 and
133 with doubling exponents 2.1 -> 1.8 (Willshaw-like; NOT a power law
claim); the multiplier over the control is 23-25x at n/k >= 67 (36x at
33), the control being ~quadratic too from 67 on; convergence-gated
rounds with T_max = 8 raise the ceiling +34% (2645 vs 1978, resolved)
and the ceiling is where items stop converging inside T_max; gating
starves the Hebbian control. Harness default stays T = 8 ungated (the
registered protocol); `--converge` is opt-in.

## Amendment 6 (2026-09-09, before running): is the gate's gain a constant, and is strength a lever

Two questions left open by Amendments 4-5, both one cell each.

**The gate at a second n/k.** (8000, 60), n/k = 133, in regime; ungated
T = 8 gives 6995 [6144, 8192). REF gated T_max = 8, 20 brains, grid
(..., 4096, 6144, 8192, 10240, 12288, 16384). A constant +34% predicts
~9,400.

    G6  GAIN IS CONSTANT.  M*(gated) / 6995 in [1.15, 1.55].
        PREDICTION: PASSES (the gate returns the same fraction of the
        write budget at every load scale).
        FAIL LOW  (< 1.15): the gain shrinks with n/k -- the rounds past
        convergence matter less when the area is larger; report the ratio.
        FAIL HIGH (> 1.55): the gain grows; report.
        The convergence-fraction U (G3) is reported at this cell too: if
        the ceiling again sits where items stop converging inside 8, that
        mechanism is stated in the entry as general.

**Strength.** (4000, 60), n/k = 67, ungated T = 8, 20 brains, grid to
4096, masked readout, at 0.3, 0.4, 0.6 beta against the 0.5 beta cell
(1978 [1536, 2048)); 0.7 beta is known not to converge at T = 8 (S7).

    S8  STRENGTH IS A LEVER.  Some strength other than 0.5 has M* >= 1.15
        x 1978 = 2275 with rank-1 at M = 8 >= 0.9.
        PREDICTION: uncertain. 0.5 beta was chosen from a grid of 0.5,
        0.7, 1.0 (PREREG_refraction_capacity); the interior between 0.3
        and 0.7 is unmeasured. Below 0.5 the anti-merging force weakens
        (toward the Hebbian 83); above it the churn transition approaches
        (0.7 does not converge at T = 8). Either monotone shape or an
        interior peak at 0.5 is a FAIL: 0.5 stands, stated as measured.
    S9  CONVERGENCE, reported: rank-1 at M = 8 per strength; the strength
        at which T = 8 stops converging brackets the churn transition
        from below (S7 put it at 0.5-0.7).

Adoption: G6 pass -> the entry's gating sentence gains "constant in n/k
(two cells)". S8 pass -> the entry's strength is restated with the
measured optimum, and the gate is re-run at that strength (post hoc,
labelled). Neither changes the harness default.

### Amendment 6 -- Strength result (2026-09-09, (4000, 60), 20 brains, arm B, T = 8, grid to 4096)

    strength (x beta)   M*      bracket          rank-1 @ M=8   distinct @ M*
    0.3                 1986    [1536, 2048) i3     1.000          1.000
    0.4                 1919    [1536, 2048) i3     1.000          1.000
    0.5 (R1)            1978    [1536, 2048) i3     0.994          1.000
    0.6                 1993    [1536, 2048) i3     0.975          1.000
    0.7 (S7)            does not converge at T = 8; 185 given T = 16

    S8  STRENGTH IS A LEVER.  No strength reaches 2275; the four cells
        agree within 4%                                              FAIL
    S9  CONVERGENCE.  All four converge at M = 8 (0.975-1.000); the
        rank-1 at M = 8 falls with strength (1.000 -> 0.975) and 0.7 is
        past the transition: the churn transition sits in (0.6, 0.7]
        beta at T = 8, one step above S7's bracket.

**Reading.** Refraction is a SWITCH, not a dial. From 0.3 beta to 0.6
beta the ceiling is a plateau at ~1950 -- the same bracket, the same
interior points, the same curves -- while the Hebbian control holds 83.
The ceiling is therefore set by the synaptic memory (the Willshaw-like
~0.40 (n/k)^2 of Amendment 4), and refraction's only job is to keep the
items from merging while it is written; any strength that does that
without crossing the churn transition gives the same memory. 0.5 beta
stands as the protocol's value, now as the middle of a measured plateau
rather than a chosen point. The gate is NOT re-run at another strength
(there is no other strength).

Artifacts of this result: [capacity_scaling_results_amend6_s0.3.json](../../results/memory/capacity_scaling_results_amend6_s0.3.json), [capacity_scaling_results_amend6_s0.4.json](../../results/memory/capacity_scaling_results_amend6_s0.4.json), [capacity_scaling_results_amend6_s0.6.json](../../results/memory/capacity_scaling_results_amend6_s0.6.json) (the 0.5 beta row is R1's cell).

### Amendment 6 -- Gate result (2026-09-09, (8000, 60), 20 brains, arm B, T_max = 8, grid to 16384, 305 s)

    arm                  M*      bracket             ratio    rounds/item (min)   conv < 8: peak -> at M*
    REF gated            8666    [8192, 10240) i2    1.24     5.9 at M ~ 1024     0.98 at 1024 -> 0.00 at 8192
    REF T = 8 (A4)       6995    [6144, 8192)  i2      --     8                   --

    G6  GAIN IS CONSTANT.  8666 / 6995 = 1.24 in [1.15, 1.55]          PASS
        (1.34 at n/k = 67, 1.24 at 133: constant within the bar, and
        the lower of the two at the larger area.)
    The convergence U replicates: 98% of items settle inside 8 rounds at
    M ~ 1024, none at 8192, and the ceiling (8666) sits where they stop
    settling -- as at n/k = 67. Stated in the entry as the mechanism.
    Gated ceilings over (n/k)^2: 0.59 at 67, 0.49 at 133 (doubling
    exponent 1.71); the ungated 0.44, 0.40 (1.82).

Artifacts of this result: [capacity_scaling_results_amend6_g6_gated133.json](../../results/memory/capacity_scaling_results_amend6_g6_gated133.json).

## Adopted (2026-09-09), Amendment 6

G6 passes: the gate's gain is constant in n/k (two cells, +35% and +24%;
the first corrected by Amendment 7),
and its ceiling is where items stop converging inside T_max at both.
S8 fails: strength is a switch (0.3-0.6 beta a plateau at ~1950; the
transition in (0.6, 0.7] at T = 8). Both go into the register entry;
the harness default stays T = 8 ungated at 0.5 beta.

What is NOT established, stated now: the gate at n/k = 33 (small areas)
and out of regime; the plateau's lower edge (below 0.3 beta the force
must fail somewhere between 0 and 0.3); and whether the gated ceiling's
exponent (1.71) keeps falling.

### Post hoc (2026-09-09, labelled): the CONTROL at T = 16

To separate the memory's rounds window from the sequence organ's (which
is refraction + clip, PREREG_s5_cliff_anatomy.md Addendum 5), the Hebbian
control was run at T = 16 on (4000, 60), 20 brains, grid to 512:

    CTL T = 8    M* = 83   [64, 128)
    CTL T = 16   M* = 8    (cliff: rank-1 below the bar from the first grid point)

The control collapses far harder than the refracted memory did (1978 ->
203). So the memory's window is NOT refraction's: it is the Hebbian
store's own -- more rounds per item potentiate the item's synapses
further, the hub collapse arrives sooner, and refraction only delays it.
Two windows, two physics: the memory's is synaptic (potentiation per item
against the store's crosstalk, which the convergence gate trims); the
organ's is intrinsic (bias against the clip). The "one mechanism, two
faces" reading offered in conversation is withdrawn in favour of this.

Artifacts of this post hoc run: [capacity_scaling_results_ctl_T16.json](../../results/memory/capacity_scaling_results_ctl_T16.json).

## Scorecard

Every bar in this registration and its amendments, with the number that
decided it. Cells are (n, k); 20 brains unless stated.

| Bar | Registered | Verdict | Deciding number |
|-----|-----------|---------|-----------------|
| R1 capacity | REF >= 4x CTL at (4000, 60) | PASS | 1961 vs 83, 23.5x |
| R2 ratio law at fixed k | M*(n)/n constant | FAIL | superseded by R6/R7: the law is in n/k |
| R3 orthogonalization | pairwise overlap below chance at M* | FAIL as stated, PASS restated | overlap at chance past fill 1.0; distinct 1.000 |
| R4 masked readout | net readout at chance | PASS | net M* = 8 |
| R5 rounds | T = 16 raises the ceiling | FAIL (inverted) | 203 at T = 16 vs 1961 at T = 8 (A7 correction) |
| R6 control law in n/k | matched cells within 25% | PASS | 64/83, 89/83, 263/307, 11.3/11.3 |
| R7 refracted law in n/k | matched cells within 25% | PASS at 33 and 67 | 1589/1961, 2230/1961, 383/431; at 133 the pair disagrees by 0.68 (one cell out of regime) |
| N1 numpy engine | REF >= 3x CTL | PASS | >= 512 vs 34 (5 brains) |
| N2 numpy distinct | distinct 1.000 at M* | PASS | 1.000 |
| Q1 quadratic | M*/(n/k)^2 in [0.35, 0.50] at both n/k = 133 cells | PASS at (8000, 60), FAIL LOW at (4000, 30) | 0.395; 0.27 (out of regime) |
| Q2 multiplier | reported | -- | 23x and 18x at n/k = 133; the control is quadratic too from n/k = 67 |
| Q3 resolution | bracket <= 1.5x | PASS | [6144, 8192); [4096, 6144) |
| Q4 distinct | distinct >= 0.99 below M* | PASS | 1.000 |
| G1 gate not worse | gated M* >= 1536 | PASS | 2645 |
| G2 gate better | gated M* > 2048 | PASS | 2645 [2560, 2816) |
| G3 rounds fall with load | fewer rounds at high load | FAIL | a U: 5.8 rounds at M ~ 400, 8 at M >= 2048 |
| G4 gated control | reported | -- | no memory forms: rank-1 0.32 at M = 8 |
| G5 out-of-regime cell, T_max 16 | converges and reaches the pair | FAIL | converges (0.956) and M* falls 13x to 362 |
| G6 gain constant in n/k | ratio in [1.15, 1.55] at (8000, 60) | PASS | 1.24 (8666 vs 6995) |
| S8 strength a lever | some strength >= 2275 | FAIL | 1986, 1919, 1961, 1993 at 0.3 to 0.6 beta (A7 correction) |
| S9 convergence | reported | -- | all converge; transition in (0.6, 0.7] beta |
| RP-1 exact reproduction (A8) | surviving legacy values equal | PASS | 3,615 + 3,050 per-seed values equal |
| RP-2 lost ceilings (A8) | inside the reported brackets | PASS | all seven, within 0.1% of the reports |
| RP-3 law as cited (A8) | in-regime M*/(n/k)^2 in [0.35, 0.50] | FAIL | 0.345 and 0.502: the quoted range was rounded inward |
| RP-4 live contrast (A8) | every brain >= 0.50 at M_s, six cells | PASS | minimum 0.938 |
| PE-V instrument (A9) | recall agreement 0.03; M*_real within 10% of A8 | PASS | gap 0.000; 0.979-1.040 |
| PE-0 calibration (A9) | random pair-sharing excess in [0.9, 1.1] | PASS | 0.998-1.001 |
| PE-1 substrate is Willshaw (A9) | M*_random in [0.5, 1.0] (n/k)^2, potentiated [0.40, 0.60] | PASS | 0.69-0.92; 0.49-0.60 |
| PE-2 constant efficiency (A9) | eta in [0.35, 0.70], max/min <= 1.5 | PASS | 0.47-0.56; 1.19 |
| PE-3 pair reuse is the gap (A9) | excess >= 1.5, failure pair load within [0.75, 1.33] of random's | FAIL | 1.43 at (4000, 120) |
| PE-4 transients help (A9) | M*_real > M*_clean | PASS | 1.17-1.47x |
| PE-5 balanced headroom (A9) | M*_balanced >= 1.5 M*_random | FAIL | 1.36 at (8000, 60) |
| PE-S weight sensitivity (A9) | c +- 2 within 30% | FAIL | +66 to +89% at k = 60 |
| PE-R gate replayed (A9) | within 10% of 2645 and 8666 | PASS | 0.980, 1.048 |
| PE-6 gate lowers pair reuse (A9) | eta_gated > eta, lower sharing, failure load within [0.75, 1.33] | FAIL | 1.39 at (8000, 60) |
| WS-V instrument (A10) | own-c ceiling replays A9 within 10% at every cell | FAIL | 16 vs 5043 and 20741 at k = 30 |
| WS-1 .. WS-5 (A10) | -- | VOID | adaptive search assumed one crossing |
| XV instrument (A11) | own-c upper edge within 10% of A9, all cells | PASS | 0.978-1.012 |
| X1 completion needs a strong write (A11) | c = 32 >= 1.5x own c | PASS | 1.9x to 37x; own c 0 at two cells |
| X2 binary completes what it identifies (A11) | ratio >= 0.85 | PASS | 0.88-0.99 |
| X3 separable law at binary write (A11) | within-k 15%, k = 120 >= 1.25x k = 60 | PASS | 0.560-0.563 vs 0.798; 1.42x |
| X4 n/k law belongs to the operating write (A11) | own-c pairs in [0.8, 1.25], c = 32 pairs <= 0.8 | PASS | 1.11, 0.89; 0.68, 0.69 |
| X5 weak writes have a load window (A11) | c = 3 lower edge, capacity above own c | PASS | 213-801 lower; 1.6-1.8x |
| X6 smallest completing write falls with k p (A11) | strictly decreasing in k per n/k level | PASS | 11 / 6 / 4 |
| TV instrument (A12) | beta = 0.1 rank-1 within 10% of A7/A8 | PASS | 0.979-1.040 |
| T1 interior optimum (A12) | completion peaks inside the beta grid | FAIL | (2000, 60): merged recall at the top |
| T2 beta* falls with k p (A12) | strictly decreasing per n/k level | FAIL | 0.209 at (8000, 240), merged recall |
| T3 gamma* within 25% (A12) | all five cells | FAIL | undefined; 1.47 |
| T4 wider is better at the optimum (A12) | best completion / (n/k)^2 rises with k p | PASS | 0.66 / 1.12 / 2.79; 0.34 / 0.94 |
| UV instrument (A13) | beta = 0.1 rank-1 within 10% of A7/A8, new brains | PASS | 0.968-1.058 |
| U1 interior optimum (A13) | distinct completion peaks inside the grid, 7 cells | PASS | all seven |
| U2 the law (A13) | gamma* within 15% of 0.29, 7 cells incl. 2 unseen | PASS | 0.257-0.303 |
| U3 the exponent (A13) | 1/2 collapses best, CV < 0.10 | PASS | 0.052 (raw beta 0.213) |
| U4 wider is better (A13) | best distinct completion / (n/k)^2 rises with k p | PASS | 0.41 to 2.46; 0.34, 0.94 |
| WV instrument (A14) | A13 capacities within 15% on new brains | PASS | 0.970-1.077 |
| W1 recall multiplier (A14) | refracted >= 10x control, each at its best beta | PASS | 14.0-57.3 at ten cells |
| W2 synapse-count form (A14) | C within 15% and tighter than (n/k)^2, per block | FAIL | block B: CV 0.10 vs 0.07 |
| W3 connectivity only as k p (A14) | capacity within 15% and same best beta | FAIL | capacity within 8%; best beta differs |
| W4 learning-rate law at sparse p (A14) | best beta interior at p < 0.5 | FAIL | two at the sweep's top |
| DV instrument (A15) | T = 8 replays A14 within 15% | PASS | 486, 1575 |
| D1 old effect (A15) | beta 0.1: T = 16 below T = 8 | PASS | 86 vs 287; 207 vs 977 |
| D2 rescaled write transfers (A15) | law point within 15% across T | FAIL | T = 12, 16 far below |
| D3 best rate as 1/T (A15) | T ln(1+beta*) within 20% | FAIL | T = 16 optimum at the grid's bottom |
| D4 depth does not matter at the optimum (A15) | best capacity within 15% across T | PASS | within 9% at both cells |
| SV instrument (A16) | A14 capacities within 15% on new brains | PASS | 0.974-1.018 |
| S1 interior optimum (A16) | all six cells | PASS | all six |
| S2 noise law (A16) | gamma_noise within 15%, tighter than gamma_plain | PASS | CV 0.081 vs 0.258 |
| S3 dense side (A16) | beta*(p .75) / beta*(p .5) <= 0.85 | PASS | 0.707 (predicted 0.71) |
| S4 capacity p-invariant (A16) | k p = 30 bests within 15% | PASS | 436-497 |
| TV instrument (A17) | (4000, 60, 0.5) within 15% of 1616 | PASS | 1589 (0.983) |
| T1 threshold fraction above floor (A17) | beta*/theta within 15% of 0.20 at k p = 30, 40, 80 | PASS | 0.180, 0.189, 0.180 |
| T2 threshold fraction below floor (A17) | >= 1 resolved; every resolved within 25% of 0.20 | FAIL | p 0.5: 0.19-0.22; p 0.05: 0.21, 0.27, 0.30; PNAS cell never completes |
| OV instrument (A18) | (4000, 60, 0.5) within 15% of 1589 | PASS | 1597 |
| O1 universal onset (A18) | onset/theta in [0.13, 0.21] at all ten, CV <= 0.10 | PASS | 0.160-0.180, CV 0.041 |
| O2 dense onset load-assisted (A18) | onset window opens at >= 64 items, p = 0.5 | PASS | 158-1098 |
| O3 sparse onset immediate (A18) | no lower edge at p = 0.05 | FAIL | none / 3 items |
| O4 dense optimum at onset (A18) | best/onset <= 2^(1/4), p = 0.5 | FAIL | 1.00-1.06 at 5 cells; 1.1897 at (8000, 120) |
| O5 sparse optimum above onset (A18) | best/onset >= 1.4, p = 0.05 | PASS | 1.78, 1.59 |
| XV instrument (A19) | A17 capacities within 15% | PASS | 1.01, 1.12 |
| X1 flat below floor (A19) | k = 10-40 within 15% of mean | FAIL | k = 14 at -16% |
| X2 linear in n below floor (A19) | x1.6-2.5 per doubling | FAIL | x2.98, x2.13 |
| X3 falling above floor (A19) | strictly decreasing, k=160 <= 0.8 k=56 | FAIL | rises 56 -> 80; 0.76 |
| R1 gap grows as fan-in falls (A20) | Spearman(k p, R) <= -0.7 | PASS | -0.883 |
| R2 recall needs fan-in (A20) | rank-1 >= 32 all; distinct 0 at kp 1, >= 32 at kp 8 | PASS | 114-207; 0; 40 |
| R3 divergence large (A20) | R(k 10) >= 3 R(k 160) | FAIL | 1.78x |
| DV instrument (A21) | (4000, 60, 0.5) within 15% of 1597 | PASS | 1608 |
| D1 degree-law prediction (A21) | new cells within 25% of 0.0135 d^1.51 | FAIL | 7 of 8; d = 6000 at 1.257 |
| D2 in-degree alone (A21) | equal-d cells within 20% of mean, n/k 2-4x | PASS | max 11% |
| D3 degree-law exponent (A21) | slope in [1.35, 1.65] | FAIL | 1.669 |
| A22 recognition at own optimum | -- | VOID | chance rank-1 at M = 2 armed the stop rule |
| QV instrument (A23) | recall at (4000, 40, 0.5) within 15% of 1473 | PASS | 1339 |
| Q1 recognition outruns recall (A23) | ratio >= 2 at every cell | PASS | 15.0-86.3 |
| Q2 gap widens as fan-in falls (A23) | Spearman <= -0.9 and k=10/k=160 >= 5 | FAIL | -0.90; 4.0x |
| Q3 recognition wants a weaker write (A23) | recognition rate <= half recall's | PASS | 1/40-1/6 |
| CV instrument (A25) | round write within [0.75, 1.10] of A18 | PASS | 0.89, 0.93, 0.98 |
| W1 deferral abolishes the memory (A25) | deferred capacity 0 at every rate, every cell | PASS | 0 |
| W2 burst write stores nothing (A25) | burst capacity 0 at every rate, every cell | PASS | 0 |
| W3 burst gating costs capacity (A25) | online_burst <= half round's | PASS | 0 (abolished) |
| W4 deferred write stores the trajectory (A25) | next >= 0.4, >= 5x same, own <= 0.1; round own >= 0.5 | PASS | next 0.64-0.79; own 0.002-0.022; round own 0.80-0.98 |
| S1 deferred write is a sequence memory (A26) | peak replay >= 0.9, own <= 0.1 every rate | PASS | 1.00; own <= 0.042 |
| S2 adaptation switches the memory type (A26) | online s=1.5: own <= 0.1, peak >= 0.7; s=0.5: own >= 0.5 | PASS | own <= 0.052, peak 1.00; control 0.71-0.85 |
| S3 sequence capacity follows n/k (A26) | same n/k within 25%, same d >= 2x | FAIL | 0.48; 1.68 |
| S3d sequence capacity follows in-degree (A26) | same d within 25%, same n/k <= 0.5 | FAIL | 1.68; 0.48 |
| L1 Hebbian sequence limit is Willshaw's (A27) | L_H / (p (n/k)^2) in [0.06, 0.12] | PASS | 0.063-0.106 |
| L2 sequence limit follows n/k (A27) | equal-n/k pairs within 30% | FAIL | 22%; 67% |
| L3 Hebbian failure is a cliff (A27) | ensemble >= 0.9 at L_H/sqrt2, <= 0.2 at sqrt2 L_H | FAIL | 3 cells; all-or-none per brain (post hoc) |
| L4 refraction extends the limit (A27) | L_R >= 2 L_H every cell | FAIL | 2.2-31x at n/k <= 67; 0.33, 0.57 at n/k = 133 |
| D1 the sequence tiles the area (A28) | fresh >= 0.95 to 0.9 n/k, <= 0.10 from 1.1 n/k | PASS | every cell |
| D2 replay breaks on the deadline (A28) | >= 80% of breaks within 5% of j n/k | PASS | 49 of 51 (96%) |
| D3 recovering bias removes it (A28) | reset >= 0.9 every cell, +0.3 at n/k = 133 | PASS | 1.00 everywhere; 0.26, 0.37 refracted |
| B1 recovery beats both ends (A29) | best >= 2x max(Hebbian, cumulative) every cell | FAIL | 1.27 at (2000, 60); 4.2-8.0 elsewhere |
| B2 the optimum is interior (A29) | best tau neither 0 nor 512 | PASS | 32, 64, 64, 64 |
| B3 upper edge scales with n/k (A29) | edge in [0.5, 4] n/k | FAIL | 15 n/k at (2000, 60); 1.9-3.8 elsewhere |
| B4 best limit grows as (n/k)^2 (A29) | best/(n/k)^2 spread <= 1.5 | FAIL | 0.14-0.61; follows the in-degree (post hoc) |
| C1 refraction codes shared elements apart (A30) | overlap <= 0.05, >= 18/20 right, gap 0 | PASS | <= 0.016; 20/20 |
| C2 without refraction the context is lost (A30) | m = 16: overlap >= 0.5, <= 10/20 right | PASS | 0.83-0.87; 0/20 |
| C3 separation is largely recency (A30) | overlap >= 0.15 after 600, every m | FAIL | 0.09-0.11 at m = 1; 0.67-0.70 at m = 16 |
| R1 forward write is forward only (A31) | r = 0: forward >= 0.9, backward <= 0.1 | PASS | 1.00; 0.00 |
| R2 one reverse count too weak (A31) | r = 1: backward <= 0.5 | PASS | 0.00-0.01 |
| R3 two counts + LRI both ways (A31) | r = 2, 3: all four reads >= 0.9 | FAIL | backward 1.00; forward 0.02-0.29 |
| R4 no LRI no direction (A31) | r = 2: forward_masked <= 0.5 | PASS | 0.01 |
| Q1 balanced chain both ways (A32) | 2 + 3: four reads >= 0.9 | PASS | 1.00 everywhere |
| Q2 stronger direction wins (A32) | 2 + 2 forward only; 0 + 2 backward only | PASS | back 0.05-0.33; forward 0.09-0.23 |
| Q3 no LRI no direction, balanced (A32) | 2 + 3 forward without LRI <= 0.5 | PASS | 0.02 |
| H1 sequences of sequences (A33) | (8, 6) link 2: every plan whole >= 0.9 | PASS | 1.00 both cells |
| H2 the links carry it (A33) | link 0: starts <= 0.1 everywhere | PASS | 0.00 |
| H3 plans sharing a run stay apart (A33) | shared-run plans chain, whole >= 0.9 | PASS | 1.00 |
| H4 links give way first, larger pair relieves (A33) | starts < chain - 0.1; larger +0.1 | PASS | 0.78 vs 0.96; 1.00 |
| N1 many sequences share one budget (A34) | budget in [0.6, 1.6] of A29 | PASS | 1.08; 0.81 |
| N2 small noise harmless (A34) | nu 0.05: >= 18/20 full | PASS | 20/20 both cells |
| N3 larger area doubles noise horizon (A34) | median(8000) >= 2 median(4000) at nu 0.1 | PASS | 232 vs 39 |
| N4 cue may be a quarter wrong (A34) | eta 0.25: >= 0.9 | PASS | 1.00 |
| O1 engine equals an independent oracle (A35) | lockstep, counts equal | PASS | ~2,900 rounds, 1 float32 near tie |
| A33 H1-H4 on the corrected divisor (A35) | as registered in A33 | PASS | max change 0.010 |
| CV instrument (A24) | (4000, 60, 0.5) within 15% of 1597 | PASS | 1613 |
| F1 onset sharpens (A24) | sd non-rising (10%), sd(16000) <= 0.6 sd(2000) | FAIL | 0.0031-0.0041 theta, at the grid's resolution at every n |
| F2 onset converges (A24) | means in [0.13, 0.21], 16000/8000 within 10% | PASS | 0.163-0.170; 1.029 |
| F3 read-out slows at onset (A24) | settle(onset) >= 1.5 settle(half-octave up) | FAIL | 0.97-1.39; 42-88% never settle (tie-fragile instrument) |


## Runner migration reproduction (2026-09-10)

The shared runner reproduced the figure control at (4000,60), B, T8,20 seeds,
all 11 recorded load checkpoints. Its [results](../../results/runs/memory.capacity-scaling/migration-capacity-20260910-v3/results.json)
and [comparison receipt](../../results/runs/memory.capacity-scaling/migration-capacity-20260910-v3/comparison.json)
match all 885 compared scalar values, including per-seed metrics and aggregate
ceiling fields. The bracket remains [64,128), too broad to treat interpolated
83.4 as a resolved ceiling. This is migration evidence, not a new adoption.
The original artifact has no run record; reconstruction inputs and their source
are documented in docs/reviews/whole-codebase/VALIDATION.md.


The subsequent [record-consumption replay](../../results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/results.json)
([comparison](../../results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/comparison-v2.json))
also matched all 885 comparisons after arm settings, device and distinctness bars
were changed from implicit globals to required execution inputs from the run
record. The registered defaults and scientific interpretation are unchanged.

## Amendment 7 (2026-09-11, before running): paired sensitivity provenance

This is a migration and instrument-sensitivity check for the already adopted R1
contrast, not a new scientific claim. Protocol version 3 executes both conditions
inside one run record with the same ordered seeds and restarted measurement-sample
stream. Each condition has its own complete configuration and organ-semantics
profile. They may differ only by refraction: CTL has strength zero; REF has
strength 0.5 beta. Both use arm B, masked ungated readout, `(n,k)=(4000,60)`,
T=8, p=0.5, beta=0.1 and seeds 42..61. The M grid is
8,16,32,64,128,192,256,384,512,768,1024,1536,2048,3072,4096.

    A7-M1  REPRODUCTION. Every retained per-seed metric at common legacy
           checkpoints is exactly equal to the corresponding figure artifact;
           ceiling fields agree under the version-3 implementation. Any changed
           observation is a failed migration and cannot update the register.
    A7-S1  SENSITIVITY. At M=128, every paired brain has
           rank1(REF)-rank1(CTL) >= 0.50. Any seed below 0.50 fails the
           sensitivity check. This bar is frozen before the paired rerun and is
           intentionally below the historical minimum; it tests a live contrast,
           not the 4x aggregate-capacity adoption bar.
    A7-S0  TRUE NEGATIVE. Replacing REF's vector with CTL's vector must fail
           A7-S1. This is a validator control, not another GPU condition.

Reproduce with `python -m research.runner capacity-scaling --registration
research/notes/memory/PREREG_refraction_memory.md --nk 4000:60 --arms B
--compare-refraction --refracted-factor 0.5
--ms 8,16,32,64,128,192,256,384,512,768,1024,1536,2048,3072,4096 --tag X`.
The run remains UNJUDGED until A7-M1 and A7-S1 are evaluated and recorded below.

### Amendment 7 result (2026-09-11)

The immutable paired [run](../../results/runs/memory.capacity-scaling/refraction-paired-sensitivity-20260911/results.json)
and its [comparison receipt](../../results/comparisons/refraction-paired-sensitivity-20260911.json)
pass both bars. A7-M1 compares 2,090 retained scalar observations against the
control and refracted figure artifacts with no differences. A7-S1 passes on all
twenty seeds at M=128: the minimum paired REF-minus-CTL rank-1 difference is
0.96875 (mean 0.9921875), above the frozen 0.50 bar. Substituting CTL for REF
fails the validator's constructed A7-S0 control.

The exact replay also resolves a transcription/estimand inconsistency in the
registration. The retained refracted figure artifact, the version-3 recomputation
and the exact GPU replay all give M*=1961.400398770613, not 1977.6. The control is
83.41525726478883, so the multiplier is 23.51x. This does not change R1, its
[1536,2048) bracket, the ~0.40(n/k)^2 description, the strength plateau or any
adopted verdict. Current summaries and the register use 1961; earlier dated result
paragraphs retain 1978 as the historical report corrected here. Against the
corrected ungated value, the gated 2645 result is about +35%, still inside G6's
registered interval.

## Amendment 8 (2026-09-30, before running): the law's cells replayed with provenance

Registered before any run of this amendment. The register records the gap
this closes: `REFRACTION-ANTI-MERGING` states the ~0.40 (n/k)^2 law, but the
capacity grids it is fitted to predate the shared runner. Their files record
no seeds, no k, no readout and no refraction strength, and they key cells by
n alone, so two cells at one n overwrote each other: the per-seed data of
seven of the law's reported ceilings does not exist (the refracted k sweep
was never written at all). Amendment 7 replayed one cell, (4000, 60), and
reproduced 2,090 retained values exactly, which establishes that the hashed
substrate is deterministic and that the legacy seeds are 42 to 61. This
amendment replays the remaining six cells the same way.

It is a provenance and reproduction check, not a new claim. Nothing is
adopted from it; its outcome decides whether the numbers the register cites
are reproducible, and the register is corrected if they are not.

### Protocol

Protocol version 3 through the shared runner (`--compare-refraction`): the
Hebbian control and the refracted memory (strength 0.5 beta, masked
readout) in one record, same ordered seeds 42 to 61, arm B (norm_init, no
column scaling), T = 8, p = 0.5, beta = 0.1, w_max = 20, the runner's
default measurement sampling. Three runs, shaped by what evidence survives,
because the comparator checks a run against a legacy file only when both
hold the same cells:

    L1  (2000, 30), (4000, 120), (8000, 120)
        M grid 8,16,32,64,128,192,256,384,512,768,1024,1536,2048,3072,4096
        (Amendment 2's grid). Control survives in
        capacity_scaling_results_ksweep_ctl.json; refracted does not.
    L2  (8000, 60), (4000, 30)
        M grid as L1 plus 6144,8192,12288,16384 (Amendment 4's grid).
        Refracted survives in capacity_scaling_results_amend4_ref133.json;
        the control (reported 306.8 and 262.8 in Amendment 2) does not.
    L3  (2000, 60)
        M grid as L1. Neither condition survives (reported 11.3 and 431).

Each run from a worktree pinned at the commit that registers this
amendment, one GPU job at a time.

### Bars

    RP-1  EXACT REPRODUCTION where legacy data survives. Every retained
          per-seed metric at every common checkpoint equals the legacy
          file: L1's control against ksweep_ctl (comparison kind
          capacity-control), L2's refracted memory against amend4_ref133
          (kind capacity-refracted), with the ceiling fields. Receipts are
          retained under research/results/comparisons/. Any differing value
          fails, and that cell's legacy number is then not reproducible.
    RP-2  REPORTED CEILINGS where legacy data was lost. Each replayed M*
          lies inside the bracket this registration reported:
              control   (2000, 60) [8, 16)       (8000, 60) [256, 384)
                        (4000, 30) [256, 384)
              refracted (2000, 60) [384, 512)    (4000, 120) [256, 384)
                        (2000, 30) [1536, 2048)  (8000, 120) [2048, 3072)
          The point values are reported beside the reported ones
          (11.3, 306.8, 262.8; 431, 383, 1589, 2230). A point value
          outside 1% but inside its bracket is a transcription discrepancy,
          corrected in the text and the register, not a failure; Amendment
          7 found one such (1978 against 1961).
    RP-3  THE LAW AS CITED. From the replayed values, M*/(n/k)^2 of the
          refracted memory at the in-regime cells (2000, 60), (4000, 120),
          (8000, 120) and (8000, 60) lies in [0.35, 0.50], with (4000, 60)
          taken from Amendment 7. The out-of-regime cells, (2000, 30) and
          (4000, 30), are reported and not judged, as in Amendment 4.
    RP-4  A LIVE CONTRAST AT EVERY CELL. At M_s, the first grid point at or
          above four times the reported control ceiling, every one of the
          twenty paired brains has rank1(refracted) - rank1(control)
          >= 0.50:
              (2000, 60) and (4000, 120): M_s = 64
              (2000, 30) and (8000, 120): M_s = 384
              (8000, 60) and (4000, 30):  M_s = 1536
          Four times the control's ceiling is past its cliff
          ([[CAP-CLIFF]]) and below every refracted ceiling, so this asks
          whether the treatment still moves the organ at each cell, the
          question A7-S1 asked at one cell.
    RP-0  TRUE NEGATIVE. Substituting each cell's control vector for its
          refracted vector must fail RP-4.

The run is UNJUDGED until RP-1 to RP-4 are evaluated and recorded below.

### Amendment 8 result (2026-09-30)

Three runs from a worktree pinned at 370cc403, 20 paired brains each, seeds
42 to 61: [L1](../../results/runs/memory.capacity-scaling/capacity-law-replay-L1-20260930/results.json),
[L2](../../results/runs/memory.capacity-scaling/capacity-law-replay-L2-20260930/results.json),
[L3](../../results/runs/memory.capacity-scaling/capacity-law-replay-L3-20260930/results.json)
(4, 11 and 2 minutes). Receipts:
[L1 control](../../results/comparisons/capacity-law-replay-L1-20260930.json),
[L2 refracted](../../results/comparisons/capacity-law-replay-L2-20260930.json).

    RP-1  EXACT REPRODUCTION                                        PASS
          L1 control vs ksweep_ctl: 3,615 per-seed values equal;
          L2 refracted vs amend4_ref133: 3,050 equal; ceiling fields
          equal. With Amendment 7's 2,090, every surviving legacy value
          behind the law is reproduced exactly.
    RP-2  REPORTED CEILINGS where the legacy data was lost          PASS
                            replayed   reported   bracket
          control (2000,60)    11.31      11.3    [8, 16)
          control (8000,60)   306.79     306.8    [256, 384)
          control (4000,30)   262.76     262.8    [256, 384)
          REF     (2000,60)   431.32     431      [384, 512)
          REF     (4000,120)  383.15     383      [256, 384)
          REF     (2000,30)  1588.53    1589      [1536, 2048)
          REF     (8000,120) 2229.73    2230      [2048, 3072)
          Every point value within 0.1% of its report.
    RP-3  THE LAW AS CITED, window [0.35, 0.50]                     FAIL
          M*/(n/k)^2 in regime: (2000,60) 0.388, (4000,120) 0.345,
          (4000,60) 0.441 (A7), (8000,120) 0.502, (8000,60) 0.393.
          Two cells fall just outside: 0.345 and 0.502.
    RP-4  LIVE CONTRAST AT EVERY CELL, every brain >= 0.50          PASS
          minimum paired REF - CTL rank-1 at M_s: 0.969, 0.969, 0.938,
          0.969, 1.000, 1.000 (cells in the registered order).
    RP-0  TRUE NEGATIVE (control substituted for refracted)         PASS

**Reading.** Every number the capacity law is fitted to is reproduced under
the shared runner, with source, environment and parameters recorded. Where
legacy per-seed data survives the reproduction is exact, and where it was
lost the replay lands on the reported value. The substrate is deterministic,
so the lost files were a storage failure, not a measurement one. RP-3 fails
as registered because the window was taken from the register's quoted range,
"0.35 to 0.50", whose endpoints were these same two cells ROUNDED inward
(383/33.3^2 = 0.3447, 2230/66.7^2 = 0.5017). The bar is not moved after the
data. The law's in-regime range as measured is 0.345 to 0.502 (n/k)^2, and
the register, the notes map and the papers plan now quote that. Out of
regime, (2000,30) is 0.357 and (4000,30) 0.267, as Amendment 4 reported.
REF/CTL multipliers: 38.1x and 33.9x at n/k = 33, 24.7x and 25.1x at 67,
22.8x and 18.1x at 133.

The register entry's provenance gap is narrowed accordingly, and its
retained sensitivity checks now cover all seven cells. Still without runner
provenance: the gated grid (Amendment 5), the strength grid (Amendment 6),
the T sweep and the numpy mirror.

## Amendment 9 (2026-09-30, before running): how much of the substrate's capacity each code realises

Registered before any run of this amendment. It asks what the law's constant
MEANS. The ceiling is ~0.35 to 0.50 (n/k)^2 assemblies; the classical
associative memory (Willshaw) stores ln 2 (n/k)^2 = 0.69 (n/k)^2 random
patterns at its information optimum, where half its synapses are
potentiated. Whether that resemblance is a coincidence is decidable: write
patterns that are NOT the network's own into the same circuit and read them
the same way.

**What was seen before registering (exploratory, labelled, nothing claimed).**
A scratch diagnostic on 4 brains at (2000, 60), (4000, 120) and (4000, 60),
32 recalls per checkpoint, grids 1.33 to 1.5x apart, gave, in (n/k)^2 units:
the refracted memory 0.375 / 0.355 / 0.43; its own assemblies written
without the transient winners 0.31 / 0.26 / 0.37; independent random
k-subsets 0.75 / 0.65 / 0.83 with about half the synapses potentiated;
usage-balanced subsets 1.66 / > 1.84 / 1.33. The refracted assemblies'
neuron usage was MORE even than random (variance/mean 0.04 to 0.07), their
mean pairwise overlap below chance, yet their pair sharing (below) was 1.9x
random. The windows and bars below were set with these numbers in view;
this amendment is the confirmatory test on 20 brains at every cell of the
law, with the bars fixed now.

### Protocol

`research/experiments/memory_pattern_efficiency.py` through the shared
runner (`python -m research.runner pattern-efficiency`), seeds 42 to 61,
one run, all seven cells of the law, one GPU job, from a worktree pinned at
the commit that registers this amendment. Every variant is read through the
SAME circuit: the hashed presence matrix of the cell's brains, norm_init's
in-degree division, the engine's potentiation table with the w_max = 20
clip, a half cue (the first k/2 stored winners), 8 frozen k-WTA rounds with
the bias masked, and the capacity study's rank-1 (argmax overlap against
every stored item, ties to the lowest index), 32 sampled items per brain
per checkpoint, the sample a function of M alone so variants at one M are
read on the same items. The variants:

    real         the refracted AssemblyMemory (0.5 beta, T = 8, ungated),
                 the capacity study's store and stimuli, read from its own
                 count matrix
    gated        the same with the convergence gate, at (4000, 60) and
                 (8000, 60)
    clean        the real assemblies, synapses rebuilt from them alone
    clean_gated  likewise for the gated assemblies
    random       independent uniformly random k-subsets
    random_cm2   random, with c - 2 counts per pair
    random_cp2   random, with c + 2
    balanced     each item takes the k least-used neurons, random tie-break

A clean write puts c counts on every internal pair of every item, where c
is the rounded mean count on the present internal pairs of the condition's
first stored item across brains (what one item's write puts on its own
assembly; 5 at k = 60 and 6 at k = 120 in the exploration). Checkpoints are
the geometric grid 16, 24, 32, 48, ... restricted to a window in multiples
of Amendment 8's ceiling (A7 for (4000, 60); Amendments 5 and 6's 2645 and
8666 for the gated cells): real, gated and clean [0.25, 3], random
[0.5, 8], balanced [1, 12]. M*_v is where the ensemble-mean rank-1 of
variant v crosses 0.5 (`ceiling_from_curve`). Statistics, from the
unclipped count matrix N = X^T X of each variant's patterns:

    potentiated   fraction of present synapses with a nonzero count (the
                  real and gated variants read their actual count matrix)
    pair sharing  sum_{i != j} N_ij (N_ij - 1) / (M k (k - 1)): the mean
                  number of other items sharing an internal synapse pair
                  with an item; independent subsets expect
                  (M - 1) k (k - 1) / (n (n - 1)); "excess" is the ratio
    eta           M*_real / M*_random, the PATTERN EFFICIENCY

Both statistics are interpolated at a variant's M* linearly in log2 M. A
ceiling is RESOLVED when its curve starts above 0.5 and crosses inside the
window. The in-regime cells (2000, 60), (4000, 120), (4000, 60),
(8000, 120), (8000, 60) are judged; (2000, 30) and (4000, 30) are reported
and not judged, as in Amendment 4. `evaluate()` in the module applies the
bars below; it is tested to fail when the random memory is substituted for
the real one.

### Bars

    PE-V  THE INSTRUMENT (gates every other bar). (a) At every real
          checkpoint the module's own recall and this study's dense recall
          give ensemble-mean rank-1 within 0.03. (b) M*_real is within 10%
          of Amendment 8's value (A7's at (4000, 60)) at every cell. A
          failure VOIDS the study; nothing below is judged.
    PE-0  CALIBRATION AND TRUE NEGATIVE. Pair-sharing excess of the random
          variant lies in [0.9, 1.1] at every cell and checkpoint, and the
          random curve substituted for the real one fails PE-2.
    PE-1  THE SUBSTRATE IS A WILLSHAW MEMORY. At every judged cell M*_random
          is resolved, M*_random / (n/k)^2 lies in [0.5, 1.0] (ln 2 = 0.69),
          and the potentiated fraction at M*_random lies in [0.40, 0.60].
    PE-2  CONSTANT PATTERN EFFICIENCY. At every judged cell eta lies in
          [0.35, 0.70], both ceilings resolved, and the largest eta is at
          most 1.5 times the smallest.
    PE-3  PAIR REUSE ACCOUNTS FOR THE GAP. At every judged cell the real
          assemblies' pair-sharing excess at M*_real is at least 1.5, and
          their absolute pair sharing at M*_real is within [0.75, 1.33] of
          the random patterns' at M*_random -- both memories fail at the
          same PAIR-level load.
    PE-4  THE TRANSIENTS HELP. M*_real > M*_clean at every judged cell.
    PE-5  BALANCED HEADROOM. M*_balanced >= 1.5 M*_random at every judged
          cell (a balanced curve still above 0.5 at its last checkpoint
          counts as that lower bound).
    PE-S  WEIGHT SENSITIVITY. M*_random at c - 2 and at c + 2 each lie
          within 30% of M*_random at c, at every judged cell.
    PE-R  THE GATE REPLAYED. M*_gated within 10% of 2645 at (4000, 60) and
          of 8666 at (8000, 60) (Amendments 5 and 6, legacy records).
    PE-6  THE GATE LOWERS PAIR REUSE. At both gated cells: eta_gated > eta;
          at the grid point both conditions share nearest M*_real, the
          gated assemblies' pair sharing is below the ungated ones'; and
          the gated pair sharing at M*_gated is within [0.75, 1.33] of the
          random patterns' at M*_random.

### Interpretation, stated now

* PE-1 and PE-2 pass: the law is the substrate's Willshaw capacity times a
  constant pattern efficiency, and P1 states it that way: random patterns
  on this circuit store ~ln 2 (n/k)^2, the refracted memory realises eta
  of it, the Hebbian control ~0.02 (Amendment 8's control ceilings over
  this study's M*_random, reported, not judged).
* PE-1 fails: the resemblance to Willshaw is not established on this
  circuit; report M*_random and the potentiated fraction as measured.
* PE-2 fails on range or constancy: no efficiency constant is quoted;
  report eta per cell.
* PE-3 passes: the residual gap is pair reuse, a second-order correlation
  refraction's per-neuron penalty cannot see; a write-side mechanism that
  penalises co-recruitment is the registered next question. PE-3 fails:
  the gap is not explained by pair sharing and stays open.
* PE-5 passes: an allocation that equalises usage stores beyond the
  random-pattern figure on this circuit; the real memory already equalises
  usage, so its headroom is the pair reuse.
* PE-6 passes: the gate's +24 to 34% is reduced pair reuse. PE-R alone
  passing moves the gated grids into runner provenance.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until PE-V to PE-6 are evaluated and recorded below.

    python -m research.runner pattern-efficiency \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag pattern-efficiency-20260930

### Amendment 9 result (2026-09-30)

One run from a worktree pinned at 6b58b1df, 20 brains, seeds 42 to 61,
all seven cells, 27 minutes:
[record](../../results/runs/memory.pattern-efficiency/pattern-efficiency-20260930/results.json),
[log](../../results/logs/pattern-efficiency-20260930.log). Write strength c
measured from the first item: 5 at k = 60, 6 at k = 120, 4 at k = 30, and 5
for the gated memory too. M* / (n/k)^2:

    cell        real   clean  random  c-2     c+2     balanced  gated  eta   eta_gated
    (2000, 60)  0.394  0.320  0.736   1.393   0.531   1.629            0.536
    (4000,120)  0.359  0.244  0.686   0.887   0.635   3.078            0.523
    (4000, 60)  0.432  0.371  0.806   1.463   0.613   1.384     0.583  0.536  0.724
    (8000,120)  0.512  0.357  0.920   1.075   0.974   2.296            0.557
    (8000, 60)  0.403  0.322  0.860   1.424   0.661   1.167     0.511  0.469  0.594
    out of regime, reported:
    (2000, 30)  0.354  0.335  1.135  >2.765   0.644   1.427            0.312
    (4000, 30)  0.283  0.261  1.167  >1.843   0.637   1.335            0.243

    PE-V  THE INSTRUMENT                                              PASS
          (a) dense and module recall agree EXACTLY at every real
          checkpoint (largest gap 0.000); (b) M*_real / Amendment 8:
          1.016, 1.040, 0.979, 1.021, 1.025 (0.991, 1.059 out of regime).
    PE-0  CALIBRATION AND TRUE NEGATIVE                               PASS
          random pair-sharing excess 0.998 to 1.001 at every checkpoint.
    PE-1  THE SUBSTRATE IS A WILLSHAW MEMORY                          PASS
          M*_random 0.736, 0.686, 0.806, 0.920, 0.860 (n/k)^2; the
          potentiated fraction there 0.516, 0.494, 0.548, 0.598, 0.571.
    PE-2  CONSTANT PATTERN EFFICIENCY                                 PASS
          eta 0.536, 0.523, 0.536, 0.557, 0.469; largest / smallest 1.19.
    PE-3  PAIR REUSE ACCOUNTS FOR THE GAP                             FAIL
          real pair-sharing excess at M*_real 2.17, 2.75, 2.14, 2.33,
          2.65 (all >= 1.5); real pair sharing at M*_real over random's
          at M*_random 1.15, 1.43, 1.14, 1.30, 1.23 -- (4000, 120) is
          outside [0.75, 1.33].
    PE-4  THE TRANSIENTS HELP                                         PASS
          M*_real / M*_clean 1.23, 1.47, 1.17, 1.43, 1.25.
    PE-5  BALANCED HEADROOM                                           FAIL
          M*_balanced / M*_random 2.21, 4.49, 1.72, 2.50, 1.36 --
          (8000, 60) is below 1.5.
    PE-S  WEIGHT SENSITIVITY                                          FAIL
          M*_random at c - 2: +89%, +29%, +81%, +17%, +66%; at c + 2:
          -28%, -7%, -24%, +6%, -23%. Every k = 60 cell moves more than
          30% two counts weaker.
    PE-R  THE GATE REPLAYED                                           PASS
          9078 / 8666 = 1.048 and 2593 / 2645 = 0.980.
    PE-6  THE GATE LOWERS PAIR REUSE                                  FAIL
          eta_gated > eta at both cells (0.724 > 0.536, 0.594 > 0.469);
          pair sharing at the common checkpoint lower under the gate
          (0.61 vs 1.00 at M = 2048; 0.73 vs 1.37 at M = 8192); gated
          pair sharing at M*_gated over random's at M*_random 1.22 and
          1.39 -- (8000, 60) is outside [0.75, 1.33].

**Reading.** The law's constant decomposes as registered at every judged
cell. Independent random k-subsets written into the same circuit at the
model's own write strength fail at 0.69 to 0.92 (n/k)^2 with half of the
present synapses potentiated (0.49 to 0.60): the circuit behaves like a
Willshaw memory at its information-optimal load, which is why the ceiling
is a function of n/k (the load is M (k/n)^2). The refracted memory realises
a constant 0.47 to 0.56 of that, the Hebbian control 0.014 to 0.023
(Amendment 8's control ceilings over this study's M*_random, reported):
refraction raises the pattern efficiency from about 2% to about 50%.

Three registered qualifications, all recorded as failures. (1) PE-S: the
random-pattern figure is a property of the circuit AT ITS WRITE STRENGTH,
not of the substrate alone. Potentiation is multiplicative, so a synapse
shared by s items weighs (1 + beta)^(c s); weaker writes amplify shared
pairs less, and at k = 60 two counts weaker raises the random ceiling
66 to 89%. The movement shrinks as each neuron sums more cue synapses
(k p / 2 = 7.5, 15, 30 at k = 30, 60, 120: > +144%, +66 to +89%, +17 to
+29%). "Pattern efficiency" is therefore defined against random patterns
written as hard as the memory writes its own. (2) PE-3: the refracted
assemblies share synapse pairs 2.1 to 2.8 times as much as random subsets,
at every cell, but they fail at 1.14 to 1.43 times the pair sharing that
random patterns fail at, not at the same load: pair reuse accounts for most
of the gap, not all of it, and the real memory tolerates more of it than a
clean write does (consistent with PE-4: its transient writes deepen its own
basins). (3) PE-5 and PE-6 each fail at (8000, 60) only: balanced headroom
is 1.36x there (1.7 to 4.5x elsewhere), and the gated memory fails at 1.39
times random's pair load.

The gate (PE-R) reproduces Amendments 5 and 6 at both cells, so those two
gated ceilings now have runner provenance. Its gain is carried by its
ASSEMBLIES, not a weaker write: the gated memory's first item is written as
hard as the ungated one's (c = 5), its assemblies share fewer pairs at equal
load, and rewritten cleanly at that same c they store 51% and 35% more than
the ungated assemblies rewritten the same way (2493 vs 1648, 7697 vs 5718;
reported, not a registered bar).

What this does not establish: the circuit's capacity at other write
strengths, where the random-pattern reference moves (Amendment 10 sweeps
it); and whether pair reuse is the CAUSE of the residual gap rather than a
correlate (no manipulation of pair reuse alone was made).

## Amendment 10 (2026-09-30, before running): the circuit's capacity across write strength

Registered before any run of this amendment. Amendment 9's PE-S failed: the
random-pattern ceiling that defines pattern efficiency moves +66 to +89% at
k = 60 when each item is written two counts weaker. Potentiation is
multiplicative and clipped, so a synapse shared by s items written c times
each weighs min((1 + beta)^(c s), w_max): with beta = 0.1 and w_max = 20 the
rule is nearly linear in s at c = 1, convex in the middle, and BINARY (1 or
20) from c = 32 on, where one item already reaches the clip. This amendment
traces the ceiling across that range so the reference Amendment 9 used is
placed on its curve.

**Seen before registering.** The bars below were drafted before a smoke run
(VOID: 3 brains, cap 1024, c in {1, 5, 32}, cells (2000, 60) and (4000, 120)).
Its readings, disclosed and not used to change any bar: c = 1 fails at the
first checkpoint (rank-1 below 0.5 at M = 16) at both cells; c = 5 reached
768 and 512, c = 32 679 and 512, both near the smoke cap.

### Protocol

`research/experiments/memory_write_strength.py` through the shared runner
(`python -m research.runner write-strength`), seeds 42 to 61, all seven
cells, one run from a worktree pinned at the commit registering this
amendment. Per cell: the circuit (presence, in-degree, chain table, half-cue
masked 8-round k-WTA recall, 32 sampled items per brain per checkpoint) as
in Amendment 9; each brain's own independent random k-subsets, generated per
(seed, cell) so the same patterns are read at every c (paired across c);
c in {1, 2, 3, 4, 5, 6, 8, 11, 16, 23, 32}; the model's own c measured as in
Amendment 9. Checkpoints per curve: from the power of two at or below
0.05 (n/k)^2, halve while the ensemble mean is at or below 0.5 (to 16 at
least), then double until it is, then add the 1.5x point in the bracket; the
cap is 131072 items (65536 at k = 120). Two ceilings per c, where the
ensemble mean crosses 0.5:

    rank-1      the recall is nearer the cued item than any other stored
                item (Amendment 9's criterion, IDENTIFICATION)
    completion  the fraction of sampled items whose recall recovers at
                least 0.8 of the item (COMPLETION); a curve already at or
                below 0.5 at its first checkpoint counts as zero

The potentiated fraction and pair sharing are recorded at every point.
Judged cells are the five in regime; (2000, 30) and (4000, 30) are reported,
and enter only WS-5.

### Bars

    WS-V  THE INSTRUMENT. At the model's own c the rank-1 ceiling is within
          10% of Amendment 9's random ceiling at every cell (818, 762, 3583,
          4088, 15288; 5043, 20741 out of regime). Fresh patterns and a
          different grid, so agreement is not exact. Failure voids WS-1..5.
    WS-1  THE BINARY END. At every judged cell the c = 32 rank-1 ceiling is
          resolved and at least 1.2 times the ceiling at the model's own c:
          a binary rule stores more than the operating write.
    WS-2  AN INTERIOR MINIMUM. At every judged cell the smallest rank-1
          ceiling over c in {4, 5, 6, 8, 11, 16}, times 1.2, is at most the
          ceiling at c = 2 and at most the ceiling at c = 32: the convex
          middle of the rule stores least.
    WS-3  COMPLETION WANTS A STRONGER WRITE THAN IDENTIFICATION. At every
          judged cell the c maximising the completion ceiling is larger
          than the c maximising the rank-1 ceiling.
    WS-4  THE n/k LAW BELONGS TO THE RULE. Calling the law "held" at c when
          both n/k-matched pairs ((2000, 60) with (4000, 120); (4000, 60)
          with (8000, 120)) have rank-1 ceilings within [0.8, 1.25] of each
          other: it holds at c = 32 and at every model c of the judged
          cells, and fails at c = 1 and c = 2.
    WS-5  SENSITIVITY FALLS WITH k p. The span (largest over smallest rank-1
          ceiling across c) is smaller at (4000, 120) than at (2000, 60),
          smaller at (8000, 120) than at (4000, 60), and larger at
          (2000, 30) than at (4000, 60).

### Interpretation, stated now

* WS-1 and WS-2 pass: the operating write sits in the rule's worst region;
  the "Willshaw" figure of Amendment 9 is the circuit at its own strength,
  and the circuit stores more under a clipped binary rule. Pattern
  efficiency is then quoted against both references.
* WS-1 fails: the binary end does not beat the operating write; the
  operating point is not a disadvantaged one.
* WS-3 passes: identification and completion prefer different write
  strengths, and a capacity quoted from rank-1 alone flatters weak writes.
* WS-4 passes: the n/k law is a property of saturating (clipped or
  strong-enough) plasticity, not of the substrate at any write.
* WS-5 passes: the dependence on write strength is a few-inputs effect
  that fades as each neuron sums more cue synapses, and a write-strength
  parameterisation that scales with k p is the registered next question.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until WS-V to WS-5 are evaluated and recorded below.

    python -m research.runner write-strength \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag write-strength-20260930

### Amendment 10 result (2026-10-01): VOID, the instrument failed

One run from a worktree pinned at 5c817ce0, 20 brains, seven cells, 8
minutes: [record](../../results/runs/memory.write-strength/write-strength-20260930/results.json),
[log](../../results/logs/write-strength-20260930.log).

    WS-V  THE INSTRUMENT                                              FAIL
          At the model's own c the rank-1 ceiling replays Amendment 9 at
          (2000, 60) 825 / 818, (4000, 120) 743 / 762, (4000, 60) 3623 /
          3583, (8000, 120) 3997 / 4088, (8000, 60) 15131 / 15288, but
          reads "below the first checkpoint" (16) at (2000, 30) and
          (4000, 30), where Amendment 9 measured 5043 and 20741.
    WS-1 .. WS-5                                                      VOID

**Why.** The adaptive search assumed one downward crossing: it halved
toward 16 whenever its first checkpoint failed. For weak writes the rank-1
curve is NOT monotone in load. It fails at low load and works at moderate
load (Amendment 9's own c - 2 curve at (2000, 60) reads 0.83 at M = 256 and
1.00 at 384 to 768 before falling), so every write that fails at the first
checkpoint was read as failing everywhere: c <= 4 at k = 60, c <= 3 at
k = 120, c <= 5 at k = 30, including the model's own c = 4 at both k = 30
cells. The bars that read those c (WS-2, WS-3, WS-4, WS-5 and WS-V) cannot be
judged, and by the registration a WS-V failure voids them all. Nothing is
adopted.

**What the record still shows (descriptive, from resolved curves only).**
Ceilings per (n/k)^2, rank-1 / completion, at the model's own c and the
binary end:

    cell        own c   rank-1   completion   c = 32 rank-1   completion
    (2000, 60)  5       0.743    0.015        0.610           0.563
    (4000,120)  6       0.669    0.320        0.915           0.798
    (4000, 60)  5       0.815    < 16         0.579           0.562
    (8000,120)  6       0.899    0.420        0.841           0.798
    (8000, 60)  5       0.851    < 16         0.566           0.561
    (2000, 30)  4       VOID     VOID         0.394           0.296
    (4000, 30)  4       VOID     VOID         0.362           0.281

Completion rises with write strength at every cell and the binary write
completes nearly every item it identifies (0.87 to 0.99 in regime, 0.75
and 0.78 out of it). At c = 32 the
completion ceiling is 0.28 to 0.30, 0.56 and 0.80 (n/k)^2 at k = 30, 60 and
120, the same within each k across n: at the binary write the n/k-matched
pairs DISAGREE (k = 120 stores 1.4x its k = 60 pair), so the n/k law of the
operating write does not hold there. The smallest write whose completion
ceiling exceeds 0.05 (n/k)^2 falls with k: c = 11 at k = 30, 6 at k = 60,
4 at k = 120, within every n/k level. These are readings of a void
study; Amendment 11 tests them on fresh patterns with an instrument that
measures both edges of a load window.

## Amendment 11 (2026-10-01, before running): the write-strength sweep with an instrument that sees load windows

Registered before any run of this amendment. It repeats Amendment 10's
question with the instrument corrected. **Every bar below tests a pattern
already SEEN** in Amendment 10's void record or in a smoke run, so a pass is
a REPLICATION on fresh patterns with a sound instrument, not a prediction.
Seen beforehand: the descriptive readings recorded under Amendment 10's
result; Amendment 9's c - 2 curve; an exploratory CPU probe (Bernoulli
presence, one brain) in which c = 3 recall holds 0.40 of the item after one
round at every load but drifts to chance (0.04) by round eight at M <= 128
and stops near 0.25 to 0.32 at M >= 256; and a smoke run of this instrument
(VOID: 3 brains, cap 1024, (2000, 60), c in {1, 5, 32}) whose rank-1 window
was [853, >= 1024) at c = 1, [16, 841) at c = 5 and [16, 685) at c = 32, and
completion window [16, 20) at c = 5 and [16, 626) at c = 32.

### Protocol

As Amendment 10 (`memory_write_strength.py`, `--scan`, protocol version 2),
with fresh patterns (generator salt `ws-scan`) and the instrument replaced:
every power of two from 16 is read for both metrics, doubling stops once a
metric has been above 0.5 and both have then been at or below it at two
consecutive points (or at the cap, 131072 items, 65536 at k = 120, or at the
first power of two at or above 8 (n/k)^2 if neither has yet risen), and the
1.5x point is added inside every doubling where either metric changes side.
Each metric's LOAD WINDOW is where its ensemble mean exceeds 0.5: a lower
edge (absent when it already does at M = 16) and an upper edge (the last
downward crossing; a curve still above 0.5 at its last point is censored and
counted at that point). "Capacity" below is the upper edge; a metric that
never exceeds 0.5 has capacity 0.

    python -m research.runner write-strength --scan \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag write-strength-scan-20261001

### Bars (judged cells: the five in regime unless stated)

    XV  THE INSTRUMENT. At the model's own c the rank-1 upper edge is
        resolved and within 10% of Amendment 9's random ceiling at all
        seven cells (818, 762, 3583, 4088, 15288; 5043, 20741). Failure
        voids X1..X6.
    X1  COMPLETION NEEDS A STRONG WRITE. The completion capacity at c = 32
        is positive and at least 1.5 times that at the model's own c.
    X2  THE BINARY WRITE COMPLETES WHAT IT IDENTIFIES. At c = 32 completion
        capacity over rank-1 capacity is at least 0.85.
    X3  A SEPARABLE LAW AT THE BINARY WRITE. Completion capacity at c = 32
        over (n/k)^2 is within 15% of its mean across the three k = 60
        cells and across the two k = 120 cells, and the k = 120 mean is at
        least 1.25 times the k = 60 mean.
    X4  THE n/k LAW BELONGS TO THE OPERATING WRITE. At each cell's own c the
        n/k-matched pairs ((2000, 60) with (4000, 120); (4000, 60) with
        (8000, 120)) have rank-1 capacities within [0.8, 1.25] of each
        other; at c = 32 both ratios (k = 60 over k = 120) are at most 0.8.
    X5  WEAK WRITES HAVE A LOAD WINDOW. At the three k = 60 cells, c = 3
        has a rank-1 lower edge (it fails at M = 16) and a rank-1 capacity
        above that of the model's own c.
    X6  THE SMALLEST COMPLETING WRITE FALLS WITH k p. Within each n/k level
        ((2000, 60), (4000, 120); (2000, 30), (4000, 60), (8000, 120);
        (4000, 30), (8000, 60)), the smallest c whose completion capacity
        exceeds 0.05 (n/k)^2 strictly decreases as k rises. All seven cells.

### Interpretation, stated now

* X1 and X2: capacity quoted from identification alone flatters weak
  writes; a memory that must COMPLETE wants the clipped, binary end.
* X3 and X4: the n/k law is a property of the operating write strength,
  not of the circuit; at the binary write capacity is g(k p) (n/k)^2.
* X5: identification by a weak write is a load window, not a ceiling.
* X6: the write a memory needs scales down with how many cue synapses each
  neuron sums -- the quantity a per-fiber, fan-in-scaled learning rate
  would hold fixed. A pass makes that parameterisation the next
  registration.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until XV to X6 are evaluated and recorded below.

### Amendment 11 result (2026-10-01)

One run from a worktree pinned at 8921c957, 20 brains, seven cells, 14
minutes: [record](../../results/runs/memory.write-strength/write-strength-scan-20261001/results.json),
[log](../../results/logs/write-strength-scan-20261001.log). Load windows
[lower, upper) in stored items; a lower edge of 0 means the metric already
exceeds 0.5 at M = 16, and an upper edge of 0 means it never does:

    cell        own c  rank-1 at c = 1 / 3 / own / 32          completion at own / 32
    (2000, 60)  5      [842, 4982) [213, 1519) [0, 828) [0, 684)       17 / 624
    (4000,120)  6      [885, 3117) [155, 1242) [0, 745) [0, 1009)      380 / 887
    (4000, 60)  5      [1666, 19348) [407, 6328) [0, 3567) [0, 2572)    0 / 2500
    (8000,120)  6      [1773, 10750) [339, 5135) [0, 4023) [0, 3723)    1887 / 3547
    (8000, 60)  5      [3310, 76028) [801, 25420) [0, 15455) [0, 10069) 0 / 9964
    out of regime:
    (2000, 30)  4      [1569, 27004) [439, 7577) [284, 5010) [0, 1761)  0 / 1315
    (4000, 30)  4      [2986, 106610) [840, 30135) [545, 20397) [0, 6503) 0 / 4965

    XV  THE INSTRUMENT                                                PASS
        own-c rank-1 upper edge over Amendment 9: 1.012, 0.978, 0.996,
        0.984, 1.011; 0.993 and 0.983 out of regime.
    X1  COMPLETION NEEDS A STRONG WRITE                               PASS
        c = 32 over own c: 624 vs 17, 887 vs 380, 2500 vs 0, 3547 vs
        1887, 9964 vs 0.
    X2  THE BINARY WRITE COMPLETES WHAT IT IDENTIFIES                 PASS
        0.91, 0.88, 0.97, 0.95, 0.99.
    X3  A SEPARABLE LAW AT THE BINARY WRITE                           PASS
        completion / (n/k)^2: 0.562, 0.563, 0.560 at k = 60; 0.798,
        0.798 at k = 120; ratio 1.42.
    X4  THE n/k LAW BELONGS TO THE OPERATING WRITE                    PASS
        n/k-matched rank-1 ratios at own c 1.11 and 0.89; at c = 32 0.68
        and 0.69.
    X5  WEAK WRITES HAVE A LOAD WINDOW                                PASS
        c = 3 lower edges 213, 407, 801; upper edges 1519, 6328, 25420
        against own c's 828, 3567, 15455.
    X6  THE SMALLEST COMPLETING WRITE FALLS WITH k p                  PASS
        c = 11 at k = 30, 6 at k = 60, 4 at k = 120, in every n/k level.

**Reading.** Amendment 10's descriptive readings replicate on fresh patterns
with an instrument that sees both window edges; every bar passes. A
recurrent area written with this multiplicative, clipped rule has two
capacities that want opposite write strengths. COMPLETION (recovering at
least 0.8 of an item from half of it) needs a strong write and is largest
at the clipped, binary end, where it follows a separable law, ~g(k p)
(n/k)^2 with g = 0.29, 0.56, 0.80 at k p = 15, 30, 60: there the n/k law
fails, and k = 120 stores 1.42 times its n/k pair. IDENTIFICATION (the
recall nearer the cued item than any other) grows as the write weakens, to
4.3 to 4.5 (n/k)^2 at c = 1 and k = 60, but inside a load window whose
lower edge also rises, and with almost no completion below c = 6 at
k = 60 (17 items at most). The model's operating write (c = 5 or 6) sits between: its
rank-1 ceiling obeys the n/k law (X4) and its completion is small. The
write that completes at all falls with the cue synapses each neuron sums
(X6). Amendment 9's "Willshaw" reference is the identification ceiling at
the operating write; it is neither the circuit's recall capacity nor its
identification capacity at other writes.

**Seen in the record, not registered.** The lower edge of the
identification window sits at a near-constant number of stored items PER
NEURON, M k / n, across n and k at each write: ~22 to 27 at c = 1, ~10 to
11 at c = 2, ~4.7 to 6.6 at c = 3, ~3 to 4.3 at c = 4 (e.g. at c = 2: 364, 673,
1331 items at n = 2000, 4000, 8000 with k = 60 are 10.9, 10.1, 10.0 per
neuron). A weak write's recall drifts off its item unless enough
potentiation has accumulated on every neuron; this is a candidate for its
own registration.

**What this does not establish.** Identification and completion here are
measured with independent random patterns on the memory's circuit, not
with the refracted memory's own assemblies at other write strengths; and
nothing here changes the write a model uses -- that is the per-fiber,
fan-in-scaled learning rate X6 motivates, the registered next question.

## Amendment 12 (2026-10-01, before running): does the best learning rate transfer across scale when scaled with fan-in?

Registered before any run of this amendment. Amendment 11's X6 found that
the weakest write that completes falls as each neuron sums more cue synapses
(11 / 6 / 4 counts at k p = 15 / 30 / 60). In muP terms the memory already
has a fan-in-scaled initialisation (norm_init divides by in-degree) and a
scale-free readout (k-WTA keeps only relative drive); what it lacks is a
learning rate that scales with fan-in. This asks the transfer question on
the REAL refracted memory: if beta is tuned for completion at one scale, is
the tuned value right at another? A signal-to-noise argument predicts it is
not, and that ln(1 + beta*) sqrt(k p / 2) is: the learned signal per item,
(1 + beta)^c - 1 summed over k p / 2 cue synapses, has to clear connectivity
noise that grows as sqrt(k p / 2).

**Seen before registering.** A first smoke run (VOID, 3 brains, cap 64) held
the refraction CHARGE fixed at 0.05; at beta = 0.05 that is 1.0 beta, past
the churn transition (Amendment 6, (0.6, 0.7] beta), and the first item
never settled (0 counts on its final set). The design was changed before
registering to keep refraction at the adopted 0.5 beta, so it scales with
the learning rate. A second smoke run (VOID, same size) then gave first-item
counts 3 / 5 / 6 at (2000, 60) and 5 / 6 / 7 at (8000, 240) for beta =
0.05 / 0.1 / 0.2, with recall still above 0.5 at the 64-item cap everywhere.

### Protocol

`research/experiments/memory_learning_rate.py` through the shared runner
(`python -m research.runner learning-rate`), seeds 42 to 61, one run from a
worktree pinned at the commit registering this amendment. The refracted
AssemblyMemory as in Amendment 8 (arm B, T = 8, w_max = 20, refraction 0.5
beta, ungated, the capacity study's stimuli), with beta, for both of its
fibers (recurrent and stimulus, whose fan-ins are equal here: k p), on the
grid 0.025, 0.0354, 0.05, 0.0707, 0.1, 0.1414, 0.2, 0.2828 (ratio sqrt 2).
Cells, two n/k levels with k p doubling within each:

    n/k = 33:  (2000, 60), (4000, 120), (8000, 240)    k p = 30, 60, 120
    n/k = 67:  (4000, 60), (8000, 120)                 k p = 30, 60

At each (cell, beta) items are stored in order and, at every checkpoint of
the geometric grid from 16 (16, 24, 32, 48, ...), the module's own masked
half-cue recall is read on 32 sampled items per brain: rank-1 and
completion (the fraction of items whose recall recovers at least 0.8 of
them). Storing stops once both metrics have been at or below 0.5 at two
consecutive checkpoints after either was above, or at a cap of 16 times the
cell's beta = 0.1 ceiling (0.4 (n/k)^2 for the new (8000, 240) cell, at most
40000). Each metric's load window is read with Amendment 11's `edges`;
capacity is the upper edge (a censored window counts at its last point
above; a metric that never exceeds 0.5 counts 0). beta* is the vertex of a
parabola in log beta through the best grid beta for COMPLETION and its two
neighbours (undefined when the best is a grid end); gamma* = ln(1 + beta*)
sqrt(k p / 2).

### Bars

    TV  THE INSTRUMENT. At beta = 0.1 the rank-1 capacity is within 10% of
        Amendments 7 and 8's ceilings at the four cells that have one
        (431, 383, 1961, 2230). Failure voids T1 to T4.
    T1  AN INTERIOR OPTIMUM. At every cell the completion capacity peaks
        strictly inside the beta grid.
    T2  THE STANDARD PARAMETERISATION DOES NOT TRANSFER. Within each n/k
        level beta* strictly decreases as k p rises.
    T3  THE FAN-IN PARAMETERISATION DOES. gamma* lies within 25% of its
        mean at all five cells.
    T4  WIDER IS BETTER AT THE OPTIMUM. Within each n/k level the best
        completion capacity over beta, per (n/k)^2, strictly increases with
        k p.

Reported, not judged: the rank-1 capacity across beta; first-item counts;
and how well ln(1 + beta*) (k p / 2)^a collapses across cells for
a = 0, 0.25, 0.5, 0.75, 1 (coefficient of variation) -- Amendment 11's X6
counts fall faster than sqrt between k p = 15 and 30, so the exponent is
not assumed.

### Interpretation, stated now

* T1 fails: completion does not have an optimum inside the grid; report the
  direction and do not quote beta*.
* T2 and T3 pass: the memory's learning rate is a per-fiber, fan-in-scaled
  quantity, as muP's is per layer; quote gamma*, not beta. T2 passes and T3
  fails: the optimum moves with scale, but not as sqrt; quote the fitted
  exponent descriptively and register it before using it.
* T2 fails: beta transfers as it stands at these scales.
* T4 passes: at its own optimum a wider area completes more per (n/k)^2,
  as Amendment 11 found for random patterns at the binary write.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until TV to T4 are evaluated and recorded below.

    python -m research.runner learning-rate \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag learning-rate-20261001

### Amendment 12 result (2026-10-01)

One run from a worktree pinned at 5fb9e2a7, 20 brains, five cells, eight
learning rates, 43 minutes:
[record](../../results/runs/memory.learning-rate/learning-rate-20261001/results.json),
[log](../../results/logs/learning-rate-20261001.log). Capacities (upper edges
of the load windows) for beta = 0.025, 0.0354, 0.05, 0.0707, 0.1, 0.1414,
0.2, 0.2828:

    (2000, 60)  completion  0     0     0     459   285   201   164   730
                rank-1      4079  2553  1668  954   438   258   180   131
    (4000,120)  completion  0     0     1241  684   342   260   219   1096
                rank-1      4096+ 4096+ 2442  911   398   266   208   114
    (8000,240)  completion  0     2672  1301  497   358   276   3097  1461
                rank-1      4975  3547  1369  511   354   256   201   113
    (4000, 60)  completion  0     0     0     1491  1030  608   492   416
                rank-1      13960 9978  5531  3636  1921  892   628   406
    (8000,120)  completion  0     0     4196  3875  1979  1227  1104  2176
                rank-1      18541 19084 10914 5570  2277  1246  1002  617

    TV  THE INSTRUMENT                                                PASS
        rank-1 at beta = 0.1 over Amendments 7/8: 1.016, 1.040, 0.979, 1.021.
    T1  AN INTERIOR OPTIMUM                                           FAIL
        (2000, 60): the largest completion capacity is at the grid's top
        (730 at 0.2828).
    T2  beta* FALLS WITH k p                                          FAIL
        n/k = 33: undefined, 0.053, 0.209; n/k = 67: 0.077, 0.058.
    T3  gamma* WITHIN 25%                                             FAIL
        undefined at (2000, 60); 1.47 at (8000, 240) against 0.29 elsewhere.
    T4  WIDER IS BETTER AT THE OPTIMUM                                PASS
        best completion / (n/k)^2: 0.657, 1.117, 2.787; 0.336, 0.944.

**Why T1 to T3 fail: completion counted MERGED recall.** At strong
learning rates the completion capacity jumps above the rank-1 capacity
(730 against 131 at (2000, 60), beta = 0.2828), which distinct items cannot
do. An exploratory check (4 brains, (2000, 60), the real memory) confirms
it: at beta = 0.0707 no stored assembly has a near-duplicate and every
completed recall is also rank-1; at beta = 0.2828, 69 to 73% of the stored
assemblies have a near-duplicate (overlap >= 0.8) and 62 of 79 completed
recalls are NOT rank-1 -- the recall recovers an assembly that several items
were merged into. The registered criterion did not require distinct items,
so these readings entered the optimum: they set (2000, 60)'s maximum at the
grid end and (8000, 240)'s at 0.2. T4's pass survives without them (best
completion that is not above the rank-1 capacity: 0.413, 1.117, 2.405;
0.336, 0.944).

**Post hoc, labelled, not adopted.** Dropping completion readings that
exceed the rank-1 capacity at the same beta (a rule chosen after seeing the
data), the optimum is beta* = 0.076, 0.053, 0.037 at n/k = 33 (k p = 30,
60, 120) and 0.078, 0.058 at n/k = 67 (k p = 30, 60), and gamma* =
ln(1 + beta*) sqrt(k p / 2) = 0.285, 0.285, 0.284, 0.289, 0.309. Its
coefficient of variation across the five cells is 0.032 with sqrt(k p),
against 0.242 for beta itself and 0.127, 0.134 and 0.265 for exponents 0.25,
0.75 and 1. The weakest learning rate that completes at all falls one grid
step (sqrt 2) per doubling of k p as well. This is the shape the
registration predicted, with the instrument it should have used;
Amendment 13 tests it on fresh brains, at cells it did not see, with
completion required to be distinct.

## Amendment 13 (2026-10-01, before running): the fan-in-scaled learning rate, confirmed or not, on fresh brains and unseen cells

Registered before any run of this amendment. Amendment 12's T1 to T3 failed
because its completion criterion counted recall of MERGED assemblies; post
hoc, with completion that is not above identification, gamma* =
ln(1 + beta*) sqrt(k p / 2) was 0.285 to 0.309 at all five cells
(coefficient of variation 0.032 at exponent 1/2). A rule chosen after the
data cannot be adopted. This amendment is the confirmatory test: the
completion criterion is fixed so it cannot count merged recall, the brains
are new, and two cells are added that Amendment 12 never saw, where the law
makes a PREDICTION rather than a fit.

**Seen before registering:** Amendment 12's record and its post hoc
analysis (above), the duplicate check, and one smoke run of this mode (VOID:
3 brains, cap 64, (3000, 90), beta 0.05 / 0.1 / 0.1414: first-item counts
4 / 6 / 6, every metric still above 0.5 at the cap).

### Protocol

As Amendment 12 (`memory_learning_rate.py`, now with `--distinct`, protocol
version 2), with four changes:

* DISTINCT COMPLETION: a recall counts as completing its item only if it
  recovers at least 0.8 of it AND is nearer it than any other stored item.
  Storing stops on rank-1 and distinct completion. Capacity is the distinct
  completion window's upper edge.
* FRESH BRAINS: seeds 62 to 81 (Amendments 7 to 12 used 42 to 61).
* A FINER GRID: beta = 0.025 x 2^(i/4), i = 0..10 (0.025 to 0.1414; the
  merged-recall region above it is dropped).
* SEVEN CELLS, five at n/k = 33 with k p = 30, 45, 60, 90, 120 --
  (2000, 60), (3000, 90), (4000, 120), (6000, 180), (8000, 240) -- and two
  at n/k = 67, (4000, 60) and (8000, 120). (3000, 90) and (6000, 180) are
  new; with gamma* = 0.29 the law predicts beta* = 0.063 and 0.044 there
  (and 0.078, 0.054, 0.038, 0.078, 0.054 at the others).

    python -m research.runner learning-rate --distinct \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag learning-rate-distinct-20261001 --seeds 62 63 64 65 66 67 68 69 70 \
        71 72 73 74 75 76 77 78 79 80 81

### Bars

    UV  THE INSTRUMENT. At beta = 0.1 the rank-1 capacity of the new brains
        is within 10% of Amendments 7 and 8's ceilings at the four cells
        that have one. Failure voids U1 to U4.
    U1  AN INTERIOR OPTIMUM of distinct completion at all seven cells.
    U2  THE LAW. gamma* lies within 15% of 0.29 at all seven cells,
        including the two the law has not seen.
    U3  THE EXPONENT. Of the exponents 0, 0.25, 0.5, 0.75 and 1 in
        ln(1 + beta*) (k p / 2)^a, a = 1/2 collapses beta* best across the
        seven cells (smallest coefficient of variation), and below 0.10.
    U4  WIDER IS BETTER AT THE OPTIMUM. Within each n/k level the best
        distinct-completion capacity per (n/k)^2 strictly increases with
        k p.

### Interpretation, stated now

* U1 to U3 pass: the refracted memory's learning rate is a fan-in-scaled
  quantity. The adoptable statement is that the completion-optimal beta
  satisfies ln(1 + beta*) = 0.29 / sqrt(k p / 2) across a fourfold range of
  fan-in, and that the adopted beta = 0.1 is above the optimum everywhere
  measured (the optimum is 0.078 at k p = 30). This is the assemblies
  counterpart of a muP learning-rate rule, for one area and one fiber
  type; transfer to multi-area circuits is not implied.
* U2 fails at the unseen cells only: the post hoc fit does not predict.
* U3 picks another exponent: report it; the law is not sqrt.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until UV to U4 are evaluated and recorded below.

### Amendment 13 result (2026-10-01)

One run from a worktree pinned at e17cdd95, 20 new brains (seeds 62 to 81),
seven cells, eleven learning rates, 69 minutes:
[record](../../results/runs/memory.learning-rate/learning-rate-distinct-20261001/results.json),
[log](../../results/logs/learning-rate-distinct-20261001.log). Distinct
completion capacity for beta = 0.025 x 2^(i/4), i = 0..10 (0.025 to 0.1414):

    (2000, 60)  0 0 0 0 0 390 457 385 290 225 197
    (3000, 90)  0 0 0 0 812 845 622 443 326 265 222        unseen
    (4000,120)  0 0 0 0 1245 1042 682 444 337 273 240
    (6000,180)  0 0 0 2149 1641 993 598 432 350 282 249    unseen
    (8000,240)  0 0 2729 2351 1305 846 495 417 336 285 255
    (4000, 60)  0 0 0 0 0 0 1505 1254 1007 771 589
    (8000,120)  0 0 0 0 4048 4157 3820 2905 1844 1312 1181

    cell        k p   beta* measured  predicted  gamma*   best / (n/k)^2
    (2000, 60)  30    0.0705          0.0778     0.264    0.411
    (3000, 90)  45    0.0558          0.0630     0.257    0.760
    (4000,120)  60    0.0532          0.0544     0.284    1.120
    (6000,180)  90    0.0444          0.0442     0.291    1.934
    (8000,240)  120   0.0377          0.0381     0.287    2.456
    (4000, 60)  30    0.0752          0.0778     0.281    0.339
    (8000,120)  60    0.0569          0.0544     0.303    0.935

    UV  THE INSTRUMENT                                                PASS
        rank-1 at beta = 0.1 on the new brains over Amendments 7/8:
        1.032, 1.058, 0.968, 0.985.
    U1  AN INTERIOR OPTIMUM                                           PASS
    U2  THE LAW, gamma* within 15% of 0.29                            PASS
        0.257 to 0.303; the unseen cells 0.257 and 0.291 (beta* 0.0558
        against 0.063 predicted, 0.0444 against 0.044).
    U3  THE EXPONENT                                                  PASS
        coefficient of variation 0.213 / 0.101 / 0.052 / 0.155 / 0.273 at
        exponents 0 / 0.25 / 0.5 / 0.75 / 1.
    U4  WIDER IS BETTER AT THE OPTIMUM                                PASS
        0.411 < 0.760 < 1.120 < 1.934 < 2.456; 0.339 < 0.935.

**Reading.** Amendment 12's post hoc finding replicates on new brains, with
completion that cannot count merged recall, and predicts the two cells it
had not seen. The refracted memory's completion-optimal learning rate obeys
ln(1 + beta*) = 0.29 / sqrt(k p / 2) across a fourfold range of fan-in: the
per-item log-weight change that best serves recall shrinks as the square
root of the number of cue synapses each neuron sums, the scaling a
signal-against-connectivity-noise argument gives. It is a learning-rate rule
in the sense of muP's per-layer rates, for one area and one fiber type.
The adopted beta = 0.1 is above the completion optimum at every cell (by
1.3 to 1.4x at k p = 30 and 2.7x at k p = 120), so the published ceilings are
identification ceilings at an over-strong write. Two features of the curve
matter for using the rule: completion switches on abruptly -- within one
grid step (2^(1/4)) it goes from nothing to near its maximum (0 at 0.042 to
812 at 0.05 at (3000, 90)) -- and from k p = 60 up the optimum sits at or
one step above that onset: the best write is about the weakest one that
completes.

**Seen in the record, not registered.** At its optimum the distinct
completion capacity does not follow (n/k)^2: it grows with k p at fixed
n/k (0.41 to 2.46 (n/k)^2 at n/k = 33). Written as M* = C n^2 p / (k
ln(n/k)), the sparse associative memory's synapse-count form, C is 0.0480
and 0.0475 at k p = 30 and 0.0655 and 0.0654 at k p = 60 across the two n/k
levels, rising to ~0.07 by k p = 90 to 120. If that holds, the refracted
memory at its own best learning rate stores in proportion to its synapses
(n^2 p) per bit of item (k ln(n/k)) -- a stronger and different law from
the (n/k)^2 of the operating write. It needs its own registration, with
cells that separate n^2 p / k from (n/k)^2 k p.

## Amendment 14 (2026-10-01, before running): what the refracted memory recalls against the Hebbian control, and in which variables it scales

Registered before any run of this amendment. P1's central sentence -- 0.345
to 0.502 (n/k)^2 assemblies, 23 to 38 times the Hebbian ceiling, a function
of n/k alone -- rests on rank-1 IDENTIFICATION at beta = 0.1. Amendments 11
to 13 showed identification can grow while completion vanishes, that the
n/k law belongs to the operating write, and that beta = 0.1 is 1.3 to 2.7
times above the completion optimum, ln(1 + beta*) = 0.29 / sqrt(k p / 2). At
that optimum Amendment 13 SAW, without registering it, distinct completion
following the sparse associative memory's synapse-count form, M* = C n^2 p /
(k ln(n/k)), with C matched across n/k at equal k p (0.0480 / 0.0475 at
k p = 30; 0.0655 / 0.0654 at 60). This amendment asks, on the criterion P1
must use, (i) whether the multiplier over the Hebbian control survives, (ii)
whether the synapse-count form holds where it and (n/k)^2 disagree, and
(iii) whether connectivity enters only through k p, including sparse p.

**Seen before registering:** Amendments 11 to 13 and their records, and a
smoke run of this study (VOID: 3 brains, cap 32, (8000, 240, 0.125) and
(1000, 60, 0.5), the refracted memory at its predicted beta and the control
at 0.1): the refracted memory completed above 0.5 to the cap at both; the
control's distinct completion ended near 20 and 5 items.

### Protocol

`research/experiments/memory_recall_law.py` through the shared runner
(`python -m research.runner recall-law`), seeds 82 to 101 (new brains), one
run from a worktree pinned at the commit registering this amendment. Ten
(n, k, p) cells in three blocks:

    A  k p = 30, p = 0.5:   (1000, 60), (2000, 60), (4000, 60), (8000, 60)
                            n/k = 16.7, 33, 67, 133
    B  k p = 60, p = 0.5:   (2000, 120), (4000, 120), (8000, 120)
                            n/k = 16.7, 33, 67
    C  sparse, n/k = 33:    (4000, 120, 0.25), (8000, 240, 0.125)  k p = 30
                            (8000, 240, 0.25)                      k p = 60

Every cell satisfies k p >= 3 ln n. Two memories per cell, each swept over
its own learning rates: the refracted AssemblyMemory (0.5 beta, T = 8,
w_max = 20, arm B, ungated) at beta_pred x 2^(j/4), j = -2..2, with beta_pred
= expm1(0.29 / sqrt(k p / 2)) (0.0778 at k p = 30, 0.0544 at 60); the
Hebbian control (strength 0, otherwise identical) at beta = 0.0125, 0.025,
0.05, 0.1, 0.2, 0.4. Recall is the module's own (masked for the refracted
memory) half-cue recall on 32 sampled items per brain at every checkpoint of
the geometric grid from M = 2. DISTINCT completion: the recall recovers at
least 0.8 of the item AND is nearer it than any other stored item. The
refracted memory stops on distinct completion (or gives up at 4 times the
synapse-count guess 0.06 n^2 p / (k ln(n/k)) if it never rises), the
control on rank-1 and distinct completion. A memory's CAPACITY at a cell is
the largest distinct-completion upper edge over its sweep (a curve that
never exceeds 0.5 is 0; the control's floor in the multiplier is 2, its
first checkpoint). C = capacity / (n^2 p / (k ln(n/k))).

    python -m research.runner recall-law \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag recall-law-20261001 --seeds 82 83 84 85 86 87 88 89 90 91 92 93 \
        94 95 96 97 98 99 100 101

### Bars

    WV  THE INSTRUMENT. The refracted capacity at (2000, 60), (4000, 60),
        (4000, 120), (8000, 120) at p = 0.5 is within 15% of Amendment 13's
        best distinct completion there (457, 1505, 1245, 4157) on new
        brains. Failure voids W1 to W4.
    W1  THE MULTIPLIER ON RECALL. At every cell the refracted capacity is
        at least 10 times the control's (each at its own best learning
        rate).
    W2  THE SYNAPSE-COUNT FORM. In block A and in block B separately, C is
        within 15% of the block's mean at every cell, and C varies less
        across the block (coefficient of variation) than capacity /
        (n/k)^2 does.
    W3  CONNECTIVITY ENTERS ONLY AS k p. Within each group of equal (n/k,
        k p) and different p -- {(2000, 60, 0.5), (4000, 120, 0.25),
        (8000, 240, 0.125)} and {(4000, 120, 0.5), (8000, 240, 0.25)} --
        capacities are within 15% of the group mean and the best
        learning rate is the same or an adjacent grid point.
    W4  THE LEARNING-RATE LAW AT SPARSE p. At the three p < 0.5 cells the
        refracted memory's best learning rate is interior to its sweep
        (the optimum is within one grid step, 2^(1/4), of the law's prediction).

### Interpretation, stated now

* W1 passes: P1's multiplier is restated on recall, each memory at its own
  optimum; the identification multiplier is reported beside it. W1 fails:
  P1 may not claim a recall multiplier of that size.
* W2 passes: at its optimal learning rate the refracted memory is a
  synapse-limited associative memory, M* = C(k p) n^2 p / (k ln(n/k)), and
  P1 quotes that law, with the (n/k)^2 law of the operating write as the
  fixed-beta special case. W2 fails: the law stays (n/k)^2 g(k p) and the
  synapse form is not quoted.
* W3 and W4 pass: the memory and its learning-rate rule depend on
  connectivity only through k p over p = 0.125 to 0.5, so a sparser,
  cortex-like area behaves like a dense one with the same fan-in.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until WV to W4 are evaluated and recorded below.

### Amendment 14 result (2026-10-01)

One run from a worktree pinned at be8d44ef, 20 new brains (seeds 82 to
101), ten cells, 27 minutes:
[record](../../results/runs/memory.recall-law/recall-law-20261001/results.json),
[log](../../results/logs/recall-law-20261001.log). Capacity is the best
distinct-completion upper edge over each memory's own learning-rate sweep;
C = capacity / (n^2 p / (k ln(n/k))):

    cell (n, k, p)    block  refracted  best beta (index)  control  multiplier  C       /(n/k)^2
    (1000, 60, .5)    A      133        0.0654 (1)         5        25.6        0.0450  0.479
    (2000, 60, .5)    A      492        0.0654 (1)         16       31.2        0.0517  0.443
    (4000, 60, .5)    A      1616       0.0654 (1)         63       25.5        0.0509  0.364
    (8000, 60, .5)    A      4360       0.0654 (1)         312      14.0        0.0400  0.245
    (2000, 120, .5)   B      303        0.0457 (1)         9        35.0        0.0511  1.090
    (4000, 120, .5)   B      1208       0.0457 (1)         38       31.7        0.0635  1.087
    (8000, 120, .5)   B      4138       0.0544 (2)         158      26.2        0.0652  0.931
    (4000, 120, .25)  C      442        0.0925 (3)         19       23.4        0.0465  0.398
    (8000, 240, .125) C      502        0.1100 (4)         19       26.1        0.0528  0.452
    (8000, 240, .25)  C      1109       0.0769 (4)         19       57.3        0.0583  0.998

    WV  THE INSTRUMENT                                                PASS
        refracted capacity over Amendment 13: 1.077, 1.074, 0.970, 0.995.
    W1  THE MULTIPLIER ON RECALL                                      PASS
        14.0 to 57.3 at every cell (the control at its own best beta,
        0.05 or 0.1).
    W2  THE SYNAPSE-COUNT FORM                                        FAIL
        block A: C within 15% of its mean (0.040 to 0.052) and tighter
        than capacity / (n/k)^2 (CV 0.10 against 0.23); block B: C within
        15% (0.051 to 0.065) but LOOSER than capacity / (n/k)^2 (CV 0.10
        against 0.07).
    W3  CONNECTIVITY ONLY AS k p                                      FAIL
        capacities agree: 492 / 442 / 502 (within 8% of their mean) and
        1208 / 1109 (within 5%); the best learning rates do not: grid
        indices 1 / 3 / 4 and 1 / 4.
    W4  THE LEARNING-RATE LAW AT SPARSE p                             FAIL
        best beta at index 3, 4, 4 of 0..4: two of the three sparse cells
        peak at the top of the sweep.

**Reading.** P1's multiplier survives on the criterion it must use: with
each memory at its own best learning rate, the refracted memory completes
14 to 57 times as many distinct items as the Hebbian control at every one
of ten cells (the smallest, 14x, at (8000, 60), where the control reaches
312). The scaling law is not settled. At k p = 30 the synapse-count form
describes the four cells better than (n/k)^2, but C is not flat -- it peaks
at n/k = 33 to 67 (0.052, 0.051) and is lower at both ends (0.045, 0.040);
at k p = 60, (n/k)^2 is the tighter description over n/k = 17 to 67. The
two forms differ by a factor of ln(n/k), which changes only 1.7-fold over
this range, so ten cells do not separate them; nothing is adopted. The
sparse cells answer half of W3: at equal (n/k, k p) the CAPACITY is the same
at p = 0.125, 0.25 and 0.5 (within 8%), so a sparser area with the same
fan-in stores as much; but its best learning rate is higher.

**Post hoc, labelled, not adopted.** The noise a write must beat is the
spread of the number of cue synapses a neuron receives, k p (1 - p) / 2,
not k p / 2. Scaling the learning-rate law by it predicts a best beta
sqrt((1 - p) / 0.5) times the dense value: 1.22x at p = 0.25 and 1.32x at
p = 0.125, i.e. 0.095 and 0.103 at k p = 30 (observed best 0.0925 and the
grid's top, 0.11) and 0.066 at k p = 60, p = 0.25 (observed the top,
0.077). This fits the direction and roughly the size (though this run's dense
cells also peaked one grid step below the law's 0.078, at 0.0654, which the
sparse optima overshoot by a further one to two steps); it needs its own
registration with a sweep that extends above these optima. Also seen:
judged on identification instead, with each memory at its best rate, the
multiplier is smaller at the largest cell (15429 against 2499 rank-1 at
(8000, 60), 6x), because a weakly written Hebbian control identifies far
more than it completes.

## Amendment 15 (2026-10-01, before running): write rounds as depth -- does the per-round learning rate scale as 1/T?

Registered before any run of this amendment. Each item is written by T
rounds of stimulus plus recurrence, every round multiplying the weights of
pairs that fire together by (1 + beta); the log-weight an item puts on its
own assembly therefore grows with T at a fixed beta. Amendment 13 fixed the
completion-optimal per-item write at T = 8: ln(1 + beta*) = 0.29 /
sqrt(k p / 2). The depth reading of muP is that rounds are depth, and the
per-round rate should shrink with them so the per-item write stays put:
beta_T = expm1(ln(1 + beta*_8) x 8 / T). The earlier record reads the other
way: fewer rounds per item raised the ceiling (R5: T = 8 above T = 16 at
beta = 0.1), and the convergence gate, which stops writing an item once it
has settled, raised it 24 to 34%. If those were OVER-WRITING, one law
explains all three.

**Seen before registering:** the earlier T results above; a smoke run of
this study (VOID: 3 brains, cap 32, (2000, 60), T = 8, beta = 0.1, distinct
completion above 0.5 to the cap); and a direct check of the T path (VOID: 3
brains, cap 32, T = 16 at beta = 0.0381 with 8 recall rounds): first-item
count 10, so the per-item write 10 ln(1.0381) = 0.37 matches T = 8's at its
law rate (about 5 ln(1.0778)); distinct completion 0.83 to 1.00 up to 32
items.

### Protocol

`research/experiments/memory_time_depth.py` through the shared runner
(`python -m research.runner time-depth`), seeds 102 to 121 (new brains), one
run from a worktree pinned at the commit registering this amendment. The
refracted AssemblyMemory (0.5 beta, w_max 20, arm B, ungated, the capacity
study's stimuli) at (2000, 60) and (4000, 60) (k p = 30), write rounds T in
{6, 8, 12, 16}; RECALL held at 8 rounds at every T (`AssemblyMemory.recall`
gains a `rounds` argument, default unchanged), so only the write's depth
varies. At each T, beta over beta_T x 2^(j/4), j = -2..3 (beta_T = 0.105,
0.0778, 0.0512, 0.0381 for T = 6, 8, 12, 16), kept at or above 0.025 --
where the int8 count matrix saturates exactly at the clip -- plus beta =
0.1, which serves D1 only. Distinct completion and capacity as in
Amendments 13 and 14; checkpoints from M = 2. beta*_T is the parabola
vertex in log beta over the regular grid.

    python -m research.runner time-depth \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag time-depth-20261001 --seeds 102 103 104 105 106 107 108 109 110 \
        111 112 113 114 115 116 117 118 119 120 121

### Bars

    DV  THE INSTRUMENT. At T = 8 the best capacity is within 15% of
        Amendment 14's (492 at (2000, 60), 1616 at (4000, 60)) on new
        brains. Failure voids D1 to D4.
    D1  THE OLD EFFECT. At beta = 0.1, the capacity at T = 16 is below
        the capacity at T = 8 at both cells.
    D2  THE RESCALED WRITE TRANSFERS. At the law point beta_T, the capacity
        at T = 6, 12 and 16 is within 15% of T = 8's, at both cells.
    D3  THE BEST RATE SCALES AS 1/T. T ln(1 + beta*_T) is within 20% of its
        mean across the four depths, and its coefficient of variation is
        below that of ln(1 + beta*_T) itself, at both cells.
    D4  DEPTH DOES NOT MATTER AT THE OPTIMUM. The best capacity over beta at
        each T is within 15% of its mean across the four depths, at both
        cells.

### Interpretation, stated now

* D2 to D4 pass: write rounds are depth in the muP sense; the per-round
  rate should be quoted as T ln(1 + beta), and "fewer rounds store more"
  (R5) and the convergence gate's gain are, at least in part, protection
  against over-writing. The learning-rate law is restated per item.
* D3 passes and D4 fails: the per-round rate scales as 1/T but some depths
  store more even at their optimum; report which, and that rounds do more
  than accumulate the write (settling, refraction's per-round charge).
* D3 fails: the per-round optimum does not scale as 1/T; report its
  scaling.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until DV to D4 are evaluated and recorded below.

### Amendment 15 result (2026-10-01)

One run from a worktree pinned at d5219e3a, 20 new brains (seeds 102 to
121), two cells, four depths, 18 minutes:
[record](../../results/runs/memory.time-depth/time-depth-20261001/results.json),
[log](../../results/logs/time-depth-20261001.log). Distinct-completion
capacity on each depth's grid (the extra beta = 0.1 point beside it):

    (2000, 60)  T = 6   0.0742: 0    0.0883: 423  0.105: 453  0.125: 395  0.149: 315  0.177: 275   (0.1: 459)
                T = 8   0.055: 0     0.0654: 486  0.0778: 424 0.0925: 320 0.11: 240   0.131: 206   (0.1: 287)
                T = 12  0.0362: 421  0.043: 440   0.0512: 316 0.0609: 225 0.0724: 171 0.0861: 143  (0.1: 120)
                T = 16  0.027: 515   0.0321: 402  0.0381: 271 0.0454: 172 0.054: 134  0.0642: 110  (0.1: 86)
    (4000, 60)  T = 6   0.0742: 0    0.0883: 0    0.105: 1408 0.125: 1269 0.149: 1112 0.177: 900   (0.1: 1493)
                T = 8   0.055: 0     0.0654: 1575 0.0778: 3*  0.0925: 1162 0.11: 871  0.131: 641   (0.1: 977)
                T = 12  0.0362: 1540 0.043: 1554  0.0512: 1218 0.0609: 848 0.0724: 561 0.0861: 419 (0.1: 348)
                T = 16  0.027: 1661  0.0321: 1359 0.0381: 1095 0.0454: 657 0.054: 426 0.0642: 314  (0.1: 207)

    DV  THE INSTRUMENT                                                PASS
        best at T = 8: 486 and 1575 against Amendment 14's 492 and 1616.
    D1  THE OLD EFFECT                                                PASS
        at beta = 0.1, T = 16 stores 86 and 207 against T = 8's 287 and 977.
    D2  THE RESCALED WRITE TRANSFERS                                  FAIL
        at the law point beta_T, T = 12 and 16 store 316 / 271 and
        1218 / 1095 against T = 8's 424 and (*) 3.
    D3  THE BEST RATE SCALES AS 1/T                                   FAIL
        the T = 16 optimum is the grid's lowest point at both cells.
    D4  DEPTH DOES NOT MATTER AT THE OPTIMUM                          PASS
        best capacity 453 / 486 / 440 / 515 and 1408 / 1575 / 1554 /
        1661 across T = 6, 8, 12, 16, within 9% of each cell's mean.

(*) An instrument artifact, recorded and not corrected: the store at (4000,
60), T = 8, beta = 0.0778 read distinct completion 0.55, 0.48, 0.49 at M =
2, 3, 4 -- a recall sample of two to four items -- and the rule "two
readings at or below 0.5 after one above" stopped it there. Its neighbours
store 1575 and 1162. The same rule from M = 2 stopped eighteen Hebbian-
control stores in Amendment 14 at M <= 16 (no refracted ones); the control's
capacities are genuinely small there, but at worst they are understated,
which can only have raised W1's multipliers, whose smallest (14x) is well
clear of its 10x bar. Studies registered from here on hold the stop
decision until M >= 32.

**Reading.** Write rounds behave like depth in the sense that matters most:
at its own best learning rate every depth from T = 6 to 16 stores the same
(D4), so the old observation that 8 rounds beat 16 (D1, replicated) was
over-writing at a fixed per-round rate, not a property of depth. The
convergence gate's gain and "stronger beta hurts" read the same way. What
fails is the specific rescaling: the per-round optimum falls FASTER than
1/T -- T ln(1 + beta*) is 0.60, 0.51, 0.50 and at most 0.43 for T = 6, 8,
12, 16 -- so later rounds count for more than earlier ones, as they would if
an item spends its first rounds settling and only its last rounds writing
the assembly it ends in. The first-item counts agree: at each depth's
optimum the measured per-item write c ln(1 + beta) is 0.30 / 0.25 / 0.30 /
0.21 and 0.30 / 0.19 / 0.21 / 0.13, closer to constant than T ln(1 + beta)
but falling at T = 16, whose optimum the grid did not reach. Registering
the invariant as "the write the final assembly receives", with a grid that
contains the T = 16 optimum, is the next test of this line.

## Amendment 16 (2026-10-01, before running): does the learning-rate law need the connectivity-noise factor (1 - p)?

Registered before any run of this amendment. Amendment 13 fixed the
completion-optimal learning rate at p = 0.5: ln(1 + beta*) = 0.29 /
sqrt(k p / 2). Amendment 14 found that at equal (n/k, k p) a sparser area
stores as much but wants a higher rate, and proposed post hoc that the
write must clear the SPREAD of the cue synapses a neuron receives,
Binomial(k/2, p) with standard deviation sqrt(k p (1 - p) / 2), not
sqrt(k p / 2):

    ln(1 + beta*) = 0.29 sqrt(2 (1 - p)) / sqrt(k p / 2).

That is identical at p = 0.5, 1.22x and 1.32x higher at p = 0.25 and 0.125,
and 0.71x LOWER at p = 0.75. The plain fan-in law predicts no change with p
at fixed k p, so the dense side (p = 0.75) is where the two disagree most.

**Seen before registering:** Amendment 14's sparse cells (optima at grid
index 3 and 4 of 0..4) and Amendment 15's result, which bears on depth, not
on p. No run of this study has been made.

### Protocol

`research/experiments/memory_sparse_law.py` through the shared runner
(`python -m research.runner sparse-law`), seeds 122 to 141 (new brains), one
run from a worktree pinned at the commit registering this amendment. The
refracted AssemblyMemory (0.5 beta, T = 8, w_max 20, arm B, ungated, the
capacity study's stimuli) at six cells with n/k = 33:

    k p = 30:  (1333, 40, 0.75), (2000, 60, 0.5), (4000, 120, 0.25),
               (8000, 240, 0.125)
    k p = 60:  (4000, 120, 0.5), (8000, 240, 0.25)

Each swept over beta_pred x 2^(j/4), j = -4..4 (an octave either side of the
(1 - p) law's prediction: 0.0544, 0.0778, 0.096, 0.104, 0.0544, 0.067), kept
at or above 0.025. Distinct completion and capacity as in Amendments 13 and
14, checkpoints from M = 2, with no stop decision before M = 32 (Amendment
15's early-stop artifact); beta* is the parabola vertex in log beta. Two normalised optima:
gamma_plain = ln(1 + beta*) sqrt(k p / 2) and gamma_noise = gamma_plain /
sqrt(2 (1 - p)).

    python -m research.runner sparse-law \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag sparse-law-20261001 --seeds 122 123 124 125 126 127 128 129 130 \
        131 132 133 134 135 136 137 138 139 140 141

### Bars

    SV  THE INSTRUMENT. The best capacity at the five cells Amendment 14
        measured is within 15% of its values there (492, 442, 502, 1208,
        1109) on new brains. Failure voids S1 to S4.
    S1  AN INTERIOR OPTIMUM at all six cells.
    S2  THE NOISE LAW. gamma_noise lies within 15% of its mean at all six
        cells and varies less across them than gamma_plain does.
    S3  THE DENSE SIDE. beta* at (1333, 40, 0.75) is at most 0.85 times
        beta* at (2000, 60, 0.5) (the noise law predicts 0.71, the plain
        law 1.0).
    S4  CAPACITY IS p-INVARIANT AT EQUAL FAN-IN. At k p = 30 the four best
        capacities (p = 0.75 to 0.125) lie within 15% of their mean.

### Interpretation, stated now

* S2 and S3 pass: the learning-rate law is the write clearing the
  connectivity noise, ln(1 + beta*) proportional to sqrt((1 - p) / (k p)),
  and it is stated that way for every p; a cortex-like sparse area needs a
  rate set by its fan-in AND its sparseness.
* S2 passes and S3 fails: the correction holds on the sparse side only.
* S4 passes: capacity at the optimum depends on connectivity only through
  k p over p = 0.125 to 0.75 (Amendment 14 found it for p <= 0.5).
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until SV to S4 are evaluated and recorded below.

### Amendment 16 result (2026-10-01)

One run from a worktree pinned at ecd17c7a, 20 new brains (seeds 122 to
141), six cells, 22 minutes:
[record](../../results/runs/memory.sparse-law/sparse-law-20261001/results.json),
[log](../../results/logs/sparse-law-20261001.log). Distinct-completion
capacity on each cell's grid, and the optimum:

    cell (n, k, p)      k p  beta*   gamma_plain  gamma_noise  best
    (1333, 40, 0.75)    30   0.0493  0.186        0.264        492
    (2000, 60, 0.5)     30   0.0698  0.261        0.261        497
    (4000, 120, 0.25)   30   0.0951  0.352        0.287        436
    (8000, 240, 0.125)  30   0.1144  0.420        0.317        489
    (4000, 120, 0.5)    60   0.0497  0.266        0.266        1205
    (8000, 240, 0.25)   60   0.0723  0.382        0.312        1129

    SV  THE INSTRUMENT                                                PASS
        best capacity over Amendment 14: 1.010, 0.986, 0.974, 0.998, 1.018.
    S1  AN INTERIOR OPTIMUM                                           PASS
    S2  THE NOISE LAW                                                 PASS
        gamma_noise 0.261 to 0.317, within 15% of its mean 0.285;
        coefficient of variation 0.081 against gamma_plain's 0.258.
    S3  THE DENSE SIDE                                                PASS
        beta* at p = 0.75 over p = 0.5: 0.707 (predicted 0.71; the
        plain fan-in law predicts 1.0).
    S4  CAPACITY IS p-INVARIANT AT EQUAL FAN-IN                       PASS
        492, 497, 436, 489 at p = 0.75, 0.5, 0.25, 0.125 (within 9%).

**Reading.** The learning-rate law needs the connectivity noise: the write
that best serves distinct completion clears the spread of the number of cue
synapses a neuron receives, sqrt(k p (1 - p) / 2), and

    ln(1 + beta*) = 0.285 sqrt(2 (1 - p)) / sqrt(k p / 2)

describes all six cells (p = 0.125 to 0.75, k p = 30 and 60) within 15%,
including the dense side, where the sparse-side correction predicted the
opposite of the plain law and was right to within 0.5%. Capacity at the
optimum does not depend on p at fixed fan-in. A sparse, cortex-like area
with the same fan-in stores as much as a dense one, but must be written with
a learning rate set by its fan-in AND its sparseness. The residual is a slow
rise of gamma_noise toward sparse p (0.26 to 0.32), not captured by the
correction.

**Seen in the record, not registered.** Against the form
sqrt((1 - p) ln n / (p k)), which the projection-convergence thresholds of
ITCS 2019 and COLT 2022 scale as (see
[LITERATURE_SYNTHESIS.md](../LITERATURE_SYNTHESIS.md)), the six optima are
0.201, 0.196, 0.209, 0.189, 0.223, 0.216 (coefficient of variation 0.057),
the same 0.20 as Amendment 13's seven cells (0.031). The extra ln n absorbs
part of the sparse-side residual. Whether the best learning rate is a fixed
fraction of the convergence threshold is the registration this points to.

## Amendment 17 (2026-10-01, before running): is the best learning rate a fixed fraction of the convergence threshold, above and below the regime floor?

Registered before any run of this amendment. Read against the literature
(research/notes/LITERATURE_SYNTHESIS.md), the completion-optimal learning
rates of Amendments 13 and 16 are a near-constant fraction of

    theta(n, k, p) = sqrt((1 - p) ln n / (p k)),

the form the assembly calculus's projection-convergence thresholds (ITCS
2019, COLT 2022) scale as: 0.20 with a coefficient of variation of 0.031
over Amendment 13's seven cells and 0.057 over Amendment 16's six. That was
seen after the data. This registers it, and carries it below the regime
floor k p >= 3 ln n, into the fan-in range every published simulation runs
at (k p = 1 to 10).

**Seen before registering:** Amendments 13 to 16 and the post hoc fractions
above; and a smoke run (VOID: 3 brains, cap 32): at the PNAS 2020 cell
(n = 10000, k = 100, p = 0.01, k p = 1) at beta = 0.6039, distinct completion
read 0.00 at every checkpoint from M = 2 to 16 while rank-1 read 0.65 to
0.83; at (4000, 10, 0.5) (k p = 5) at beta = 0.1821, distinct completion read
0.25 to 0.46, never above 0.5.

### Protocol

`research/experiments/memory_threshold_law.py` through the shared runner
(`python -m research.runner threshold-law`), seeds 142 to 161 (new brains),
one run from a worktree pinned at the commit registering this amendment. The
refracted AssemblyMemory (0.5 beta, T = 8, w_max 20, arm B, ungated, the
capacity study's stimuli) at ten cells:

    p = 0.5, n = 4000:     k = 10, 20, 40 (below the floor, k p = 5-20)
                           k = 60, 80, 160 (above it, k p = 30-80)
    p = 0.05, n = 8000:    k = 100, 200, 400 (below, k p = 5-20)
    PNAS 2020:             n = 10000, k = 100, p = 0.01 (below, k p = 1)

Each swept over 0.2 theta x 2^(j/4), j = -4..4 (an octave either side of the
fraction's prediction), at or above 0.025. Distinct completion and capacity
as in Amendments 13 to 16; checkpoints from M = 2; no stop decision before
M = 32; a cell whose completion never rises stops at min(4 x the synapse-count
guess + 64, 8192) items. beta* is the parabola vertex in log beta; its
FRACTION is beta* / theta.

    python -m research.runner threshold-law \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag threshold-law-20261001 --seeds 142 143 144 145 146 147 148 149 150 \
        151 152 153 154 155 156 157 158 159 160 161

### Bars

    TV  THE INSTRUMENT. The best capacity at (4000, 60, 0.5) is within 15% of
        Amendment 14's 1616 on new brains. Failure voids T1 and T2.
    T1  THE FRACTION ABOVE THE FLOOR. At the three cells with k p >= 3 ln n,
        an interior optimum exists and its fraction lies within 15% of 0.20.
    T2  THE FRACTION BELOW THE FLOOR. At least one of the seven cells below
        the floor has an interior optimum, and at every one that does the
        fraction lies within 25% of 0.20.

Reported, not judged: every cell's best capacity per synapse-count scale,
its rank-1 capacity, and which cells below the floor complete at all.

### Interpretation, stated now

* T1 passes: the completion-optimal learning rate of a refracted area is
  0.20 of the convergence-threshold form, a rule from the field's own theory
  that sets plasticity from (n, k, p) alone; it replaces the fitted constants
  of Amendments 13 and 16 above the floor.
* T2 passes: the rule extends to the fan-in the published simulations use.
* T2 fails because no cell below the floor completes: the regimes the
  literature runs at (k p = 1 to 10) do not support completion-grade memory
  at any learning rate, only identification -- the published operations are
  existence proofs at low load, as the synthesis argued.
* T2 fails with interior optima at the wrong fraction: the rule is an
  above-floor rule.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until TV, T1 and T2 are evaluated and recorded below.

### Amendment 17 result (2026-10-01)

One run from a worktree pinned at a8b95efe, 20 new brains (seeds 142 to
161), ten cells, 37 minutes:
[record](../../results/runs/memory.threshold-law/threshold-law-20261001/results.json),
[log](../../results/logs/threshold-law-20261001.log). theta =
sqrt((1 - p) ln n / (p k)); the floor is k p >= 3 ln n (24.9 at n = 4000,
27.0 at n = 8000); rank-1 capacity at 8192 is the give-up cap (a lower
bound):

    cell (n, k, p)      k p  floor  theta   beta*   fraction  distinct  rank-1
    (4000, 60, 0.5)     30   above  0.3718  0.0669  0.180     1589      >=8192
    (4000, 80, 0.5)     40   above  0.3220  0.0608  0.189     1312      7719
    (4000, 160, 0.5)    80   above  0.2277  0.0411  0.180     1017      2510
    (4000, 10, 0.5)     5    below  0.9107  0.1968  0.216     1356      >=8192
    (4000, 20, 0.5)     10   below  0.6440  0.1241  0.193     1382      >=8192
    (4000, 40, 0.5)     20   below  0.4554  0.0968  0.213     1452      >=8192
    (8000, 100, 0.05)   5    below  1.3067  0.3872  0.296     201       1492
    (8000, 200, 0.05)   10   below  0.9240  0.2496  0.270     182       926
    (8000, 400, 0.05)   20   below  0.6534  0.1377  0.211     161       484
    (10000, 100, 0.01)  1    below  3.0196  --      --        0         112

    TV  THE INSTRUMENT                                                PASS
        best capacity at (4000, 60, 0.5): 1589 against Amendment 14's
        1616 (0.983).
    T1  THE FRACTION ABOVE THE FLOOR                                  PASS
        0.180, 0.189, 0.180 at k p = 30, 40, 80 (bar: 0.17 to 0.23).
    T2  THE FRACTION BELOW THE FLOOR                                  FAIL
        six of seven cells resolved; within 25% of 0.20 at the three
        p = 0.5 cells (0.216, 0.193, 0.213) and at (8000, 400, 0.05)
        (0.211); outside it at (8000, 100, 0.05) (0.296, +48%) and
        (8000, 200, 0.05) (0.270, +35%). The PNAS 2020 cell completes at
        no learning rate.

**Reading.** Above the floor the completion-optimal learning rate is a
fixed fraction of the convergence-threshold form, now on new brains and at
k p = 80, beyond the fan-in Amendments 13 and 16 measured: 0.18 to 0.19 of
sqrt((1 - p) ln n / (p k)), set from (n, k, p) alone. It replaces the fitted
constants of Amendments 13 and 16 there. The three fractions sit a little
below the post hoc 0.20 (mean 0.183), inside the bar.

Below the floor the rule does not hold as registered, and how it fails is
not one of the readings stated in advance. It is not that nothing below the
floor completes: six of seven cells do. Nor is the rule simply an above-floor
rule: at p = 0.5 it holds unchanged down to k p = 5 (0.19 to 0.22). It fails
at the SPARSE cells, where the fraction climbs as fan-in falls (0.21, 0.27,
0.30 at k p = 20, 10, 5). Below the floor the optimum depends on p at fixed
k p, which theta's (1 - p) factor does not capture. The registered reading
"the rule is an above-floor rule" is the nearest stated one, and it holds
for sparse areas only.

Two things below the floor that the bars did not ask about:

* **Capacity loses its laws.** At p = 0.5 the distinct-completion capacity
  is 1356, 1382, 1452 at k = 10, 20, 40: flat in k, where (n/k)^2 would
  predict a 16-fold fall from k = 10 to k = 40. At equal fan-in, capacity is
  no longer p-invariant (Amendment 16's S4 held above the floor). At
  k p = 5 the sparse cell stores 201 items against the dense cell's 1356, on
  twice the neurons.
* **The literature's own cell identifies but never completes.** At the PNAS
  2020 cell (n = 10000, k = 100, p = 0.01, k p = 1), no learning rate from
  0.10 theta to 0.40 theta completes a single item at the distinct
  criterion; rank-1 identification peaks at 112 items (beta = 0.36). At the
  dense low-k cells identification runs past the 8192-item cap at rates
  where completion is zero: there, a stored item can be told apart long
  after it can be recalled.

**Seen in the record, not registered.** At p = 0.5 the best capacity is the
FIRST grid point that completes at all (4 of 6 cells) or one step above it
(2 of 6). The optimum is the completion ONSET: the weakest write under which
an item's assembly converges. The parabola vertex there is fitted through a
zero neighbour, so it marks the cliff more than a smooth peak. This is why
the optimum scales as the convergence threshold: it IS the convergence
threshold, measured. At p = 0.05 capacity keeps rising for three grid steps
above the onset, so the vertex moves up, and that is the sparse cells'
excess fraction. A registration that measures the onset directly (the
weakest completing rate on a finer grid) and the capacity's rise above it,
as separate quantities, is what this points to.

## Amendment 18 (2026-10-02, before running): the completion onset -- where does the memory begin to complete, and is its best write the weakest that does?

Registered before any run of this amendment. Amendment 17's record, read
after its bars were judged, showed two things its bars did not ask about.
At p = 0.5 the best capacity sits at the first rate of the grid that
completes at all (4 of 6 cells) or one step above it. And the weakest
completing rate completes only inside a load WINDOW: at (4000, 60, 0.5),
completion holds between 869 and 1589 stored items at 0.168 theta and
between 31 and 1388 one grid step up; at (4000, 20, 0.5) between 982 and
1306. At p = 0.05 the first completing rate completes from the first
checkpoint, and capacity keeps rising for three grid steps above it. On
Amendment 17's grid (0.2 theta x 2^(j/4)) the first completing rate was
0.168 theta at eight of ten cells and 0.200 theta at two ((4000, 10) and
(4000, 40)): the onset brackets are (0.141, 0.168] and (0.168, 0.200].

If the optimum IS the onset, the convergence threshold sets the memory's
operating point, and the post hoc fraction 0.20 of Amendments 13-17 is the
onset seen through a coarse grid. This measures the onset directly.

**Seen before registering:** Amendment 17's record and the readings above;
a smoke run (VOID: 3 brains, cap 32, give-up 16, two rates each at
(4000, 60, 0.5) and (8000, 100, 0.05), 0.19 to 0.24 theta): no distinct
completion window at either, as expected when completion near the onset
opens only after hundreds of items.

### Protocol

`research/experiments/memory_onset.py` through the shared runner
(`python -m research.runner onset`), seeds 162 to 181 (new brains), one run
from a worktree pinned at the commit registering this amendment. The
refracted AssemblyMemory (0.5 beta, T = 8, w_max 20, arm B, ungated, the
capacity study's stimuli) at ten cells:

    above the floor:  (2000, 60, 0.5), (4000, 60, 0.5), (4000, 80, 0.5),
                      (4000, 160, 0.5), (8000, 120, 0.5), (8000, 240, 0.25)
    below the floor:  (4000, 10, 0.5), (4000, 20, 0.5),
                      (8000, 100, 0.05), (8000, 200, 0.05)

each swept over theta x 0.12 x 2^(j/12) (twelve rates to the octave) up to
0.30 theta above the floor and 0.40 theta below it, at or above 0.025.
Distinct completion and capacity as in Amendments 13-17; checkpoints from
M = 2; no stop decision before M = 32; a rate that never completes stores to
8192 items before giving up, so a window that opens late is seen. The ONSET
beta_on is a cell's weakest grid rate whose distinct-completion capacity is
at least 32 items; the onset's LOWER EDGE is the load at which that rate's
completion window opens (none if it completes from the first checkpoint);
the BEST is the rate of largest capacity.

    python -m research.runner onset \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag onset-20261001 --seeds 162 ... 181

### Bars

    OV  THE INSTRUMENT. The best capacity at (4000, 60, 0.5) is within 15% of
        Amendment 17's 1589 on new brains. Failure voids O1 to O5.
    O1  A UNIVERSAL ONSET. At every one of the ten cells the onset is found
        (not at the grid's first rate) and beta_on / theta lies in
        [0.13, 0.21], and across the ten the coefficient of variation of
        beta_on / theta is at most 0.10.
    O2  THE DENSE ONSET IS LOAD-ASSISTED. At every p = 0.5 cell the onset
        rate's completion window opens at a load of at least 64 items.
    O3  THE SPARSE ONSET IS IMMEDIATE. At both p = 0.05 cells the onset rate
        completes from the first checkpoint (no lower edge).
    O4  THE DENSE OPTIMUM IS THE ONSET. At every p = 0.5 cell the best rate
        is at most 2^(1/4) beta_on.
    O5  THE SPARSE OPTIMUM LIES ABOVE THE ONSET. At both p = 0.05 cells the
        best rate is at least 1.4 beta_on.

Reported, not judged: (8000, 240, 0.25)'s onset window and best/onset ratio
(between the dense and sparse families), every rate's window, and the onset
fraction's mean.

### Interpretation, stated now

* O1 passes: the weakest write under which an item's assembly completes is a
  fixed fraction of the convergence threshold everywhere measured, above and
  below the floor, dense and sparse: the threshold the theorems derive for
  convergence is where completion begins.
* O1 and O4 pass: in a dense area the best memory writes each item at the
  weakest plasticity that still completes; the completion-optimal rate of
  Amendments 13-17 is this onset seen on a coarse grid.
* O2 passes: in a dense area completion at the onset NEEDS load -- the first
  items do not complete, later ones do -- so stored items help later items
  converge; O3 passes: a sparse area's onset needs none.
* O5 passes and O4 passes: the sparse cells' higher optimal fraction
  (Amendment 17's T2 failure) is capacity rising above a universal onset,
  not a different onset.
* O1 fails on the band but passes on the CV: the onset is universal at a
  fraction other than the one Amendment 17's grid bracketed. O1 fails on the
  CV: there is no universal onset.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until OV and O1 to O5 are evaluated and recorded below.

## Amendment 19 (2026-10-02, before running): two capacity regimes -- does capacity change law at the floor?

Registered before any run of this amendment, together with Amendment 20,
which shares its run. Amendment 17's record showed, below the floor at
n = 4000, p = 0.5, best distinct-completion capacities of 1356, 1382, 1452
at k = 10, 20, 40: flat in k, where (n/k)^2 falls 16-fold over that range.
Above the floor at the same n and p the capacities fell with k: 1589, 1312,
1017 at k = 60, 80, 160. At equal fan-in (k p = 5) the sparse cell stored
201 items on 8000 neurons against the dense cell's 1356 on 4000. If the law
changes at the floor, capacity below it is not set by synapses per item but
by something else -- most simply by neurons, linear in n and blind to k.

**Seen before registering:** Amendments 14, 17 and the readings above;
a smoke run (VOID: 3 brains, cap 32, give-up 16, two rates each at
(4000, 10, 0.5) and (10000, 100, 0.01), 0.04 and 0.157 theta): distinct
completion 0 everywhere; rank-1 3, 3 and 32 (the cap) items.

### Protocol

`research/experiments/memory_regimes.py` (`python -m research.runner
regimes`), seeds 182 to 201 (new brains), one run, shared with Amendment 20.
Fifteen cells:

    the k sweep:     n = 4000, p = 0.5, k = 10, 14, 20, 28, 40 (below the
                     floor) and 56, 80, 112, 160 (above it; the floor is
                     k p = 24.9, k = 49.8)
    the n sweep:     k = 20, p = 0.5, n = 2000, 4000, 8000 (k p = 10, below
                     the floor at every n)
    the PNAS family: n = 10000, p = 0.01, k = 100, 200, 400, 800 (Amendment 20)

each swept over theta x 0.04 x 2^(j/4) below 0.14 theta and theta x 0.14 x
2^(j/6) from 0.14 to 0.40 theta, at or above 0.025. A store runs until BOTH
rank-1 and distinct completion have been above 0.5 and then at or below it
twice (or to 32768 items; a rate on which neither rises stops at 8192);
checkpoints from M = 2, no stop before M = 32. A cell's capacity is its
best distinct-completion capacity over its grid.

    python -m research.runner regimes \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag regimes-20261001 --seeds 182 ... 201

### Bars

    XV  THE INSTRUMENT. Best capacity within 15% of Amendment 17's at
        (4000, 40, 0.5) (1452) and (4000, 80, 0.5) (1312). Failure voids
        X1 to X3.
    X1  FLAT BELOW THE FLOOR. At n = 4000, p = 0.5, k = 10, 14, 20, 28, 40,
        every capacity lies within 15% of their mean.
    X2  LINEAR IN n BELOW THE FLOOR. At k = 20, p = 0.5, capacity multiplies
        by 1.6 to 2.5 from n = 2000 to 4000 and from 4000 to 8000.
    X3  FALLING WITH k ABOVE THE FLOOR. At n = 4000, p = 0.5, capacity
        strictly decreases over k = 56, 80, 112, 160, and capacity at
        k = 160 is at most 0.8 of capacity at k = 56.

### Interpretation, stated now

* X1 and X2 pass: below the floor the memory is NEURON-limited -- capacity
  is proportional to n and independent of k (and so of the synapses an item
  can use) -- and above it SYNAPSE-limited; the floor k p = 3 ln n is a
  change of law, not a degradation of one law.
* X1 passes, X2 fails: flat in k below the floor, but not linear in n; the
  limit is neither synapses nor neurons alone (its n-exponent is reported).
* X1 fails: capacity below the floor depends on k; Amendment 17's flatness
  was a three-point coincidence.
* X3 passes: above the floor, at fixed n and p, more neurons per item cost
  capacity even as fan-in grows.
* A failed bar is recorded as failed and not moved after the data.

## Amendment 20 (2026-10-02, before running): recognition against recall -- do they diverge as fan-in falls?

Registered before any run, on Amendment 19's run. Amendment 17's record:
below the floor at p = 0.5, rank-1 identification ran past the 8192-item
cap at rates where distinct completion was zero; at the PNAS 2020 cell
(k p = 1) the memory identified up to 112 items and completed none at any
rate. Human memory dissociates the two -- recognition capacity far exceeds
recall -- and the dual-process reading predicts that the gap grows as the
substrate weakens. Here the substrate's strength is the fan-in k p.

R is a cell's RATIO of rank-1 capacity to distinct-completion capacity,
both read at the cell's completion-optimal rate (the best write for recall).
A distinct-completed item is also rank-1, so R >= 1 by construction; the
bars are about how R moves with fan-in, never about R >= 1.

**Seen before registering:** as Amendment 19.

### Protocol

Amendment 19's run (identification's capacity is read because a store
continues until rank-1 also falls). Rank-1 censored at 32768 items enters
as its lower bound.

### Bars

    R1  THE GAP GROWS AS FAN-IN FALLS. Over the nine k-sweep cells, the
        Spearman correlation of R with k p is at most -0.7.
    R2  RECALL NEEDS FAN-IN, RECOGNITION DOES NOT. In the PNAS family
        (n = 10000, p = 0.01), the best rank-1 capacity is at least 32 items
        at all four cells; the best distinct-completion capacity is 0 at
        k p = 1 and at least 32 items at k p = 8.
    R3  THE DIVERGENCE IS LARGE. R at k = 10 (k p = 5) is at least 3 times
        R at k = 160 (k p = 80).

### Interpretation, stated now

* R1 to R3 pass: recognition and recall dissociate in the substrate itself,
  and the dissociation deepens as fan-in falls: a weakly connected area can
  tell its items apart long after it can recall them; the regimes the
  literature runs at (k p = 1 to 10) are recognition memories.
* R2 passes alone: there is a fan-in threshold for recall that recognition
  does not have, but the ratio does not trend smoothly.
* R1 fails: recognition and recall track each other across fan-in; the
  Amendment 17 readings were a stopping artefact (completion's stop censored
  rank-1).
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until XV, X1 to X3 and R1 to R3 are evaluated and
recorded below.

### Amendment 18 result (2026-10-02)

One run from a worktree pinned at b72d1b02 (this registration, 9fa58851,
plus two engine changes that replay committed studies exactly: launches
sized against the card's free memory, and in-place compaction when a sweep
drops its finished rates), the registered seeds 162 to 181, 9 minutes:
[record](../../results/runs/memory.onset/onset-20261002-r3/results.json),
[log](../../results/logs/onset-regimes-20261002-r3.log).
**Disclosed:** two earlier attempts on the same seeds were stopped. The
first (tag `onset-20261002`, from 9fa58851) paged GPU memory to system RAM
and was stopped after two cells
([log](../../results/logs/onset-20261002-aborted.log)); the second
(`onset-20261002-r2`, from feb649de) ran out of memory when dropping
finished rates at (8000, 120, 0.5)
([log](../../results/logs/onset-regimes-20261002-r2-failed.log)). Their
reservations are kept. Every line the earlier attempts printed is
reproduced exactly by this run.

theta = sqrt((1 - p) ln n / (p k)); the onset is the weakest rate of the
twelve-to-the-octave grid that completes at least 32 distinct items; its
lower edge is the load at which that rate's completion window opens:

    cell (n, k, p)      k p  onset/theta  lower edge  best   best/theta  best/onset
    (2000, 60, 0.5)     30   0.170        295         507    0.180       1.060
    (4000, 60, 0.5)     30   0.170        714         1597   0.180       1.059
    (4000, 80, 0.5)     40   0.170        614         1435   0.180       1.060
    (4000, 160, 0.5)    80   0.170        158         1083   0.180       1.060
    (8000, 120, 0.5)    60   0.170        1098        4243   0.202       1.190
    (8000, 240, 0.25)   60   0.180        17          1160   0.214       1.189
    (4000, 10, 0.5)     5    0.180        961         1561   0.191       1.060
    (4000, 20, 0.5)     10   0.180        633         1554   0.180       1.000
    (8000, 100, 0.05)   5    0.160        none        199    0.285       1.782
    (8000, 200, 0.05)   10   0.160        3           185    0.254       1.587

    OV  THE INSTRUMENT                                                PASS
        best at (4000, 60, 0.5): 1597 against Amendment 17's 1589.
    O1  A UNIVERSAL ONSET                                             PASS
        every onset found, 0.160 to 0.180 theta; mean 0.171, coefficient
        of variation 0.041 (bar: [0.13, 0.21], CV <= 0.10).
    O2  THE DENSE ONSET IS LOAD-ASSISTED                              PASS
        at the six p = 0.5 cells the onset window opens at 158 to 1098
        stored items (bar: >= 64).
    O3  THE SPARSE ONSET IS IMMEDIATE                                 FAIL
        (8000, 100, 0.05) completes from the first checkpoint; (8000, 200,
        0.05)'s window opens at 3 items -- immediate in effect, but not
        "no lower edge" as registered.
    O4  THE DENSE OPTIMUM IS THE ONSET                                FAIL
        best/onset 1.000 to 1.060 at five of the six p = 0.5 cells (the
        best within one fine step of the onset); 1.190 at (8000, 120, 0.5),
        where the best is exactly three steps (2^(1/4)) above the onset and
        the grid's four-decimal rates make the ratio 1.1897 against the
        bar's 1.1892.
    O5  THE SPARSE OPTIMUM LIES ABOVE THE ONSET                       PASS
        best/onset 1.78 and 1.59 at the two p = 0.05 cells (bar: >= 1.4).

**Reading.** The weakest write under which the refracted memory completes
its items is a fixed fraction of the convergence threshold, 0.16 to 0.18
theta, at all ten cells: dense and sparse, above and below the floor, k p
from 5 to 80. The convergence threshold the assembly-calculus theorems
derive for projection is where completion begins -- the theorems' constant
sets the memory's operating point, everywhere this repository has measured
it.

What differs between regimes is not the onset but what happens above it,
and how the onset opens. In a dense area (p = 0.5) completion at the onset
needs LOAD: the first several hundred items stored at that rate do not
complete, later ones do (lower edges 158 to 1098; 295 and 714 at n = 2000
and 4000 with k p = 30, 1098 at n = 8000), and the best capacity sits within one fine step of the onset at five
of six cells. As p falls the load assistance vanishes -- 17 items at
p = 0.25, 0 to 3 at p = 0.05 -- and the best write moves up, 1.6 to 1.8
times the onset in the sparse cells. Amendment 17's T2 failure (the
sparse cells' optimum at 0.27 to 0.30 theta) is this: a universal onset,
with capacity that keeps rising above it in sparse areas. Amendments 13 to
17's "0.20 theta" was the dense onset seen through a coarse grid: 0.17,
one quarter-octave step below.

Two bars fail on their letter, not their substance, and are recorded as
failed: a 3-item lower edge where "none" was registered (O3), and a
rounding of 0.0005 at a best exactly 2^(1/4) above its onset (O4).

### Amendments 19 and 20 result (2026-10-02)

One run from the worktree pinned at b72d1b02, the registered seeds 182 to
201, 10 minutes:
[record](../../results/runs/memory.regimes/regimes-20261002-r3/results.json),
[log](../../results/logs/onset-regimes-20261002-r3.log) (after Amendment
18's lines). **Disclosed:** an attempt from feb649de (`regimes-20261002-r2`)
was stopped during its first cell when Amendment 18's run beside it failed;
its reservation is kept. Each cell's capacity is its best distinct-completion
capacity over its grid; rank-1 at 32768 is the cap (a lower bound, marked *);
R is rank-1 over distinct capacity, both at the cell's recall-optimal rate:

    cell (n, k, p)      k p  floor  distinct (rate)    rank-1 at own best    R
    (4000, 10, 0.5)     5    below  1314 (0.180)       32768* (0.036)        2.97
    (4000, 14, 0.5)     7    below  1148 (0.152)       32768* (0.031)        4.19
    (4000, 20, 0.5)     10   below  1513 (0.128)       32768* (0.031)        3.22
    (4000, 28, 0.5)     14   below  1383 (0.108)       29175 (0.031)         3.51
    (4000, 40, 0.5)     20   below  1473 (0.090)       19880 (0.031)         2.87
    (4000, 56, 0.5)     28   above  1392 (0.076)       14732 (0.026)         2.60
    (4000, 80, 0.5)     40   above  1468 (0.057)       9959 (0.026)          2.46
    (4000, 112, 0.5)    56   above  1239 (0.054)       5182 (0.031)          1.97
    (4000, 160, 0.5)    80   above  1059 (0.040)       2511 (0.026)          1.67
    (2000, 20, 0.5)     10   below  508 (0.122)        14368 (0.029)         3.61
    (8000, 20, 0.5)     10   below  3222 (0.149)       32768* (0.045)        3.12
    (10000, 100, 0.01)  1    below  0                  114 (0.342)           --
    (10000, 200, 0.01)  2    below  3 (0.336)          179 (0.203)           44.9
    (10000, 400, 0.01)  4    below  52 (0.423)         207 (0.144)           1.08
    (10000, 800, 0.01)  8    below  40 (0.237)         137 (0.102)           1.43

    XV  THE INSTRUMENT                                                PASS
        1473 against 1452 at (4000, 40, 0.5); 1468 against 1312 at
        (4000, 80, 0.5) (+12%).
    X1  FLAT BELOW THE FLOOR                                          FAIL
        1314, 1148, 1513, 1383, 1473 at k = 10 to 40 (mean 1366): k = 14
        lies 16% below the mean (bar: 15%).
    X2  LINEAR IN n BELOW THE FLOOR                                   FAIL
        508, 1513, 3222 at n = 2000, 4000, 8000: x2.98 and x2.13 per
        doubling (bar: 1.6 to 2.5); n-exponents 1.58 and 1.09.
    X3  FALLING WITH k ABOVE THE FLOOR                                FAIL
        1392, 1468, 1239, 1059 at k = 56, 80, 112, 160: not strictly
        decreasing (k = 56 to 80 rises), though k = 160 is 0.76 of k = 56.
    R1  THE GAP GROWS AS FAN-IN FALLS                                 PASS
        Spearman of R with k p over the nine k-sweep cells: -0.883 (bar:
        <= -0.7); R 2.9 to 4.2 below the floor, 1.7 to 2.6 above it.
    R2  RECALL NEEDS FAN-IN, RECOGNITION DOES NOT                     PASS
        PNAS family: rank-1 114, 179, 207, 137 items at k p = 1, 2, 4, 8;
        distinct completion 0 at k p = 1 and 40 at k p = 8.
    R3  THE DIVERGENCE IS LARGE                                       FAIL
        R(k = 10) / R(k = 160) = 2.97 / 1.67 = 1.78 (bar: >= 3).

**Reading (Amendment 19).** Capacity does not change law at the floor as
registered. What the k sweep shows instead is that at fixed n and p, a
memory written at its own best rate stores nearly the same number of items
whatever its assembly size: 1059 to 1513 over a sixteen-fold range of k
(k p 5 to 80), falling only at the largest assemblies, where the synapse-
count form n^2 p / (k ln(n/k)) would fall about thirty-fold and (n/k)^2
256-fold. Below the floor, capacity grows faster than linearly in n
(exponent 1.6, then 1.1). Amendment 17's three-point flatness was the low
end of this near-flat line, not a regime of its own.

**Reading (Amendment 20).** Recognition and recall dissociate, and the
dissociation deepens as fan-in falls (R1), though by less than registered
at the recall-optimal write (R3: 1.8-fold, not 3). In the PNAS family recall
appears only from k p = 4, while recognition holds 114 to 207 items from
k p = 1 (R2): the parameters the published simulations use are a
recognition memory.

**Seen in the record, not registered.** Read each at its OWN best write,
the two capacities part far more than R shows: recognition's best write is
much weaker (0.03 to 0.05 against recall's 0.04 to 0.18), and there it
holds at least 32768 items (the cap) at k p <= 10, 29175 at k p = 14, and
falls to 2511 at k p = 80 -- at least 25 times recall's capacity below the
floor against 2.4 times at k p = 80. A weakly connected memory written
weakly is a vast recognition memory and a modest recall memory; the same
memory written for recall gives up most of its recognition. Registering
the two capacities each at its own optimum, with a cap high enough not to
censor recognition, is what this points to.

## Amendment 21 (2026-10-02, before running): is capacity at the best write set by the in-degree d = n p alone?

Registered before any run of this amendment. Read across the records of
Amendments 13, 14, 16, 17, 18 and 19 after their bars were judged, the best
distinct-completion capacity of the 46 cells with in-degree d = n p >= 1000
(n 1333 to 8000, k 10 to 240, p 0.125 to 0.75, k p 5 to 120) follows

    C = 0.0135 d^1.51                (log-scale residual sd 0.143, x1.15)

and adding n/k to the fit barely helps (its exponent 0.08, residual x1.14).
Amendment 16's four cells at d = 1000 (n 1333 to 8000, p 0.75 to 0.125)
store 436 to 497 items; Amendment 19's (2000, 20, 0.5), also d = 1000,
stores 508. Below d = 1000 the relation does not hold as written (d = 400 at
p = 0.05 sits ~1.6x above it; at d = 100 recall barely exists), and cells
with small n/k (<= 25) sit 20-35% below it. The synapse-count form of
Amendment 14 (n^2 p / (k ln(n/k))) and (n/k)^2 both predict a strong fall
with k at fixed n and p, which Amendment 19 did not find (1059 to 1513 over
k = 10 to 160).

If capacity at the optimum is set by the in-degree alone, what a refracted
area can store is fixed by how many synapses each neuron receives from it --
not by its assembly size, its neuron count or its connection probability
separately -- and grows as d^1.5.

**Seen before registering:** the records above and the fit; a smoke run
(VOID: 3 brains, cap 32, two rates each at (2500, 25, 0.4) and (12000, 60,
0.5), 0.18 and 0.20 theta): distinct completion's window opened only at
(2500, 25, 0.4), 0.195 theta, with its upper edge at 24 items, below the
cap.

### Protocol

`research/experiments/memory_degree_law.py` (`python -m research.runner
degree-law`), seeds 202 to 221 (new brains), one run from a worktree pinned
at the commit registering this amendment. The refracted AssemblyMemory
(0.5 beta, T = 8, w_max 20, arm B, ungated, the capacity study's stimuli) at
nine cells never measured, eight of them new and six at in-degrees never
measured; within each in-degree n/k varies two- to fourfold, so a law in n/k
and the in-degree law disagree at equal d:

    d = 1000:  (2500, 25, 0.4)   n/k 100      (5000, 100, 0.2)   n/k 50
    d = 1500:  (3000, 15, 0.5)   n/k 200      (6000, 60, 0.25)   n/k 100
               (12000, 240, 0.125) n/k 50
    d = 3000:  (6000, 30, 0.5)   n/k 200      (12000, 240, 0.25) n/k 50
    d = 6000:  (12000, 60, 0.5)  n/k 200
    the instrument: (4000, 60, 0.5)            (Amendment 18: 1597)

each swept over theta x 0.15 x 2^(j/8) up to 0.30 theta (nine rates), theta
= sqrt((1 - p) ln n / (p k)): Amendment 18 put every onset at 0.16 to 0.18
theta and the best write within 1.8 times it. Distinct completion and
capacity as in Amendments 13 to 20; checkpoints from M = 2; no stop before
M = 32; cap 3 C_pred + 256; a rate that never completes stores to
max(8192, 2 C_pred) items. Predictions, fixed now from C = 0.0135 d^1.51:

    d = 1000: 457    d = 1500: 844    d = 3000: 2403    d = 6000: 6845

    python -m research.runner degree-law \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag degree-law-20261002 --seeds 202 ... 221

### Bars

    DV  THE INSTRUMENT. The best capacity at (4000, 60, 0.5) lies within 15%
        of Amendment 18's 1597. Failure voids D1 to D3.
    D1  THE PREDICTION. At each of the eight new cells the best capacity lies
        within 25% of its predicted value.
    D2  THE IN-DEGREE ALONE. Within d = 1000, 1500 and 3000, each cell's best
        capacity lies within 20% of its group's mean, while n/k varies two-
        to fourfold.
    D3  THE EXPONENT. The least-squares slope of log C on log d over the
        eight new cells lies in [1.35, 1.65].

### Interpretation, stated now

* D1 to D3 pass: the capacity of a refracted area written at its best rate
  is set by its neurons' in-degree, C ~ 0.0135 d^1.5, independent of
  assembly size and of how n and p combine to give d; it replaces the
  (n/k)^2 and synapse-count forms as the memory's scaling law.
* D2 and D3 pass, D1 fails: the in-degree law holds with a different
  constant or a drift outside the fitted range (d = 6000).
* D2 fails: n/k (or p) matters at equal in-degree; the fit's d-alone form
  was an artefact of which cells had been measured.
* D3 fails: capacity is set by d but not as d^1.5 (the slope is reported).
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until DV and D1 to D3 are evaluated and recorded below.

### Amendment 21 result (2026-10-02)

One run from a worktree pinned at b6e9ca41, the registered seeds 202 to
221:
[record](../../results/runs/memory.degree-law/degree-law-20261002/results.json),
[log](../../results/logs/degree-law-20261002.log). Best distinct-completion
capacity over each cell's grid against the prediction 0.0135 d^1.51:

    cell (n, k, p)      d      n/k   best    predicted  ratio
    (2500, 25, 0.4)     1000   100   417     457        0.911
    (5000, 100, 0.2)    1000   50    439     457        0.960
    (3000, 15, 0.5)     1500   200   971     844        1.150
    (6000, 60, 0.25)    1500   100   811     844        0.961
    (12000, 240, 0.125) 1500   50    944     844        1.119
    (6000, 30, 0.5)     3000   200   2871    2403       1.195
    (12000, 240, 0.25)  3000   50    2739    2403       1.140
    (12000, 60, 0.5)    6000   200   8605    6845       1.257
    (4000, 60, 0.5)     2000   67    1608    (instrument: 1597)

    DV  THE INSTRUMENT                                                PASS
        1608 against Amendment 18's 1597.
    D1  THE PREDICTION                                                FAIL
        seven of eight new cells within 25% (0.91 to 1.20); the cell at
        d = 6000, beyond every in-degree measured before, reads 1.257.
    D2  THE IN-DEGREE ALONE                                           PASS
        d = 1000: 417, 439 (n/k 100, 50); d = 1500: 971, 811, 944 (n/k 200,
        100, 50); d = 3000: 2871, 2739 (n/k 200, 50) -- each within 20% of
        its group's mean (largest deviation 11%).
    D3  THE EXPONENT                                                  FAIL
        slope of log C on log d over the eight new cells: 1.669 (bar:
        1.35 to 1.65).

**Reading.** The central claim holds: at equal in-degree the capacity is
the same whatever the assembly size and however n and p combine to give d
-- with n/k varying fourfold at d = 1500, the three cells agree within 11%,
where (n/k)^2 would differ sixteenfold. What a refracted area written at its
best rate can store is set by how many synapses each neuron receives from
it. The registered law's numbers are slightly off: capacity grows a little
faster than d^1.51 (1.67 over the new cells), so the fixed prediction
undershoots by 26% at d = 6000, just outside its band. The in-degree law is
adopted in form, not in its fitted constants.

**Seen in the record, not registered.** Refitted on every cell measured
with d >= 1000 (55 cells from Amendments 13 to 21): C = 0.0099 d^1.554,
residual x1.15.

## Amendment 22 (2026-10-03, before running): recognition and recall, each at its own best write

Registered before any run of this amendment. Amendment 20 compared
recognition (rank-1: the recall lies nearer the cued item than any other
stored item) with recall (distinct completion) at the RECALL-optimal write.
Read after its bars, its record showed recognition's own best write at the
bottom of its grid (0.026 to 0.045), where int8 counts stopped the grid, and
there recognition held at least 32768 items (the cap) at k p <= 10 against
about 1300 for recall, falling to 2511 at k p = 80. int16 counts (a38c1977)
now let the write go as weak as needed. A probe before registering (20
brains, (4000, 10, 0.5)): at beta = 0.0015 and 0.0042 rank-1 starts at
0.50 to 0.53 with two items stored -- chance for two -- and decays as chance
does; recognition does not form at so weak a write, so its optimum, if
interior, lies between ~0.004 and the 0.026 to 0.045 above.

A rank-1 reading of a few items is chance: with M items stored a random
recall is rank-1 with probability 1/M, which is the 0.5 bar at M = 2. The
stop rule's guard (no stop decision before M = 32) and the cap-scale
capacities measured here keep that floor far below every bar.

**Seen before registering:** the records and readings above; a smoke run
(VOID: 3 brains, cap 32, two rates each at (4000, 10, 0.5) and (4000, 160,
0.5)): rank-1 reached the cap of 32 at 0.21 theta in both cells and was 0 at
the weak rates (0.026 and 0.053 theta); distinct completion reached 32 at
(4000, 160, 0.5), 0.21 theta, and 0 elsewhere.

### Protocol

`research/experiments/memory_recognition.py` (`python -m research.runner
recognition`), seeds 242 to 261 (new brains), one run from a worktree pinned
at the commit registering this amendment. The refracted AssemblyMemory
(0.5 beta, T = 8, w_max 20, arm B, ungated, the capacity study's stimuli),
int16 counts wherever a rate's clip binds past count 127, at seven cells:

    the k sweep:  n = 4000, p = 0.5, k = 10, 20, 40, 80, 160 (k p 5 to 80;
                  the in-degree d = 2000 throughout)
    the n sweep:  k = 40, p = 0.5, n = 2000, 8000 (d = 1000, 4000)

each swept over beta = 0.0015 x 2^(j/2) up to 0.30 theta, theta = sqrt((1 -
p) ln n / (p k)) (12 to 16 rates). A store runs until both rank-1 and
distinct completion have been above 0.5 and then at or below it twice, to a
cap of 131072 items (32768 at k >= 80); a rate on which neither rises stops
at 8192; checkpoints from M = 2; no stop before M = 32. Each metric's
capacity is its best over the cell's grid, each at its own rate; the RATIO
is recognition's capacity over recall's.

    python -m research.runner recognition \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag recognition-20261003 --seeds 242 ... 261

### Bars

    QV  THE INSTRUMENT. Recall's best capacity at (4000, 40, 0.5) lies
        within 15% of Amendment 19's 1473. Failure voids Q1 to Q3.
    Q1  RECOGNITION OUTRUNS RECALL. At every cell the ratio is at least 2.
    Q2  THE GAP WIDENS AS FAN-IN FALLS. Over the five k-sweep cells the
        Spearman correlation of the ratio with k p is at most -0.9, and the
        ratio at k = 10 is at least 5 times the ratio at k = 160.
    Q3  RECOGNITION WANTS A WEAKER WRITE. At every cell recognition's best
        rate is at most half of recall's.

Reported, not judged: whether recognition's optimum is interior or at the
grid's floor; recognition's capacity against d over the n sweep; censoring
at the cap (a lower bound, marked).

### Interpretation, stated now

* Q1 to Q3 pass: the same refracted area is two memories, and which one it
  is depends on how hard it is written: a weak write makes it a recognition
  memory of far greater capacity than the recall memory a stronger write
  makes it, and the less fan-in it has, the more it is the former.
* Q1 and Q3 pass, Q2 fails: recognition outruns recall at its own weaker
  write, but by a margin that does not depend on fan-in.
* Q1 fails: at their own optima the two capacities are close; Amendment 20's
  gap was the cost of reading recognition at recall's write.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until QV and Q1 to Q3 are evaluated and recorded below.

### Amendment 22 result (2026-10-03): VOID, the stop rule cut recognition short

The run (from d2f6d75e, seeds 242 to 261) was stopped after two of seven
cells, its log kept
([log](../../results/logs/recognition-20261003-void.log); the reservation
`recognition-20261003` holds no record). The instrument was defective. A
store stops once a stop metric has been above 0.5 and then at or below it at
two checkpoints from M = 32; but with M items stored, rank-1 is right with
probability 1/M by chance -- the 0.5 bar at M = 2 -- and with twenty brains
the mean reached 0.53 at M = 2 on some rates. That chance reading armed the
stop, the next readings fell below 0.5 as chance does, and the store ended
at M = 32, before a recognition window that opens only after thousands of
items. The log shows it: at (4000, 20, 0.5) rank-1 read 109253 at beta 0.017
and 2 at 0.024, 0.034 and 0.048, then 13515 at 0.068. A probe on the same
brains confirmed the cause: at (4000, 20, 0.5), beta 0.024, the store under
the registered rule ended at 32 items with no window; with no reading before
M = 32 allowed to arm the stop, it ran to 98304 items and found recognition
between 5081 and 60113 stored items. Distinct completion is not affected (it
needs 0.8 recovery, which chance does not give). Nothing of Amendment 22 is
judged.

The same stop rule ran Amendment 20's stores; its registered R bars read
rank-1 at the recall-optimal write, whose windows open from the first items,
and stand, but its post hoc readings of recognition at its own best write
may be underestimates where a chance reading stopped a store early.

## Amendment 23 (2026-10-03, before running): recognition and recall, each at its own best write, with a stop rule chance cannot arm

Registered before any run of this amendment. Amendment 22's question,
protocol and bars (QV, Q1, Q2, Q3), unchanged but for the instrument, on new
brains (seeds 262 to 281): `memory_recognition.py` protocol version 2.

* No reading before M = 32 arms the stop (`seen_from` = 32), where chance
  rank-1 is at most 1/32.
* Recognition's load window is read from the checkpoints at M >= 32; a
  rank-1 reading at fewer items is chance-level and is not a capacity.

**Seen before registering:** Amendment 22's two VOID cells and the probe
above (at (4000, 10, 0.5) recognition read 81366 items at beta 0.034 and
chance-level below 0.024; at (4000, 20, 0.5), with the corrected rule, a
window from 5081 to 60113 items at beta 0.024).

    python -m research.runner recognition \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag recognition-20261003-a23 --seeds 262 ... 281

The run is UNJUDGED until QV and Q1 to Q3 are evaluated and recorded below.

### Amendment 23 result (2026-10-03)

One run from a worktree pinned at 2e9ee09a, seeds 262 to 281:
[record](../../results/runs/memory.recognition/recognition-20261003-a23/results.json),
[log](../../results/logs/recognition-20261003-a23.log). Each capacity at its
own best rate over the cell's grid (none censored at the cap):

    cell (n, k, p)     k p  recognition (rate, /theta)   recall (rate, /theta)   ratio
    (4000, 10, 0.5)    5    81290 (0.034, 0.037)         1350 (0.192, 0.211)     60.2
    (4000, 20, 0.5)    10   109640 (0.017, 0.026)        1292 (0.136, 0.211)     84.8
    (4000, 40, 0.5)    20   51316 (0.012, 0.026)         1339 (0.096, 0.211)     38.3
    (4000, 80, 0.5)    40   27917 (0.006, 0.019)         1299 (0.068, 0.211)     21.5
    (4000, 160, 0.5)   80   12521 (0.003, 0.013)         837 (0.048, 0.211)      15.0
    (2000, 40, 0.5)    20   39578 (0.004, 0.010)         458 (0.096, 0.220)      86.3
    (8000, 40, 0.5)    20   83381 (0.024, 0.051)         4133 (0.096, 0.203)     20.2

    QV  THE INSTRUMENT                                                PASS
        recall at (4000, 40, 0.5): 1339 against Amendment 19's 1473 (-9%).
    Q1  RECOGNITION OUTRUNS RECALL                                    PASS
        ratio 15.0 to 86.3 at every cell (bar: >= 2).
    Q2  THE GAP WIDENS AS FAN-IN FALLS                                FAIL
        Spearman of the ratio with k p over the k sweep: -0.90 (bar: <=
        -0.9, met); ratio at k = 10 over k = 160: 60.2 / 15.0 = 4.0 (bar:
        >= 5, missed) -- the ratio peaks at k = 20 (84.8), not k = 10.
    Q3  RECOGNITION WANTS A WEAKER WRITE                              PASS
        recognition's best rate is 1/40 to 1/6 of recall's at every cell.

**Reading.** At their own best writes the refracted area holds 15 to 86
times as many items for recognition as for recall, and recognition's best
write is six to forty times weaker -- below the int8 range the earlier
studies could reach. The same circuit is two memories, chosen by how hard
it is written. Below recognition's onset nothing is recognised (rank-1 at
chance); at its onset recognition jumps to its maximum and then declines
as the write strengthens -- the same cliff-then-decline that recall shows at
its own onset (Amendment 18), at a write several times weaker. The gap
widens as fan-in falls overall (Spearman -0.90) but not monotonically at the
sparse end, so Q2 fails.

**Seen in the record, not registered.** At the constant in-degree of the k
sweep (d = 2000) the two memories follow different laws. Recall stays put
(1292 to 1350 over k = 10 to 80), as Amendment 21's in-degree law says.
Recognition halves with each doubling of k from k = 20 (109640, 51316,
27917, 12521): it scales with n/k, the area's sparsity, not with d. And its
best write falls as 1/k (0.034, 0.017, 0.012, 0.006, 0.003 for k = 10 to
160): k beta, the total plasticity an item spends, is roughly constant
(0.3 to 0.5) at recognition's optimum, where recall's optimum is a fixed
fraction of the convergence threshold.

## Amendment 24 (2026-10-03, before running): is the recall onset a phase transition? Finite-size scaling and critical slowing down

Registered before any run of this amendment. Amendment 18 put recall's onset
at 0.16 to 0.18 theta at ten cells, with capacity jumping from nothing to its
maximum within one step of a twelve-per-octave grid, and the measured phase
diagram (assembly_statmech.tex, Propositions on the write-load plane) shows it
as the boundary of the recall region at every cell. A sharp jump is not yet a
transition. Two signatures separate a phase transition from a threshold that
only looks sharp at the sizes measured:

* **The pseudo-critical point sharpens with size.** Each brain has its own
  onset -- the weakest rate at which its own distinct completion exceeds one
  half at some load of at least 32 items. At a transition the spread of these
  onsets across brains shrinks as n grows and their mean converges.
* **Critical slowing down.** The read-out's relaxation time grows on
  approach to the onset from above. A probe before registering (5 brains,
  (2000, 60, 0.5), 200 items, 32 frozen rounds) showed the read-out ends in a
  2-cycle as often as at a fixed point: at beta = 0.1, 52% at a fixed point
  and 92% at period <= 2 by round 31; at the onset rate (0.0604), 4% and 32%
  -- most read-outs there reach no short orbit at all. Settling is therefore
  defined as the first round whose winner set equals the one TWO rounds
  before (a fixed point or a 2-cycle), and the share that never settles
  within 32 rounds is reported.

**Seen before registering:** Amendment 18's record and the phase diagram;
the settling probe above; a smoke run (VOID: 3 brains, cap 32, two rates
each at (2000, 60, 0.5) and (16000, 60, 0.5), 0.17-0.18 theta), whose
read-outs, scored by the earlier fixed-point definition, never settled
within 32 rounds -- the reason for the period-2 definition.

### Protocol

`research/experiments/memory_criticality.py` (`python -m research.runner
criticality`), seeds 282 to 301 (new brains), one run from a worktree pinned
at the commit registering this amendment. The refracted AssemblyMemory
(0.5 beta, T = 8, w_max 20, arm B, ungated) at four cells of fixed fan-in
k p = 30, p = 0.5, k = 60: n = 2000, 4000, 8000, 16000. Each is swept over
theta x 0.14 x 2^(j/24) to 0.22 theta (24 rates to the octave, 16 rates).
Distinct completion and capacity as in Amendments 13 to 21; checkpoints from
M = 2, no stop before M = 32; cap 3 C_pred + 256 with C_pred = 0.0135 d^1.51
(Amendment 21); a rate that never completes stores to max(8192, 2 C_pred).
At every checkpoint every sampled cue is also read for 32 frozen rounds,
separately from the registered 8-round read, for its settling round (the
first round equal to the one two before; 33 if none) and whether it settles.

A brain's onset is read from the record's per-seed values; the grid resolves
onsets to about 0.005 theta (one step near 0.17 theta), so spreads at that
floor are not distinguished. A rate's settling time is the mean settling round
over the checkpoints inside its distinct-completion window.

    python -m research.runner criticality \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag criticality-20261003 --seeds 282 ... 301

### Bars

    CV  THE INSTRUMENT. The best capacity at (4000, 60, 0.5) lies within 15%
        of Amendment 18's 1597. Failure voids F1 to F3.
    F1  THE ONSET SHARPENS. The across-brain standard deviation of the onset
        (in units of theta) does not rise by more than 10% from one n to the
        next, and at n = 16000 it is at most 0.6 of its value at n = 2000.
    F2  THE ONSET CONVERGES. The mean onset lies in [0.13, 0.21] theta at
        every n, and its value at n = 16000 lies within 10% of its value at
        n = 8000.
    F3  THE READ-OUT SLOWS AT THE ONSET. At every n, the settling time at
        the onset rate (the weakest rate whose capacity reaches 32 items) is
        at least 1.5 times the settling time half an octave above it.

Reported, not judged: each rate's share of read-outs that never settle,
by n.

### Interpretation, stated now

* F1 to F3 pass: the recall onset behaves as a phase transition -- a
  pseudo-critical point that sharpens and converges with size, and a
  relaxation time that grows on approach; 0.17 theta is a critical
  coupling, not a fitted threshold, and the next step is its exponents.
* F1 and F2 pass, F3 fails: a sharpening threshold without slowing -- a
  first-order-like jump rather than a continuous transition.
* F1 fails: the onset's spread does not shrink with size at this fan-in;
  the jump is a threshold of the finite system.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until CV and F1 to F3 are evaluated and recorded below.

### Amendment 24 result (2026-10-06)

One run from a worktree pinned at 78b983e0, seeds 282 to 301:
[record](../../results/runs/memory.criticality/criticality-20261003/results.json),
[log](../../results/logs/memory-criticality-20261003.log). Per-brain onsets
in units of theta (20 of 20 brains have one at every cell); settling times
in rounds (33 = never settled in 32):

    n        best capacity   onset mean   onset sd   settle at onset   settle half-octave up   ratio
    2000     496             0.1680       0.0031     21.40             19.32                   1.11
    4000     1613            0.1627       0.0035     21.71             22.33                   0.97
    8000     4543            0.1656       0.0041     22.99             18.62                   1.23
    16000    12973           0.1704       0.0030     22.56             16.23                   1.39

    CV  THE INSTRUMENT                                                PASS
        1613 at (4000, 60, 0.5) against Amendment 18's 1597 (+1%).
    F1  THE ONSET SHARPENS                                            FAIL
        sd 0.0031, 0.0035, 0.0041, 0.0030: rises 17% from 4000 to 8000
        (bar: no rise over 10%), and 16000's is 0.97 of 2000's (bar <= 0.6).
    F2  THE ONSET CONVERGES                                           PASS
        means 0.163 to 0.170 theta (band [0.13, 0.21]); 16000 / 8000 = 1.029
        (bar: within 10%).
    F3  THE READ-OUT SLOWS AT THE ONSET                               FAIL
        ratios 0.97 to 1.39 (bar: >= 1.5 at every n).

**Reading.** The onset converges: a critical coupling of 0.163 to 0.170
theta at every size over an eightfold range of n, with no drift between the
two largest. Both failures say the instruments could not see the
transition, not that it is absent.

F1: at every n the brains' onsets lie within about one grid step of one
another (sd 0.003 to 0.004 theta, about 2% of the onset; the step near
0.17 theta is 0.005 theta). The onset is already as sharp as this grid
resolves at n = 2000, so any sharpening with size lies below its
resolution. A finite-size study of this onset needs a grid that resolves
0.001 theta, or a per-brain bisection, around the onset.

F3: the period <= 2 settling time is dominated by read-outs that never
settle. The share that reaches no fixed point or 2-cycle within 32 rounds is
0.42 to 0.88 at every rate of every cell, well above the onset too, and it
falls smoothly with the write, with no feature at the onset. Read-outs that
complete their item (distinct completion above one half) keep changing a few
marginal winners round after round: exact-set statistics are tie-fragile, as
the exact@L tables of the sequence studies were. The registered measure
therefore cannot carry critical slowing either way. The saddle-node
conjecture (assembly_statmech.tex) predicts slowing in the OVERLAP: the
rounds a read-out takes for its overlap with the stored item to stop
changing. That is a different instrument, and a separate registration.

**Seen in the record, not registered.** Capacity at fixed k = 60, p = 0.5
rises 3.25x, 2.82x, 2.86x per doubling of n (Amendment 21's in-degree law
predicts 457, 1303, 3711, 10568; measured 496, 1613, 4543, 12973, i.e. 9 to
24% above it, with the excess growing with n). The unsettled share at a
given fraction of theta is higher at larger n.

## Amendment 25 (2026-10-03, before running): when must the write happen? Burst-timing-dependent plasticity and the two controls that take it apart

Registered before any run of this amendment. The memory's write is
spike-timing plasticity at the scale of one round: every round, a synapse
whose pre fired the round before and whose post fires now gains a count, and
the counts feed straight back into the item's next round. Burst-timing-
dependent plasticity (Butts, Kanold & Shatz 2007, PLoS Biol 5:e61, at
developing retinogeniculate synapses: potentiation by the coincidence of
pre- and postsynaptic BURSTS over a window of about a second, order within
it largely ignored) departs from it twice: it is gated on bursts and
symmetric in time, and it acts on the item's activity as a whole rather than
round by round. Four write rules separate the departures
(`AssemblyMemory(write_rule=...)`, `burst_min = 2`):

* `round` -- online, every firing, causal: the memory's write.
* `online_burst` -- online and causal, but only between neurons that have
  already fired twice in the item (the pre by the round before, the post
  counting this round): burst gating alone. With `burst_min = 1` it is the
  round write bit for bit (tested).
* `deferred` -- the round write's own counts (pre the round before, post
  now), written AFTER the item's eight rounds, so nothing written feeds back
  into them: deferral alone. Its counts equal the recorded rounds'
  transition counts (tested).
* `burst` -- deferred and symmetric: one count, both directions, between
  every two present neurons that fired in at least two of the item's rounds,
  and one stimulus potentiation for each (the BTDP rule; tested).

**Seen before registering** (5 brains, seeds 900 to 904, outside every
registered set; (2000, 60, 0.5); disclosed in full):

* A sweep of 0.1 to 3.2 theta (4 rates to the octave) stored to 1024 items:
  the round write's capacity peaked at 439 (0.2 theta) and fell to zero
  above 1.9 theta; `burst`, `deferred` and `online_burst` completed
  nothing at any rate, at any load from M = 2.
* After 200 items, half-cue reads of items 0 to 199: `burst` and `deferred`
  read at chance (0.02 to 0.06 of the stored assembly, k/n = 0.03), whether
  cued from the item's final winners or from its burst core; only 9 to 35% of
  an item's final winners lie in its burst set. `online_burst` read 0.36
  to 0.56 at 0.5 to 1 theta. Burst sets ran 1.6 k at 0.17 theta and 0.04 to
  0.6 k at 1.5 theta (refraction suppresses re-firing).
* One frozen round from half of round t's winners: under `deferred`, 0.54
  to 0.63 of round t + 1's winners and 0.03 of round t's, while consecutive
  rounds themselves overlap 0.02 to 0.04; under `round`, consecutive rounds
  overlap 0.72 to 0.87.
* A smoke run of the module (VOID: 3 brains, cap 32, two rates, two cells):
  the pipeline end to end; nothing of it is evidence.

The reading the probe suggests, registered here to be tested on new brains
and cells: an online write converges the item's rounds onto one assembly
and stores an ATTRACTOR; a deferred write -- whatever its window -- writes
rounds that refraction has relocated every round, and stores the item's
TRAJECTORY, a chain of near-disjoint patterns that a half cue of the last
round cannot complete.

### Protocol

`research/experiments/memory_write_rules.py` (`python -m research.runner
write_rules`), seeds 302 to 321 (new brains), one run from a worktree pinned
at the commit registering this amendment. The refracted AssemblyMemory
(0.5 beta, T = 8, w_max 20, arm B, ungated) at three of Amendment 18's cells:
(2000, 60, 0.5), (4000, 20, 0.5), (4000, 160, 0.5). Each rule is swept over
theta x 0.1 x 2^(j/4), j = 0 to 20 (0.1 to 3.2 theta), with distinct
completion and capacity as in Amendments 13 to 21; checkpoints from M = 2,
no stop before M = 32; cap 3 C_pred + 256 (Amendment 21's law); a rate that
has never completed stops at the first checkpoint past 256 items. A rule's
capacity is its best over its own grid.

Trajectory reading, per rule at 1.0 theta: 200 items stored on fresh brains;
for items 199, 150, 100 and 50, one frozen masked round from the first half
of round t's winners (t = 0 to 6), scored as the fraction of round t + 1's
winners recovered ("next") and of round t's ("same"), and the overlap of
round t with round t + 1 themselves ("own"); means per brain over t and
items, then over brains.

    python -m research.runner write_rules \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag write-rules-20261003 --seeds 302 ... 321

### Bars

    CV  THE INSTRUMENT. The round write's best capacity at each cell lies
        within [0.75, 1.10] of Amendment 18's (506.7, 1553.6, 1083.2); the
        band is wider below because the quarter-octave grid's best rate can
        sit an eighth of an octave off Amendment 18's optimum. Failure voids
        W1 to W4.
    W1  DEFERRAL ABOLISHES THE MEMORY. Under `deferred`, distinct completion
        never exceeds one half at any rate of the grid, at every cell
        (capacity 0).
    W2  THE BURST WRITE STORES NOTHING. The same for `burst`.
    W3  BURST GATING COSTS CAPACITY. `online_burst`'s best capacity is at
        most half the round write's, at every cell.
    W4  A DEFERRED WRITE STORES THE TRAJECTORY. At every cell, under
        `deferred`: "next" >= 0.4, "next" >= 5 x "same", and "own" <= 0.1;
        and under `round`, "own" >= 0.5.

Reported, not judged: each rule's capacity curve; `burst`'s burst-set size
in units of k; `online_burst`'s and `burst`'s trajectory readings.

### Interpretation, stated now

* W1, W2 and W4 pass: what the assembly memory stores is decided by WHEN it
  writes, not by the window or symmetry of the rule. A write that feeds back
  into the item's own rounds builds an attractor; a write after the item --
  the regime of a rule that integrates over a burst window longer than the
  dynamics -- stores a heteroassociative trajectory. A BTDP-like rule can
  store attractors only if its window is short against the time the
  assembly takes to relocate, or if something other than refraction holds
  the assembly still while it writes.
* W3 passes: the first firings carry much of the write that converges the
  item; gating them out costs capacity even online.
* W1 or W2 fails: a deferred write does store attractors at some rate, and
  the probe's chance readings were a property of its cell or its load.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until CV and W1 to W4 are evaluated and recorded below.

### Amendment 25 result (2026-10-03)

One run from a worktree pinned at 78b983e0, seeds 302 to 321:
[record](../../results/runs/memory.write_rules/write-rules-20261003/results.json),
[log](../../results/logs/memory-write-rules-20261003.log). Capacity at each
rule's own best rate over 0.1 to 3.2 theta; trajectory readings at 1.0 theta
(means over 20 brains):

    cell (n, k, p)     rule           capacity (best rate/theta)   next    same    own
    (2000, 60, 0.5)    round          451 (0.199)                  0.753   0.658   0.796
                       online_burst   0                            0.500   0.528   0.433
                       deferred       0                            0.642   0.028   0.020
                       burst          0                            0.182   0.147   0.026
    (4000, 20, 0.5)    round          1442 (0.200)                 0.999   0.982   0.982
                       online_burst   0                            0.065   0.011   0.010
                       deferred       0                            0.786   0.003   0.002
                       burst          0                            0.060   0.010   0.003
    (4000, 160, 0.5)   round          1062 (0.167)                 0.806   0.723   0.862
                       online_burst   0                            0.611   0.584   0.542
                       deferred       0                            0.765   0.027   0.022
                       burst          0                            0.208   0.135   0.026

    CV  THE INSTRUMENT                                                PASS
        round write 451 / 1442 / 1062 against Amendment 18's 506.7 /
        1553.6 / 1083.2: 0.89, 0.93, 0.98 (band [0.75, 1.10]).
    W1  DEFERRAL ABOLISHES THE MEMORY                                 PASS
        `deferred` capacity 0 at every rate of every cell.
    W2  THE BURST WRITE STORES NOTHING                                PASS
        `burst` capacity 0 at every rate of every cell.
    W3  BURST GATING COSTS CAPACITY                                   PASS
        `online_burst` capacity 0 at every cell (bar: <= half the round
        write's) -- gating out the first firings does not cost capacity, it
        abolishes it.
    W4  A DEFERRED WRITE STORES THE TRAJECTORY                        PASS
        `deferred`: next 0.64 to 0.79 (>= 0.4), 23 to 262 times same (>= 5),
        own 0.002 to 0.022 (<= 0.1); `round` own 0.80 to 0.98 (>= 0.5).

**Reading.** What the assembly memory stores is decided by WHEN it writes.
The round write feeds each round's counts into the next round, the item's
rounds converge onto one assembly (consecutive rounds overlap 0.80 to 0.98),
and the memory holds attractors -- 451 to 1442 items. The same counts
written after the item store no attractor at all, at any of 21 rates over
a 32-fold range, at any load from two items: with nothing written during
the item, refraction relocates the assembly every round (consecutive rounds
overlap at chance, 0.002 to 0.022), and what the deferred write stores is
the item's TRAJECTORY -- one frozen round from half of round t's winners
recovers 64 to 79% of round t + 1's. The burst write, the BTDP rule
(deferred, symmetric, once per item between neurons that fired twice),
stores neither: its symmetric counts close the trajectory's chain into one
union, from which a half cue of any one round recovers little (0.06 to
0.21 of the next round, 0.01 to 0.15 of its own). And gating the online
write on bursts abolishes the memory too: the write that converges an item
is carried by neurons' FIRST firings, which refraction makes the common case.

For burst-timing-dependent plasticity this is a constraint, not a verdict:
a burst rule integrates over a window longer than the dynamics it shapes,
which here are relocated by refraction every round. It can store
attractors only if its window is short against the relocation period
(1 round under the write's own refraction), or if something other than
plasticity -- the round write's own feedback, here -- holds the assembly
still while it writes. Developing retinogeniculate synapses, where BTDP was
found, refine maps from slow retinal waves rather than store items; the
memory reading is that such a rule shapes structure and a fast, online
rule stores items.

**Seen in the record, not registered.** `online_burst` holds the assembly
together in proportion to k (own 0.01 at k = 20, 0.43 at k = 60, 0.54 at
k = 160): with more winners a round, more neurons fire twice before
refraction moves them, so more of the item is written. `burst`'s burst
set at 1.0 theta is 0.76 k, 0.13 k and 0.86 k -- smaller than an
assembly, the refraction cost of asking a neuron to fire twice -- and its
next-round reading (0.06 to 0.21) is barely above its same-round one.

## Amendment 26 (2026-10-06, before running): sequences from adaptation -- replay, the switch, and sequence capacity

Registered before any run of this amendment. Amendment 25 found that the
round write's own counts, written after the item, store no attractor but the
item's trajectory: refraction relocates the unwritten activity every round,
and one frozen round from half of round t recovers 64 to 79% of round t + 1.
A one-step read is not yet a sequence memory. This amendment asks three
things of the same circuit:

* **Replay.** From half of an item's FIRST state, does the area run through
  the item's states in order, by itself -- frozen, masked rounds, each fed
  the previous round's winners, nothing from the stimulus?
* **The switch.** With the write kept online, does the refraction-to-
  plasticity ratio s / beta decide what is stored? Below the churn
  transition (s ~ 0.8 beta, REFRACTION-CANCELS-CONVERGENCE) the online write
  converges an item onto one assembly; above it the activity moves every
  round as it is written, which by Amendment 25's reading should store the
  trajectory.
* **Sequence capacity, and its law.** How many stored sequences does the
  area replay, and does that number follow n / k (a Willshaw-type
  heteroassociative count) or the in-degree d = n p (Amendment 21's law for
  the attractor memory)?

**Seen before registering** (5 brains, seeds 900 to 904, (4000, 60, 0.5),
T = 8 unless stated; disclosed in full):

* Replay length after 200 items (fraction of the 7 steps): deferred write at
  1.0 theta: 1.00 (overlap rising along the chain, 0.79 to 0.97); online
  write at s = 1.5 beta and 3.0 beta, 1.0 theta: 0.96 and 0.93, during-write
  consecutive overlap 0.02 and 0.00; at 0.5 theta every sequence arm
  replayed at most one step. Online write at s = 0.5 and 0.8 beta: own
  0.77 to 0.93 (the item holds still; replay trivially follows it).
* At T = 16 (1.0 theta): deferred, s = 0.5 beta: 13.55 of 15 steps;
  deferred, s = 1.5 beta: 4.0; online, s = 1.5 beta: 3.45; online at s =
  0.5 to 0.8 beta: the item dwells, relocates near the clip period, and the
  masked replay follows the dwell but never the hop.
* Sequence capacity of the deferred write (replay over 16 items spread over
  the stored ones): full replay to M = 400 and below 0.3 by M = 800 at 1.0,
  1.4 and 2.0 theta; at 0.7 theta never above 0.64. The attractor control
  at 0.2 theta read 0.75 at M = 1600 and 0.40 at 3200.
* A smoke run of the module (VOID: 3 brains, one rate per arm, (2000, 60,
  0.5), to 64 items): the pipeline end to end.

### Protocol

`research/experiments/memory_sequences.py` (`python -m research.runner
sequences`), seeds 322 to 341 (new brains), one run from a worktree pinned
at the commit registering this amendment. The AssemblyMemory (T = 8, w_max
20, arm B, ungated, masked readout) at three cells chosen so that two laws
make opposite predictions: (2000, 60, 0.5) and (4000, 120, 0.5) share
n / k = 33; (4000, 60, 0.5) and (4000, 120, 0.5) share d = 2000. Three arms:

    deferred-s0.5   write_rule "deferred", s = 0.5 beta, rates 0.5, 0.7, 1.0,
                    1.4, 2.0, 2.8 theta
    round-s1.5      write_rule "round",    s = 1.5 beta, the same rates
    round-s0.5      write_rule "round",    s = 0.5 beta, 0.2 theta (control)

Each rate stores items and is read at checkpoints 32 x 2^(j/4) items (to
16384). At each checkpoint 16 items spread evenly over those stored are
replayed: half of the item's round-0 winners, then seven frozen masked
rounds; an item's replay length is the number of steps before its overlap
with its own round j first falls below 0.5, over 7. The control arm reads
8-round half-cue completion of the item's last round instead. A rate stops
two checkpoints after its mean read has fallen to 0.25 or below, having
been above 0.5; or past 1024 items if it never rose above 0.5. Capacity is
the load at which the mean read falls through 0.5, log-interpolated (0 if
it never rose). During-write consecutive overlap ("own") is the mean
overlap of round t with t + 1 over items 16 to 31 and t = 0 to 6, at each
rate (every rate stores at least 32 items). An arm's PEAK READ is its
largest mean read over every checkpoint of every rate: the mechanism,
whatever the cell's capacity.

    python -m research.runner sequences \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag sequences-20261006 --seeds 322 ... 341

### Bars

    S1  THE DEFERRED WRITE IS A SEQUENCE MEMORY. At every cell, the deferred
        arm's peak read (mean replay length) is at least 0.9, and its own
        is at most 0.1 at every rate.
    S2  ADAPTATION SWITCHES THE MEMORY TYPE. At every cell, the online arm
        at s = 1.5 beta has own at most 0.1 at every rate and a peak read of
        at least 0.7; the control (s = 0.5 beta) has own at least 0.5.
    S3  SEQUENCE CAPACITY FOLLOWS n / k. The deferred arm's best capacity at
        (2000, 60) is within 25% of its value at (4000, 120), and at
        (4000, 60) at least twice it.
    S3d SEQUENCE CAPACITY FOLLOWS THE IN-DEGREE. The deferred arm's best
        capacity at (4000, 60) is within 25% of its value at (4000, 120),
        and at (2000, 60) at most half of it.

S3 and S3d cannot both pass; both may fail. Reported, not judged: each
arm's capacity curve over rates; the online arm's capacity against the
deferred arm's; sequence capacity in transitions (7 per item) against the
attractor control's in items.

### Interpretation, stated now

* S1 and S2 pass: one recurrent area with one causal Hebbian rule is an
  attractor memory or a sequence memory according to whether its
  plasticity, while it writes, outpaces the adaptation that moves its
  activity -- s / beta below or above the churn transition, or the write
  online or deferred. Sequences then need no dedicated connectivity,
  asymmetric rule, or delay line: adaptation supplies the motion and the
  ordinary write records it.
* S3 passes: a sequence is stored as heteroassociative pairs and its
  capacity is Willshaw's, set by the code's sparsity n / k. S3d passes:
  sequences share the attractor memory's in-degree law, and what limits
  both is how many synapses a neuron receives.
* S1 fails: the deferred write's trajectory reading does not extend to
  autonomous replay at these cells.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until S1 to S3d are evaluated and recorded below.

### Amendment 26 result (2026-10-06)

One run from a worktree pinned at 9365ca84, seeds 322 to 341:
[record](../../results/runs/memory.sequences/sequences-20261006/results.json),
[log](../../results/logs/memory-sequences-20261006.log). Capacity in
sequences of eight states at each arm's best rate (rate in theta); peak read;
during-write consecutive overlap ("own") over the arm's rates:

    cell (n, k, p)     n/k   d      deferred-s0.5          round-s1.5             round-s0.5 (control)
    (2000, 60, 0.5)    33    1000   207 (1.0), own <=0.04  250 (2.8*), own <=0.05  own 0.84
    (4000, 60, 0.5)    67    2000   719 (1.4), own <=0.03  950 (2.8*), own <=0.04  own 0.71
    (4000, 120, 0.5)   33    2000   427 (0.7), own <=0.04  402 (1.4),  own <=0.05  own 0.85
    (* the grid's strongest rate: the optimum may lie above it)

    S1  THE DEFERRED WRITE IS A SEQUENCE MEMORY                       PASS
        peak replay 1.00 at every cell; own 0.000 to 0.042 at every rate.
    S2  ADAPTATION SWITCHES THE MEMORY TYPE                           PASS
        online at s = 1.5 beta: own 0.004 to 0.052 at every rate, peak
        replay 1.00 at every cell; control at s = 0.5 beta: own 0.71 to 0.85.
    S3  SEQUENCE CAPACITY FOLLOWS n / k                               FAIL
        (2000, 60) / (4000, 120) = 0.48 (bar: within 25% of 1).
    S3d SEQUENCE CAPACITY FOLLOWS THE IN-DEGREE                       FAIL
        (4000, 60) / (4000, 120) = 1.68 (bar: within 25% of 1); (2000, 60)
        / (4000, 120) = 0.48 (bar: <= 0.5, met).

**Reading.** One recurrent area and one causal Hebbian rule store either
attractors or sequences, and what decides is whether the activity holds
still while it is written. Written online under refraction weaker than the
plasticity (s = 0.5 beta), the item's rounds converge and the area stores
attractors. Written online under refraction stronger than the plasticity
(s = 1.5 beta), or written after the item, the activity moves every round
(consecutive overlap at chance) and the area stores the trajectory: from
half of an item's first state it replays all seven following states, in
order, by itself, with nothing from the stimulus -- 207 to 950 sequences of
eight states, 1450 to 6650 transitions, depending on the cell. Adaptation
supplies the motion; the ordinary write records it. No asymmetric rule,
delay line or dedicated connectivity is involved.

Sequences need a stronger write than attractors: no sequence arm stores
anything at 0.5 theta at any cell, where the attractor memory's onset is
0.17 theta.

Sequence capacity follows neither law. Doubling the in-degree at fixed n/k
multiplies it by 2.06; doubling n/k at fixed in-degree, by 1.68. Both
matter, and the in-degree slightly more: with three cells the exponents are
determined exactly, not fitted -- C ~ d^1.05 (n/k)^0.75 -- and are a
statement about these three cells until a registration tests them on others.

**Owed, and not claimed here.** The replay criterion is overlap with the
item's OWN state at each step (>= 0.5); it does not require that overlap to
beat every other item's state. Refraction keeps stored states nearly
disjoint (Amendment 25: consecutive overlap at chance), so a replay that
fell onto another item's chain would read near zero, but a distinctness
read -- own state against the best other item's -- is the stronger
instrument and is owed before the capacities are compared with the
attractor memory's distinct capacities. The control arm's 1852 to 14612
"capacity" is mean half-cue overlap and counts merged read-outs; it is not
comparable with Amendment 18's distinct completion and is not used.

**Seen in the record, not registered.** The online arm's capacity at the
two k = 60 cells is still rising at 2.8 theta (censored by the grid); at
(4000, 60) it exceeds the deferred arm's (950 against 719), at (4000, 120)
it does not (402 against 427). At (4000, 120) the deferred arm's best rate
is the grid's weakest storing rate (0.5 theta stores nothing), so its
optimum may lie between them. The best rate falls as k rises at fixed n
(1.4 theta at k = 60, 0.7 theta at k = 120).

**Post hoc, exploratory (2026-10-06; 5 brains, seeds 900 to 904, not the
registered brains; deferred write at (4000, 60, 0.5), 1.4 theta).** The owed
distinctness read, made once: replay's overlap with the item's own state
against its best overlap with ANY other stored state (every round of every
item). At M = 200: own 0.95, best other 0.10, own the best in 100% of steps;
at M = 600: 0.53 against 0.11, 95%; at M = 1000, past capacity: 0.18
against 0.41, 32% -- the replay is captured by other items' states, not
merely weakened. Replay is forward only: one frozen round from half of round
t recovers 0.57 to 0.84 of round t + 1 and 0.01 of round t - 1. A
registration on fresh brains would make the distinctness a claim; until
then it is a probe.

## Amendment 27 (2026-10-06, before running): the sequence-length limit

Registered before any run of this amendment. The calculus's sequence
operation now advances when each element is written in one stimulus-and-
recurrence round at a write equal to the convergence threshold theta
(`ORDERED-RECALL-BY-TRANSITIONS`). The ordered-recall registration left one
question blocked on that: how long a sequence an area can recall by itself.
This amendment measures the limit for a single chosen sequence per brain,
with and without refraction during the write.

**Seen before registering** (exploratory, disclosed in full):

* The calculus's own operation on the numpy engine (no refraction while
  writing; seeds 42 to 44): at the strict xfail's cell (k p = 2.5, out of
  regime), recall reaches 7 to 8 steps whatever L is (L = 16, 32), 5 to 7 at
  L = 64 and 4 at L = 128, with overlap decaying along the chain; at
  (2000, 60, 0.5) it is perfect at L = 8 and 16, breaks at L = 32 (4 to 12
  steps) and fails from the first step at L = 64 and 128.
* The hashed substrate, `store_sequence`, beta = theta, 5 brains on seeds
  900 to 904. Without refraction the mean replay fraction is 1.00 up to a
  length and then collapses, failing from the first steps: perfect to
  L = 32 then 0.27 at 45 at (2000, 60, 0.5); to 128 then 0.80 at 181 and 0 at
  256 at (4000, 60, 0.5); to 724 then 0 at 1024 at (8000, 60, 0.5); to 512
  then 0.01 at 724 at (4000, 30, 0.5); to 32 then 0.80 at 45 and 0.01 at 64
  at (4000, 120, 0.5); to 64 then 0.90 at 91 and 0.02 at 128 at
  (4000, 60, 0.25); below 16 at (1000, 60, 0.5). These cliffs sit near
  0.09 p (n/k)^2. With refraction at 0.5 beta: 1.00 at L = 256 and 0.49 at
  512 at (2000, 60, 0.5); 0.81 at 256, 0.56 at 512, 0.13 at 1024 at
  (4000, 60, 0.5) -- with per-brain breaks along the chain rather than a
  collapse; the refracted arm was NOT probed at the n/k = 133 cells.
* A smoke run of the module (VOID: 3 brains, three ladder points at
  (2000, 60, 0.5)): the pipeline end to end.

### Protocol

`research/experiments/memory_sequence_length.py` (`python -m research.runner
sequence_length`), seeds 342 to 361 (new brains), one run from a worktree
pinned at the commit registering this amendment. Six cells: two pairs of
equal n/k (33: (2000, 60, 0.5) and (4000, 120, 0.5); 133: (8000, 60, 0.5) and
(4000, 30, 0.5)) whose in-degrees differ twofold, and one change of p at fixed
n/k ((4000, 60) at p = 0.5 and 0.25). At each cell, beta = theta, T = 1 round
per element, w_max 20, norm_init, two arms: Hebbian (no refraction) and
refracted (s = 0.5 beta, masked readout). For L on the ladder 16 x 2^(j/4)
(to 8192), each brain stores ONE chosen sequence of L elements
(`store_sequence`) and replays it from half of element 0's winners by frozen
masked rounds; its replay fraction is the elements matched in order (own
overlap >= 0.3) before the first miss, over L - 1. An arm stops after two
consecutive ladder points with mean fraction below 0.2, or at the first such
point if it never rose above 0.5. Its LENGTH LIMIT is where the mean fraction
falls through 0.5, log-interpolated.

    python -m research.runner sequence_length \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag sequence-length-20261006 --seeds 342 ... 361

### Bars

    L1  THE HEBBIAN LIMIT IS WILLSHAW'S. L_H / (p (n/k)^2) lies in
        [0.06, 0.12] at every cell.
    L2  n/k, NOT THE IN-DEGREE. Within each equal-n/k pair the Hebbian limits
        agree within 30%.
    L3  THE HEBBIAN FAILURE IS A CLIFF. At every cell the mean fraction is at
        least 0.9 at the largest ladder point at or below L_H / sqrt 2, and at
        most 0.2 at the smallest at or above L_H x sqrt 2.
    L4  REFRACTION EXTENDS THE LIMIT. The refracted limit is at least twice
        the Hebbian at every cell.

Reported, not judged: each arm's whole curve; the refracted limit against
p (n/k)^2 and against n; the spread of per-brain fractions at the refracted
limit (breaks along the chain against collapse).

### Interpretation, stated now

* L1 to L3 pass: without refraction, one area holds a single autonomous
  sequence of about 0.09 p (n/k)^2 elements and loses it all at once past
  that length -- the Willshaw count for heteroassociative pairs, reached by
  a collapse of the whole chain (merging) rather than by errors that
  accumulate along it. A limit "that varies with the parameters" (the
  sequences paper's 20 to 40) is then this one, at small n/k.
* L4 passes: refraction, the anti-merging force, moves the limit and
  changes its kind -- from a collapse to breaks along the chain.
* L4 fails at the n/k = 133 cells: the refracted limit does not keep pace
  with the Hebbian one where the code is sparse (the probe saw it barely move
  from n = 2000 to 4000); refraction then trades length for something else,
  which the curves will say.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until L1 to L4 are evaluated and recorded below.

### Amendment 27 result (2026-10-06)

One run from a worktree pinned at 75fe2ac6, seeds 342 to 361:
[record](../../results/runs/memory.sequence-length/sequence-length-20261006/results.json),
[log](../../results/logs/memory-sequence-length-20261006.log). Length limits
(elements) and the Hebbian constant L_H / (p (n/k)^2):

    cell (n, k, p)     n/k   d      Hebbian L_H   constant   refracted L_R   L_R / L_H
    (2000, 60, 0.5)    33    1000   38            0.068      520             13.8
    (4000, 120, 0.5)   33    2000   49            0.087      1488            30.6
    (4000, 60, 0.5)    67    2000   193           0.087      418             2.17
    (4000, 60, 0.25)   67    1000   97            0.087      395             4.08
    (8000, 60, 0.5)    133   4000   939           0.106      305             0.33
    (4000, 30, 0.5)    133   2000   562           0.063      322             0.57

    L1  THE HEBBIAN LIMIT IS WILLSHAW'S                               PASS
        0.063 to 0.106 at every cell (band [0.06, 0.12]).
    L2  n/k, NOT THE IN-DEGREE                                        FAIL
        n/k = 33: 38 against 49 (22%, within 30%); n/k = 133: 939 against
        562 (67%).
    L3  THE HEBBIAN FAILURE IS A CLIFF                                FAIL
        the ensemble curve is not within a factor sqrt 2 of 0.9 and 0.2 at
        three cells ((4000, 120), (4000, 60, 0.5), (8000, 60)).
    L4  REFRACTION EXTENDS THE LIMIT                                  FAIL
        2.2 to 31 times at the four cells with n/k <= 67; 0.33 and 0.57 at
        the two with n/k = 133.

**Reading.** Without refraction, one area holds a single autonomous
sequence of 0.063 to 0.106 p (n/k)^2 elements: the Willshaw count for
heteroassociative pairs, with p entering exactly (193 at p = 0.5, 97 at
0.25). n/k carries it at the dense pair but not at the sparse one, where
doubling the in-degree at fixed n/k multiplied the limit by 1.67 (L2 fails).
Refraction lengthens the limit 2 to 31 times where the code is dense
(n/k <= 67) and SHORTENS it where it is sparse (n/k = 133): at
(8000, 60, 0.5) the refracted area replays 305 elements against the
Hebbian 939. Refraction removes the merging cliff and introduces a limit of
its own.

**Post hoc, from the record (not registered).** Two structures explain the
failures.

1. *The Hebbian collapse is all-or-none in every brain.* Only 0.0 to 5.6% of
   per-brain replay fractions lie strictly between 0.1 and 0.9: a brain
   replays its whole sequence or fails from the first steps. Brains collapse
   at different lengths (per-brain cliffs span 1.3 to 2.3-fold, e.g. 861 to
   1218 at (8000, 60)), which spreads the ENSEMBLE curve over more than the
   sqrt 2 window L3 allowed. L3 read the ensemble; the cliff it was written
   for is per brain.

2. *The refracted chain breaks at a deadline: step n/k.* At the four cells
   with n/k >= 67 the median break step (elements replayed before the first
   miss, on brains that break) is 0.96 to 1.02 n/k on every ladder point
   inspected -- 130 and 132 at n/k = 133, 64 to 68 at n/k = 67 -- with a
   spread of a few steps across twenty brains. At the two n/k = 33 cells the
   first breaks also sit near n/k (minimum 33 to 37) and later ones near
   2 to 3 n/k. Step n/k is when a sequence of k-neuron states has used about
   n neurons: every neuron has then fired about once and carries a
   refraction bias. A reading to test, not a finding: before that step the
   write picks each next state from neurons with no bias, by the previous
   state's drive; after it, every candidate carries a bias and the choice is
   made partly by bias, so the transition written is weak and the replay
   breaks there. It is the same kind of weight-arithmetic deadline as the
   refraction period law (REFRACTION-CANCELS-CONVERGENCE), set here by the
   code's sparsity.

A failed bar is recorded as failed and not moved.

## Amendment 28 (2026-10-06, before running): the tiling deadline

Registered before any run of this amendment. Amendment 27 found that a
refracted area's single sequence breaks, on replay, at step n/k on nearly
every brain at the cells with n/k >= 67 (post hoc, 0.96 to 1.02 n/k). The
reading offered there: refraction charges every winner a bias that never
decays, so each new element is drawn from neurons that have never fired; the
sequence TILES the area, and at n/k elements the fresh pool is empty.

**Seen before registering** (exploratory, 5 brains, seeds 900 to 904,
(4000, 60, 0.5), beta = theta, L = 200, disclosed in full): with refraction
at 0.5 beta the share of each element's winners that had never fired was
1.00 at elements 10 and 40, 0.98 at 60, 0.58 at 66, 0.03 at 70 and 0.00 from
80 on; replay broke at 66 and 132 on two brains and ran all 199 steps on
three. With the bias zeroed every 30 elements the fresh share fell
gradually (0.55 at 40, 0.31 at 60) and all five brains replayed all 199
steps. The one-step transition read (one frozen round from the true state)
showed no dip at n/k (0.65 at 60, 0.72 at 66, 0.80 at 70): the break is in
the chained replay, not in a single weak transition. A smoke run of the
module (VOID: 3 brains, L = 24) ran the pipeline.

### Protocol

`research/experiments/memory_tiling.py` (`python -m research.runner
tiling`), seeds 362 to 381 (new brains), one run from a worktree pinned at
the commit registering this amendment. Four cells: (2000, 60, 0.5),
(4000, 60, 0.5), (8000, 60, 0.5), (4000, 30, 0.5) (n/k = 33, 67, 133, 133).
At each, beta = theta, refraction 0.5 beta, one chosen sequence per brain of
L = 4 n/k elements written one round per element (`store_sequence`), in two
arms: refracted, and reset (the bias zeroed before every element whose
index is a positive multiple of floor(n / 2k)). Per element, the share of
its winners that had never fired before it (`return_fresh`); per brain, the
replay's first-break step (half of element 0, frozen masked rounds, own
overlap >= 0.3; L - 1 if it never breaks).

    python -m research.runner tiling \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag tiling-20261006 --seeds 362 ... 381

### Bars

    D1  THE SEQUENCE TILES THE AREA. Refracted arm, every cell: the mean
        fresh share is at least 0.95 at every element up to 0.9 n/k and at
        most 0.10 at every element from 1.1 to 1.5 n/k.
    D2  REPLAY BREAKS ON THE DEADLINE. Refracted arm, pooled over cells: of
        the brains whose replay breaks before the end (at least 10 such
        brains, else the bar is void), at least 80% break within
        max(0.05 n/k, 2) steps of a positive multiple of n/k.
    D3  A RECOVERING BIAS REMOVES THE DEADLINE. Reset arm: mean replay
        fraction (first-break step over L - 1) at least 0.9 at every cell,
        and at least 0.3 above the refracted arm's at the two n/k = 133
        cells.

Reported, not judged: the whole fresh-share curves; the per-brain break
steps; the reset arm's fresh share.

### Interpretation, stated now

* D1 to D3 pass: the refracted area's sequence memory is limited by a
  deadline set by the code's sparsity alone -- a single sequence exhausts the
  area's fresh neurons at n/k elements and replay breaks at the wraps --
  and a refraction that recovers removes it. The cumulative, never-decaying
  bias of the reference RefractedArea is then the limit, and adaptation
  with a finite time constant the remedy.
* D1 passes, D2 fails: the area tiles, but replay breaks are not tied to the
  wraps; the deadline in Amendment 27 was a coincidence of these cells.
* D3 fails at (2000, 60, 0.5): its sequence (133 elements) is more than three
  times its Hebbian limit (Amendment 27: 38), and a reset that lets neurons
  be reused gradually may meet the Hebbian merging limit instead. Stated now
  as the risk.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until D1 to D3 are evaluated and recorded below.

### Amendment 28 result (2026-10-06)

One run from a worktree pinned at 2d86e0d6, seeds 362 to 381:
[record](../../results/runs/memory.tiling/tiling-20261006/results.json),
[log](../../results/logs/memory-tiling-20261006.log). L = 4 n/k elements:

    cell (n, k, p)     n/k    fresh share at 0.9 / 1.0 / 1.1 n/k   refracted breaks (brains)              refracted   reset
    (2000, 60, 0.5)    33.3   0.95 / 0.48 / 0.03                   none in 132 steps (20 of 20 complete)  1.00        1.00
    (4000, 60, 0.5)    66.7   0.98 / 0.57 / 0.00                   65-68 (5), 132-134 (3), 199-203 (3)    0.70        1.00
    (8000, 60, 0.5)    133.3  0.99 / 0.48 / 0.00                   128-142 (19), 264 (1)                  0.26        1.00
    (4000, 30, 0.5)    133.3  0.99 / 0.47 / 0.00                   125-138 (11), 261-267 (8), 398 (1)     0.37        1.00

(last two columns: mean replay fraction, first-break step over L - 1)

    D1  THE SEQUENCE TILES THE AREA                                    PASS
        fresh share >= 0.95 up to 0.9 n/k and <= 0.10 from 1.1 to 1.5 n/k
        at every cell.
    D2  REPLAY BREAKS ON THE DEADLINE                                  PASS
        49 of 51 breaking brains within max(0.05 n/k, 2) steps of a
        multiple of n/k (96%; bar 80%).
    D3  A RECOVERING BIAS REMOVES THE DEADLINE                         PASS
        reset arm 1.00 at every cell -- every brain replays all 4 n/k
        elements, 532 at the sparse cells -- against 0.26 and 0.37
        refracted at n/k = 133.

**Reading.** A refracted area writes a single sequence onto neurons that
have never fired until none are left: the sequence tiles the area exactly,
element n/k is half fresh, and from there on every new state reuses
neurons. Replay breaks at the wraps -- at one, two or three times n/k, within
two steps -- and almost never elsewhere. The deadline is set by the code's
sparsity alone: the same 133 at (8000, 60) and (4000, 30), whose in-degrees
differ twofold. A bias that recovers -- zeroed every n/(2k) elements --
removes the deadline at every cell; the refracted area then replays
sequences of 532 elements whole, where refraction alone breaks at 130.
The never-decaying bias of the reference RefractedArea is the limit of its
sequence memory, and a finite adaptation time constant lifts it. The
densest code (n/k = 33) survives its wraps on every brain.

This resolves Amendment 27's L4 failure: the refracted limit fell with n at
fixed k because the deadline n/k grows more slowly than the Hebbian
limit's p (n/k)^2, and at n/k = 133 the first wrap comes before the
Hebbian cliff.

## Amendment 29 (2026-10-07, before running): a refraction that recovers -- the recovery time sets the sequence-length limit

Registered before any run of this amendment. Amendments 27 and 28 measured
the two ends of one knob. With no refraction the single-sequence limit is a
merging cliff near 0.09 p (n/k)^2; with the reference's cumulative bias,
which never decays, the sequence tiles the area and replay breaks at n/k.
A bias that decays by exp(-1/tau) every writing round
(`AssemblyMemory(bias_decay=...)`; the area's `bias_decay`) spans both:
decay 0 is the unrefracted area bit for bit, decay 1 the cumulative bias bit
for bit (tested). The exact replay gate after the engine change reproduces
onset-20261002-r3 at (2000, 60, 0.5) identically.

**Seen before registering** (exploratory, 5 brains on seeds 900 to 904,
beta = theta, s = 0.5 beta, one chosen sequence per brain, one round per
element; disclosed in full). Mean replay fraction at L = 4 n/k and 8 n/k:

    tau     (4000, 60, 0.5)   (8000, 60, 0.5)
    1       0.21 / 0.00       1.00 / 0.20
    4       1.00 / 0.00       1.00 / 1.00
    16-133  1.00 / 1.00       1.00 / 1.00
    266     1.00 / 0.82       0.85 / 1.00
    inf     0.70 / 0.33       0.25 / 0.12

and, at tau = 16, 33, 67: 1.00 at L = 1067 and 0.00 to 0.42 at 2133 at
(4000, 60, 0.5); at tau = 33 and 67, 1.00 at L = 4267 and 0.00 at 8533 at
(8000, 60, 0.5). A smoke run of the module (VOID: 3 brains, three ladder
points, three taus at (2000, 60, 0.5)) ran the pipeline.

### Protocol

`research/experiments/memory_recovery.py` (`python -m research.runner
recovery`), seeds 382 to 401 (new brains), one run from a worktree pinned
at the commit registering this amendment. Amendment 28's four cells:
(2000, 60, 0.5), (4000, 60, 0.5), (8000, 60, 0.5), (4000, 30, 0.5)
(n/k = 33, 67, 133, 133). beta = theta, s = 0.5 beta, w_max 20, norm_init,
one chosen sequence per brain written one round per element, replay from
half of element 0 by frozen masked rounds, replay fraction = elements
matched in order (own overlap >= 0.3) before the first miss over L - 1.
Recovery times tau = 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 rounds and
infinity (no decay). For each, L on the ladder 16 x 2^(j/2) to 16384, stopping
after two consecutive points with mean below 0.2 (or the first, if it never
rose above 0.5); the LENGTH LIMIT is where the mean falls through 0.5,
log-interpolated.

    python -m research.runner recovery \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag recovery-20261007 --seeds 382 ... 401

### Bars

    B1  RECOVERY BEATS BOTH ENDS. At every cell the limit at the best tau is
        at least twice the larger of the Hebbian (tau = 0) and cumulative
        (tau = infinity) limits.
    B2  THE OPTIMUM IS INTERIOR. At every cell the best tau is neither 0 nor
        the largest finite tau of the grid (512).
    B3  THE UPPER EDGE SCALES WITH n/k. The largest tau whose limit is at
        least half the best lies within [0.5, 4] n/k at every cell.
    B4  THE LIMIT AT THE BEST tau GROWS AS (n/k)^2. best / (n/k)^2 varies by
        at most a factor 1.5 across the four cells.

Reported, not judged: every tau's limit and curve; the best limit's constant
in (n/k)^2 and in p (n/k)^2; the lower edge.

### Interpretation, stated now

* B1 to B4 pass: the length of a sequence one area can recall by itself is
  set by how fast its adaptation recovers. Too fast and states merge (the
  Hebbian cliff); too slow and the area is used up (the tiling deadline);
  in between, the limit rises severalfold above both, with an upper edge
  set by n/k, and grows as the square of the code's sparsity -- a capacity
  of the synapses, reached once neither failure mode intervenes.
* B3 fails: the window's upper edge is not set by the tiling scale.
* B4 fails: the best limit follows a different law; the curves say which.
* The risks stated now: (2000, 60, 0.5) was not probed, and its cumulative
  limit (Amendment 27: 520) is already 14 times its Hebbian one, so B1 there
  is the closest; (4000, 30, 0.5) was not probed either.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until B1 to B4 are evaluated and recorded below.

### Amendment 29 result (2026-10-07)

One run from a worktree pinned at 9b55949a, seeds 382 to 401:
[record](../../results/runs/memory.recovery/recovery-20261007/results.json),
[log](../../results/logs/memory-recovery-20261007.log). Length limits
(elements) by recovery time tau (writing rounds):

    tau           0     1     2     4     8     16    32    64    128   256   512   inf
    (2000,60)     43    69    107   153   305   553   673   616   473   558   589   529
    (4000,60)     208   233   305   431   747   1219  1738  2309  1685  1184  560   506
    (8000,60)     861   1132  1218  1579  1871  3444  4872  6890  6838  3825  576   264
    (4000,30)     591   621   751   861   1218  1722  2422  2462  2325  1380  331   302

    B1  RECOVERY BEATS BOTH ENDS                                       FAIL
        best / max(Hebbian, cumulative): 1.27 at (2000, 60); 4.6, 8.0 and
        4.2 at the other three (bar >= 2 at every cell).
    B2  THE OPTIMUM IS INTERIOR                                        PASS
        best tau 32 at (2000, 60), 64 at the other three.
    B3  THE UPPER EDGE SCALES WITH n/k                                 FAIL
        edge 15 n/k at (2000, 60) (the limit never falls below half its
        best); 3.8, 1.9 and 1.9 n/k at the other three (band [0.5, 4]).
    B4  THE LIMIT AT THE BEST tau GROWS AS (n/k)^2                     FAIL
        best / (n/k)^2 = 0.61, 0.52, 0.39, 0.14 (spread 4.4, bar 1.5).

**Reading.** A refraction that recovers lengthens the sequence one area can
recall by itself severalfold over both ends, with the best recovery time
interior at every cell: an 8000-neuron area replays a single chosen sequence
of about 6900 elements whole from half of its first, where the unrefracted
area holds 861 and the never-recovering bias 264. The limit rises from the
Hebbian end as recovery slows, peaks at tau = 32 to 64 writing rounds, and
falls back toward the tiling deadline once recovery is slower than a few
n/k. The densest code (n/k = 33) is the exception the bars record: its
never-recovering bias already survives its wraps (Amendment 28), so recovery
there buys only 1.27 times, and its limit plateaus instead of falling (B1,
B3 fail there and only there).

B4 fails because the best limit does not follow (n/k)^2. Post hoc, it follows
the in-degree d = n p: the two cells with d = 2000 hold 2309 and 2462 (n/k 67
and 133), the two with n/k = 133 hold 2462 and 6890 (d 2000 and 4000).
Against Amendment 21's in-degree law for the attractor memory,
0.0135 d^1.51, the best limits are 1.47, 1.77, 1.86 and 1.89 times it: at its
best recovery time the area holds sequences on the attractor memory's
synapse budget, at about one and a half to two times its constant. A
registration on new cells would make that a claim. The best tau does not
scale with n/k (32 to 64 over a fourfold range of n/k); the upper edge
(256 at the three sparse cells) does not either, beyond the bars' band.

## Amendment 30 (2026-10-07, before running): context -- the same element in two sequences

Registered before any run of this amendment. Two chosen sequences per brain
share a run of m identical elements, S1 = A0 A1 A2 C0..C(m-1) D0 D1 D2 and
S2 = B0 B1 B2 C0..C(m-1) E0 E1 E2, written one after the other with G
unrelated elements between them. Replay from half of A0 (B0) can end on D (E)
only if the shared elements' states differ between the sequences.

**Seen before registering** (exploratory, 5 brains on seeds 900 to 904,
(4000, 60, 0.5), beta = theta, s = 0.5 beta; disclosed in full): at G = 0,
with refraction recovering over 33 rounds or never, the shared states
overlapped 0.00 to 0.01 across the two sequences for m = 1 to 16, and the
branch read 0.94 to 0.98 on the own continuation against 0.00 to 0.01 on
the other; without refraction the shared overlap rose 0.17, 0.27, 0.51, 0.76,
0.87 for m = 1, 2, 4, 8, 16 and the branch was ambiguous from m = 4. With
m = 8 and a gap of 100, 300, 600 elements, recovery over 33 rounds gave a
shared overlap of 0.34, 0.29, 0.38 (branch own 0.85 to 0.95 against other
0.28 to 0.70); recovery over 8 rounds 0.53 to 0.69 with the first sequence's
branch on the wrong ending; the never-recovering bias 0.13, 0.15, 0.42. A
smoke run of the module (VOID: 3 brains, m = 4, G = 0 and 16) agreed.

### Protocol

`research/experiments/memory_context.py` (`python -m research.runner
context`), seeds 402 to 421 (new brains), one run from a worktree pinned at
the commit registering this amendment. Cells (4000, 60, 0.5) and
(8000, 60, 0.5); beta = theta, s = 0.5 beta, one round per element; three
refractions: tau = 0 (none), tau = 33 rounds, and the cumulative bias (inf);
m = 1, 4, 16; G = 0 and 600. Per brain: the shared states' mean overlap
across the two sequences, and BOTH BRANCHES RIGHT -- each sequence's replay,
from half of its first state through its prefix and shared run, overlaps its
own continuation at least 0.5 and more than the other sequence's.

### Bars

    C1  REFRACTION CODES THE SHARED ELEMENTS APART. With tau = 33 and with the
        cumulative bias, at G = 0, for every m at both cells: mean shared
        overlap <= 0.05, and both branches right on at least 18 of 20 brains.
    C2  WITHOUT REFRACTION THE CONTEXT IS LOST. tau = 0, m = 16, G = 0, both
        cells: mean shared overlap >= 0.5 and both branches right on at most
        10 of 20 brains.
    C3  THE SEPARATION IS LARGELY RECENCY. tau = 33, G = 600, every m, both
        cells: mean shared overlap >= 0.15.

Reported, not judged: every case's overlap and branch count; the cumulative
bias at G = 600.

### Interpretation, stated now

* C1 to C3 pass: the area gives a repeated element a new code in each
  sequence it appears in, because refraction sends it to neurons the first
  sequence has just used and suppressed -- context without a dedicated
  circuit -- and that separation fades once the bias has recovered, so the
  recovery time that maximises sequence length (Amendment 29) also bounds
  how far apart in time two sequences can share elements and still be told
  apart.
* C2 fails: an unrefracted area keeps context too, through the previous
  state alone.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until C1 to C3 are evaluated and recorded below.

## Amendment 31 (2026-10-07, before running): bidirectional recall, steered by long-range inhibition

Registered before any run of this amendment. The round write records forward
transitions only, and replay runs forward only (Amendment 26, post hoc:
backward at chance). `store_sequence(reverse_counts=r)` adds every element's
reverse transition r times after the sequence (a stand-in for a rule that
potentiates post-before-pre pairs from a trace); `HashedArea.prime_lri`
enters states into the long-range-inhibition (LRI) history. A chain linked
both ways points each state at both neighbours; LRI at recall vetoes the
state just left, so the direction is set by where the replay came from.
Both engine options are tested to leave the plain write bit for bit.

**Seen before registering** (exploratory, 5 brains on seeds 900 to 904,
(4000, 60, 0.5), beta = theta, tau = 33, L = 200; disclosed in full): with
forward links only, replay forward 1.00, backward 0.00; one reverse count
written alongside, with no LRI, forward 0.02 and backward 0.01; LRI periods
2, 3, 8 at recall restored forward (1.00) but not backward (0.01) while the
cue was not in the LRI history. A partner area holding only reverse links
read 0.39 to 0.41 one step back with one count (chained backward 0.01) and
0.84 to 0.95 with two or three (chained backward 1.00). One area with extra
forward and reverse counts and LRI period 4 replayed forward 1.00 and
backward 1.00 once the cue was entered into the LRI history; from the middle
it went forward unless primed with the neighbour it came from, and primed,
backward 1.00 and forward 1.00. A smoke run of the module (VOID: 3 brains,
L = 24, r = 0 and 2) read r = 0: forward 1.00, backward 0.00; r = 2: forward
without LRI 0.07, with LRI forward, backward, middle-backward and
middle-forward 1.00.

### Protocol

`research/experiments/memory_bidirectional.py` (`python -m research.runner
bidirectional`), seeds 422 to 441 (new brains), one run from a worktree
pinned at the commit registering this amendment. Cells (4000, 60, 0.5) and
(8000, 60, 0.5); beta = theta, s = 0.5 beta, tau = 33, one chosen sequence of
L = 200 per brain, reverse counts r = 0, 1, 2, 3. Replay fractions (own
overlap >= 0.3 in order before the first miss): forward from element 0 with
no LRI (forward_masked); with LRI period 4, strength 100, the cue primed:
forward from element 0, backward from element L - 1, and from element 100
backward (primed with 101) and forward (primed with 99).

### Bars

    R1  THE FORWARD WRITE IS FORWARD ONLY. r = 0: forward_masked >= 0.9 and
        backward <= 0.1 at both cells.
    R2  ONE REVERSE COUNT IS TOO WEAK. r = 1: backward <= 0.5 at both cells.
    R3  WITH TWO COUNTS AND LRI, BOTH WAYS FROM ANYWHERE. r = 2 and 3: forward,
        backward, middle-backward and middle-forward >= 0.9 at both cells.
    R4  WITHOUT LRI THE TWO-WAY CHAIN HAS NO DIRECTION. r = 2:
        forward_masked <= 0.5 at both cells.

### Interpretation, stated now

* R1 to R4 pass: a sequence stored with reverse links at least twice one
  write's strength can be replayed in either direction from any point, and
  what chooses the direction is long-range inhibition at recall -- the state
  just left cannot win again. Direction is set by recent history, not by
  wiring; LRI's role in this circuit is steering at readout.
* R2 fails: one reverse count suffices once LRI removes the competition.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until R1 to R4 are evaluated and recorded below.

### Amendment 30 result (2026-10-07)

One run from a worktree pinned at 6fc3c85a, seeds 402 to 421:
[record](../../results/runs/memory.context/context-20261007/results.json),
[log](../../results/logs/memory-context-20261007.log). Shared-state overlap
across the two sequences, and brains (of 20) with both branches right:

    case (tau / m / G)    (4000, 60, 0.5)    (8000, 60, 0.5)
    0 / 16 / 0            0.865, 0           0.826, 0
    33 / 1, 4, 16 / 0     <= 0.016, 20       <= 0.004, 20
    inf / 1, 4, 16 / 0    0.000, 20          0.000, 20
    33 / 1 / 600          0.109, 20          0.094, 20
    33 / 4 / 600          0.224, 20          0.205, 20
    33 / 16 / 600         0.668, 0           0.701, 0
    inf / 16 / 600        0.462, 12          0.134, 20

    C1  REFRACTION CODES THE SHARED ELEMENTS APART                     PASS
        overlap <= 0.016 and 20 of 20 brains right, every m, both
        refractions, both cells.
    C2  WITHOUT REFRACTION THE CONTEXT IS LOST                         PASS
        m = 16: overlap 0.865 and 0.826, 0 of 20 right.
    C3  THE SEPARATION IS LARGELY RECENCY                              FAIL
        overlap 0.094 and 0.109 at m = 1 (bar >= 0.15 at every m); 0.21 to
        0.22 at m = 4, 0.67 to 0.70 at m = 16.

**Reading.** Written back to back, a repeated run of elements gets a new code
in the second sequence -- overlap at most 0.016 across m = 1 to 16 -- and every
brain ends each sequence on its own continuation; without refraction the
codes converge (0.83 to 0.87 at m = 16) and no brain does. After 600
unrelated elements, with a refraction that recovers over 33 rounds, the
separation is partly gone, and how much depends on the length of the shared
run: one shared element stays nearly apart (0.09 to 0.11, below C3's bar,
all brains right), four overlap about 0.2 (all right), sixteen converge
(0.67 to 0.70, none right). Context after recovery is carried by the
previous state, which holds for a few shared elements and is lost over
sixteen. The never-recovering bias keeps more separation over the gap
(0.13 at (8000, 60), 0.46 at (4000, 60), where 600 elements tile the area
several times). C3 is recorded as failed.

### Amendment 31 result (2026-10-07)

One run from a worktree pinned at 6fc3c85a, seeds 422 to 441:
[record](../../results/runs/memory.bidirectional/bidirectional-20261007/results.json),
[log](../../results/logs/memory-bidirectional-20261007.log). Mean replay
fractions:

    r    read               (4000, 60, 0.5)   (8000, 60, 0.5)
    0    forward_masked     1.00              1.00
         backward_lri       0.00              0.00
    1    backward_lri       0.01              0.00
    2    forward_masked     0.01              0.01
         forward_lri        0.08              0.25
         backward_lri       1.00              1.00
         middle_backward    1.00              1.00
         middle_forward     0.10              0.29
    3    forward_lri        0.02              0.08
         backward_lri       1.00              1.00
         middle_forward     0.03              0.14

    R1  THE FORWARD WRITE IS FORWARD ONLY                              PASS
    R2  ONE REVERSE COUNT IS TOO WEAK                                  PASS
        backward 0.01 and 0.00 at r = 1.
    R3  WITH TWO COUNTS AND LRI, BOTH WAYS FROM ANYWHERE                FAIL
        backward and middle-backward 1.00 at r = 2, 3; forward and middle-
        forward 0.02 to 0.29.
    R4  WITHOUT LRI THE TWO-WAY CHAIN HAS NO DIRECTION                 PASS
        forward_masked 0.01.

**Reading.** Reverse links written twice or three times give backward recall
from the end and from the middle on every brain (1.00), with LRI vetoing the
state just left. But the forward direction then fails, with LRI and without:
the forward links carry about one write (1.35 counts on average, Amendment
31's probe) against the reverse links' two or three, and the replay follows
the stronger set. The probe that motivated R3 had written EXTRA forward
counts as well (forward +1 or +2 against reverse 2 or 3) and replayed both
ways; the registered design strengthened the reverse links alone, which the
probe had not tested at L = 200 (the smoke's L = 24 passed). The reading the
data support: LRI steers a two-way chain only when both directions are
written comparably strongly; it cannot make a weaker direction win against a
stronger one. R3 is recorded as failed and not moved.

## Amendment 32 (2026-10-07, before running): balanced bidirectional recall

Registered before any run of this amendment. Amendment 31 found that reverse
links written twice or three times give backward recall from anywhere but
that forward recall then fails: the forward links (about 1.35 counts on
average) were the weaker set, and LRI could not make the weaker direction win.
`store_sequence(forward_counts=f)` writes f extra forward transitions after
the sequence (tested: exactly the forward transitions); arms are named
forward + reverse.

**Seen before registering** (exploratory, 5 brains on seeds 900 to 904,
L = 200, tau = 33, disclosed in full). Mean replay fractions (forward with
LRI, backward with LRI, middle-backward, middle-forward; forward without LRI):

    arm (f + r)   (4000, 60, 0.5)                    (8000, 60, 0.5)
    0 + 2         0.08, 1.00, 1.00, 0.09; 0.01       0.34, 1.00, 1.00, 0.22; 0.01
    1 + 2         1.00, 0.24, 0.50, 1.00; 0.02       1.00, 0.87, 1.00, 1.00; 0.02
    2 + 2         1.00, 0.04, 0.09, 1.00; 1.00       1.00, 0.28, 0.77, 1.00; 1.00
    2 + 3         1.00, 1.00, 1.00, 1.00; 0.02       1.00, 1.00, 1.00, 1.00; 0.02

The direction written more strongly wins; only the arm whose two directions
are nearly equal (forward about 3.35 counts, reverse 3) replays both ways.

### Protocol

`research/experiments/memory_bidirectional.py --design balanced` (protocol
version 2), seeds 442 to 461 (new brains), one run from a worktree pinned at
the commit registering this amendment. Amendment 31's cells, sequence and
reads; arms 0 + 2 (Amendment 31's reverse-heavy arm), 1 + 2, 2 + 2 and 2 + 3.

    python -m research.runner bidirectional --design balanced \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag bidirectional-balanced-20261007 --seeds 442 ... 461

### Bars

    Q1  A BALANCED CHAIN GOES BOTH WAYS FROM ANYWHERE. Arm 2 + 3: forward,
        backward, middle-backward and middle-forward (with LRI) >= 0.9 at
        both cells.
    Q2  THE STRONGER DIRECTION WINS. Arm 2 + 2: forward >= 0.9 and backward
        <= 0.5; arm 0 + 2: backward >= 0.9 and forward <= 0.5; both cells.
    Q3  WITHOUT LRI A BALANCED CHAIN HAS NO DIRECTION. Arm 2 + 3: forward
        without LRI <= 0.5 at both cells.

Reported, not judged: arm 1 + 2, between the two.

### Interpretation, stated now

* Q1 to Q3 pass: one area replays a stored sequence in either direction from
  any point when its two directions are written about equally, and long-range
  inhibition at recall chooses which -- the state just left cannot win again.
  The balance is narrow: an imbalance of a fraction of one write's counts
  hands the chain to the stronger direction whatever LRI does.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until Q1 to Q3 are evaluated and recorded below.

### Amendment 32 result (2026-10-07)

One run from a worktree pinned at ed880d7a, seeds 442 to 461:
[record](../../results/runs/memory.bidirectional/bidirectional-balanced-20261007/results.json),
[log](../../results/logs/memory-bidirectional-balanced-20261007.log). Mean
replay fractions (forward with LRI, backward with LRI, middle-backward,
middle-forward; forward without LRI):

    arm (f + r)   (4000, 60, 0.5)                    (8000, 60, 0.5)
    0 + 2         0.09, 1.00, 1.00, 0.10; 0.01       0.23, 1.00, 1.00, 0.34; 0.01
    1 + 2         1.00, 0.27, 0.47, 1.00; 0.02       1.00, 0.79, 0.99, 1.00; 0.02
    2 + 2         1.00, 0.05, 0.11, 1.00; 1.00       1.00, 0.33, 0.73, 1.00; 1.00
    2 + 3         1.00, 1.00, 1.00, 1.00; 0.02       1.00, 1.00, 1.00, 1.00; 0.02

    Q1  A BALANCED CHAIN GOES BOTH WAYS FROM ANYWHERE                  PASS
        arm 2 + 3: all four reads 1.00 at both cells.
    Q2  THE STRONGER DIRECTION WINS                                    PASS
        2 + 2: forward 1.00, backward 0.05 and 0.33; 0 + 2: backward 1.00,
        forward 0.09 and 0.23.
    Q3  WITHOUT LRI A BALANCED CHAIN HAS NO DIRECTION                  PASS
        arm 2 + 3: forward without LRI 0.02 at both cells.

**Reading.** One area replays a stored sequence of 200 elements forward from
its start, backward from its end, and either way from its middle, on every
brain, when its two directions are written about equally (forward about 3.35
counts, reverse 3) and long-range inhibition at recall vetoes the state just
left. Without LRI the same chain goes nowhere (0.02). With the directions
unequal the stronger one wins whatever LRI does -- a third of one write's
counts is enough to tip it (arm 1 + 2, between the two, goes forward on every
brain and backward on some). LRI's role in this circuit is the steering
wheel of a two-way chain: it chooses between directions of equal strength;
it does not make a weaker direction win.

## Amendment 33 (2026-10-07, before running): sequences of sequences across two areas

Registered before any run of this amendment. Within one area the refraction
that keeps contexts apart (Amendment 30) also codes a shared element anew in
every sequence, so two separately stored sequences cannot be joined at a
shared junction (exploratory, 5 brains, (4000, 60, 0.5): replay never crossed
the junction under refraction -- 0.00 -- and crossed it indiscriminately
without, 0.22 to 0.80). This amendment splits the two jobs across two areas.

* S, the SEQUENCE area (k = 60, p = 0.5): each chunk, a chosen sequence of 12
  elements, is written ONCE and reused by every plan.
* C, the CHUNK area (k = 60, p = 0.5): each plan, an order of chunks, is
  written as a chosen sequence of chunk symbols, one round per chunk.
* C -> S, a `DenseOrganFiber` (n_C -> n_S, p = 0.5, beta = theta of S): each
  plan position's C state is linked onto its chunk's first S state, `link`
  times, after the plans.

Both areas: beta = theta, refraction 0.5 beta recovering over 33 rounds.
Recall runs on two clocks: from a plan's first C state, for each chunk, the
C state drives S through C -> S (one frozen masked round), S replays the
chunk, and C advances one frozen masked round.

**Seen before registering** (exploratory, 5 brains on seeds 900 to 904;
disclosed in full). With 8 chunks and 6 plans of 5 (the last two sharing
their second and third chunks): whole-plan replay 1.00 for every plan at
link 1, 2 and 3, and 0.00 at link 0, at (S, C) = (4000, 2000) and
(8000, 4000). Under load at (4000, 2000) (plans of 8 chunks): 32 chunks and
32 plans, whole 0.95, chunk starts 1.00, chunk-order chain 1.00; 64 and 96,
whole 0.30 to 0.34, starts 0.72 to 0.78, chain 0.96; 128 and 256, whole 0.09
to 0.11, starts 0.12, chain 0.44; link 1 and 2 alike. A smoke run of the
module (VOID: 3 brains, chunks of 4, the mechanism load) read link 0: whole
0.00; link 2: whole 1.00.

### Protocol

`research/experiments/memory_hierarchy.py` (`python -m research.runner
hierarchy`), seeds 462 to 481 (new brains), one run from a worktree pinned at
the commit registering this amendment. Two cells, (n_S, n_C) = (4000, 2000)
and (8000, 4000). Loads (chunks, plans): (8, 6) with plans of 5 chunks
(the registered plan set, seed 20261007, the last two sharing a run of two),
and (32, 32) and (64, 96) with plans of 8; links 0 and 2. Per load, 16 plans
evenly spaced over those stored are recalled (plus the shared-run pair at the
light load). Per brain and plan: WHOLE -- elements (chunk starts and
contents, in plan order) matched at own overlap >= 0.3 before the first miss,
over all of them; STARTS -- chunks whose first state the C state evoked; CHAIN
-- the C chain's in-order fraction.

    python -m research.runner hierarchy \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag hierarchy-20261007 --seeds 462 ... 481

### Bars

    H1  SEQUENCES OF SEQUENCES. Load (8, 6), link 2: whole >= 0.9 for every
        plan at both cells.
    H2  THE LINKS CARRY IT. Link 0: chunk starts <= 0.1 for every plan, every
        load, both cells.
    H3  PLANS SHARING A RUN STAY APART. Load (8, 6), link 2: the two plans
        that share a run of two chunks each have chain >= 0.9 and whole >=
        0.9 at both cells.
    H4  THE LINKS GIVE WAY FIRST, AND A LARGER PAIR RELIEVES THEM. Load
        (64, 96), link 2: at (4000, 2000) the mean chunk-start rate is more
        than 0.1 below the mean chain fraction; at (8000, 4000) the mean
        chunk-start rate is at least 0.1 above that at (4000, 2000).

Reported, not judged: load (32, 32); whole-plan replay under load at both
cells.

### Interpretation, stated now

* H1 to H3 pass: two areas do what one cannot -- chunks stored once are
  reused in any order a plan dictates, and the plans themselves stay apart
  where they share chunks, because the chunk area codes each occurrence on
  its own neurons. That is a two-level sequence memory, with the top level
  carrying context and the bottom level carrying reusable content.
* H4 passes: the hierarchy's first limit is the many-to-one links from plan
  positions onto chunk starts (interference among links, not their
  strength), and it scales with the areas' size.
* H4 fails at its second clause: a larger pair does not relieve the links;
  the limit is set by the number of links per chunk start, not by n.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until H1 to H4 are evaluated and recorded below.

### Amendment 33 result (2026-10-07)

One run from a worktree pinned at 0f95af53, seeds 462 to 481:
[record](../../results/runs/memory.hierarchy/hierarchy-20261007/results.json),
[log](../../results/logs/memory-hierarchy-20261007.log). Means over the plans
read (whole-plan replay, chunk starts, chunk-order chain):

    load (chunks, plans), link   (S, C) = (4000, 2000)    (8000, 4000)
    (8, 6), 0                    0.00, 0.00, 1.00          0.00, 0.00, 1.00
    (8, 6), 2                    1.00, 1.00, 1.00          1.00, 1.00, 1.00
    (32, 32), 0                  0.00, 0.00, 1.00          0.00, 0.00, 1.00
    (32, 32), 2                  0.95, 1.00, 1.00          1.00, 1.00, 1.00
    (64, 96), 0                  0.00, 0.00, 0.96          0.00, 0.00, 1.00
    (64, 96), 2                  0.34, 0.78, 0.96          0.94, 1.00, 1.00

    H1  SEQUENCES OF SEQUENCES                                         PASS
        load (8, 6), link 2: every plan 1.00 at both cells.
    H2  THE LINKS CARRY IT                                             PASS
        link 0: chunk starts 0.00 for every plan, every load, both cells.
    H3  PLANS SHARING A RUN STAY APART                                 PASS
        the two plans sharing a run of two chunks: chain and whole 1.00.
    H4  THE LINKS GIVE WAY FIRST, AND A LARGER PAIR RELIEVES THEM      PASS
        (64, 96) at (4000, 2000): starts 0.78 against chain 0.96; at
        (8000, 4000): starts 1.00 (and whole 0.94).

**Reading.** Two areas do what one could not. Chunks of 12 elements, each
written once into the sequence area, are replayed in whatever order a plan in
the chunk area dictates: every plan, every brain, both cells, at the light
load -- 60 elements in plan order from the plan's first state alone, with no
outside guidance. Plans that share chunks stay apart because the chunk area
codes each occurrence on its own neurons, and the content stays shared
because the sequence area stores it once. A new plan over known chunks costs
only its chunk order. Without the links from plan positions to chunk starts
nothing is evoked at any load. Under load the first thing to give way is
those links -- many plan positions pointing at one chunk start begin to evoke
the wrong one (0.78 at 96 plans) while the chunk area still holds every
plan's order (0.96) -- and doubling both areas removes it (1.00). The chunk
starts were linked by construction: the circuit does not yet find a chunk's
boundaries itself.

## Amendment 34 (2026-10-07, before running): robustness -- many sequences, activity noise, corrupted cues

Registered before any run of this amendment. Every earlier sequence study
stored one sequence per brain and replayed it noiselessly. At the recovery
time that maximised length (tau = 33, Amendment 29), this amendment asks
whether many sequences share the single-sequence budget, how much activity
noise during replay the memory tolerates, and how much of a cue may be wrong.

**Seen before registering** (exploratory, 5 brains on seeds 900 to 904;
disclosed in full). (4000, 60, 0.5): M sequences of 32 replayed 1.00 at
M = 1, 8, 32, 0.98 at M = 64 (2048 elements), 0.04 at M = 128 (4096). On one
sequence of L = 200, a fraction nu of every replay step's winners replaced at
random: first-miss median 199 (none) at nu = 0.02 and 0.05, 47 at 0.1
(quartiles 40 to 49), 5 at 0.2; one step from a state with 10% noise read
0.94 of the next state, 30% noise 0.81, and a chained trace at 10% held 0.86
to 0.89 per step -- the losses are rare derailments, not decay. A half cue
with 25% of its neurons wrong replayed 1.00, with 50% wrong 0.02. On
L = 400 at nu = 0.1: tau = 33 first-miss median 42 at (4000, 60) and 110 at
(8000, 60); tau = 8 43 and none; the cumulative bias 43 and 81 (and at
nu = 0.07 it broke at 57 and 114, near 0.86 n/k, where the recovering arms
did not break). The step at which reuse begins (fresh share below 0.9) did
not predict the noisy first miss. A smoke run of the module (VOID: 3 brains,
two ladder points, L = 40) ran the pipeline.

### Protocol

`research/experiments/memory_robustness.py` (`python -m research.runner
robustness`), seeds 482 to 501 (new brains), one run from a worktree pinned
at the commit registering this amendment. Cells (4000, 60, 0.5) and
(8000, 60, 0.5); beta = theta, refraction 0.5 beta recovering over 33 rounds.
MANY: M chosen sequences of 32 elements per brain on the ladder
8 x 2^(j/2), the mean replay fraction over up to 16 of them, stopping after
two points below 0.2; the BUDGET is 32 times the M at which the mean falls
through 0.5. NOISE: one sequence of L = 400; nu = 0, 0.05, 0.07, 0.1, 0.2;
each brain's first-miss step, mean over four noise draws (seeded). CUE:
eta = 0.25 and 0.5 of the half cue replaced; the replay fraction.

    python -m research.runner robustness \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag robustness-20261007 --seeds 482 ... 501

### Bars

    N1  MANY SEQUENCES SHARE ONE BUDGET. At each cell the many-sequence
        budget lies within [0.6, 1.6] of Amendment 29's single-sequence best
        limit (2309 and 6890).
    N2  SMALL NOISE IS HARMLESS. nu = 0.05: at least 18 of 20 brains replay
        all 399 steps at both cells.
    N3  A LARGER AREA DOUBLES THE NOISE HORIZON. nu = 0.1: the median
        first-miss step at (8000, 60) is at least twice that at (4000, 60).
    N4  A CUE MAY BE A QUARTER WRONG. eta = 0.25: mean replay fraction >= 0.9
        at both cells.

Reported, not judged: every nu's first-miss distribution; eta = 0.5; the
whole many-sequence curve.

### Interpretation, stated now

* N1 to N4 pass: the length limit of Amendment 29 is a budget of elements,
  spent the same way by one long sequence or many short ones; replay
  survives a few percent of its winners being wrong at every step and a cue
  a quarter wrong; and a larger area pushes the noisy horizon out, so
  robustness, like length, is bought with neurons.
* N1 fails: many sequences cost more (or less) than their elements.
* N3 fails: the noisy horizon does not scale with the area.
* A failed bar is recorded as failed and not moved after the data.

The run is UNJUDGED until N1 to N4 are evaluated and recorded below.

### Amendment 34 result (2026-10-07)

One run from a worktree pinned at da3d3bbf, seeds 482 to 501:
[record](../../results/runs/memory.robustness/robustness-20261007/results.json),
[log](../../results/logs/memory-robustness-20261007.log).

    MANY (sequences of 32; mean replay)   (4000, 60, 0.5)        (8000, 60, 0.5)
    M = 45 / 64 / 91 / 128 / 181          1.00 / 0.98 / 0.13 /   1.00 / 1.00 / 1.00 /
                                          0.03 / --              1.00 / 0.44
    budget (elements) / A29 limit          2501 / 2309 = 1.08     5578 / 6890 = 0.81

    NOISE (first-miss median of 399)      nu = 0.05: 399, 399; 0.07: 399, 399;
                                          0.1: 39, 232; 0.2: 3.5, 3.3
    CUE (mean replay)                     eta = 0.25: 1.00, 1.00; 0.5: 0.45, 0.05

    N1  MANY SEQUENCES SHARE ONE BUDGET                                PASS
        1.08 and 0.81 of the single-sequence limit (band [0.6, 1.6]).
    N2  SMALL NOISE IS HARMLESS                                        PASS
        nu = 0.05: 20 of 20 brains replay all 399 steps at both cells.
    N3  A LARGER AREA DOUBLES THE NOISE HORIZON                        PASS
        nu = 0.1: median first miss 232 against 39 (5.9 times).
    N4  A CUE MAY BE A QUARTER WRONG                                   PASS
        eta = 0.25: 1.00 at both cells.

**Reading.** The length limit of Amendment 29 is a budget of elements: one
area spends it the same way on one long sequence or on many short ones --
2501 elements as 32-element sequences against a 2309-element single sequence
at (4000, 60); 5578 against 6890 at (8000, 60). Replay survives activity
noise up to about 7% of its winners replaced at every step with no loss at
all; at 10% it holds for a while and then derails, at a horizon a larger
area pushes out almost sixfold; at 20% it derails within a few steps. A cue
a quarter wrong works; a cue half wrong fails. Robustness, like length, is
bought with neurons.

## Amendment 35 (2026-10-07, before running): an engine audit -- an independent oracle, a defect in fibers between areas of different sizes, and Amendment 33 again

Registered before the rerun it names. Every sequence study from Amendment 26
on rests on paths added to `AssemblyMemory` during this program
(`store_sequence`, the recovering bias, `bias_reset`, forward and reverse
counts, `prime_lri`) and on the one-round masked `recall` that replays them.
Their tests were RELATIONS (an option that never fires leaves the write bit
for bit; a decay of 1 is the cumulative bias); none said the write computes
what the registrations say it computes. So the memory was written again.

**The oracle** (`neural_assemblies/tests/test_memory_oracle.py`): float64
numpy, dense matrices, from the stated semantics -- drive, k-WTA with the
lowest index on a tie, the count write, the stimulus potentiation, the bias
decayed at every writing round and charged s raw at the winners, the masked
one-round recall -- with none of the engine's kernels, chain tables or count
storage; only the connectome's hash is shared. It runs in lockstep with the
engine, predicting each round's winners from the engine's previous ones.
Over the cumulative bias, the recovering bias (tau = 33; tau = 8 at two
rounds per element), a reset bias, the Hebbian control, a two-way chain and
a stimulus of 45 at (600, 30, 0.5), and over 200 elements at (4000, 60, 0.5)
-- three tiling wraps, the clip binding -- the engine's winners are the
oracle's at every one of about 2,900 rounds and recalls but one, a near tie
at float32 rounding; the counts are equal and the bias agrees to 1e-4.

**The defect.** Checking the fiber that Amendment 33 adds between its two
areas found one. The norm_init divisor of a generated organ fiber
(`DenseOrganFiber`; `AreaFiber` and `PresentFiber` share it) was the
in-degree counted over rows [0, n_post) where the reference divides by the
fiber's own column sum, the in-degree over its n_pre source rows
(`.reference/mdabagia-nemo/brain.py`, `normalize`). The kernel took one size
for both. Measured: n_pre 2000 -> n_post 4000, the divisor 2.00 times the
true in-degree on average and correlated 0.71 with it neuron by neuron;
4000 -> 2000, 0.50 times, 0.70; a square fiber exact. Both parity gates of
the sequence port ran with norm_init off, so neither could see it. Fixed:
the kernel takes n_pre and n_post. A square fiber is unchanged bit for bit
(the exact replays of onset-20261002-r3 and of robustness-20261007 at
(4000, 60, 0.5) are identical under the fix).

**Scope.** In this program only Amendment 33 has a fiber between areas of
different sizes (C -> S, 2000 -> 4000 and 4000 -> 8000); every memory area's
own fiber is square, so Amendments 24 to 32 and 34 are untouched. Elsewhere:
the A3 transducer's n_arc = 2000 and 50,000 cells (PREREG_seq_a3_transducer.md,
where it is disclosed; its 10,000 cell and its Amendment 2 are square).
The arc-FSM studies ran norm_init off; the high-order and temporal-position
transducers are square.

### Protocol

Amendment 33's protocol unchanged -- `python -m research.runner hierarchy`,
the same cells, loads, links, plans and bars -- on the SAME seeds, 462 to
481, so the same brains and plans and only the C -> S divisor differs; one
run from a worktree pinned at the commit registering this amendment, tag
hierarchy-20261007-a35.

    python -m research.runner hierarchy \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag hierarchy-20261007-a35 --seeds 462 ... 481

### Bars

H1 to H4 exactly as registered in Amendment 33, judged on this run. The
claim SEQUENCES-OF-SEQUENCES-ACROSS-AREAS stands if all four pass and is
restated to what passes if not; the first run stays recorded, as a run on
the defective divisor.

### Prediction, stated now

When a plan position evokes a chunk start, the C -> S fiber is S's only
input, so the uniform part of the error (a factor of two) cannot change a
k-WTA; what remains is a per-neuron error of one to two percent in the
divisor, against learned links up to twenty times the base weight. Expected:
H1 to H4 pass again, the readings within a few hundredths of Amendment 33's;
the reading most likely to move is H4's chunk starts at (64, 96) on
(4000, 2000), 0.78, where links compete.

The rerun is UNJUDGED until H1 to H4 are evaluated and recorded below.

### Amendment 35 result (2026-10-07)

One run from a worktree pinned at dc92d895 (the corrected divisor), seeds 462
to 481 -- the brains and plans of Amendment 33:
[record](../../results/runs/memory.hierarchy/hierarchy-20261007-a35/results.json),
[log](../../results/logs/memory-hierarchy-20261007-a35.log). Means over the
plans read (whole-plan replay, chunk starts, chunk-order chain), Amendment 33
in brackets where it differs:

    load (chunks, plans), link   (S, C) = (4000, 2000)          (8000, 4000)
    (8, 6), 0                    0.00, 0.00, 1.00                0.00, 0.00, 1.00
    (8, 6), 2                    1.00, 1.00, 1.00                1.00, 1.00, 1.00
    (32, 32), 0                  0.00, 0.00, 1.00                0.00, 0.00, 1.00
    (32, 32), 2                  0.964 [0.954], 1.00, 1.00       0.997, 1.00, 1.00
    (64, 96), 0                  0.00, 0.00, 0.958               0.00, 0.00, 0.998
    (64, 96), 2                  0.341 [0.343], 0.782 [0.784],   0.938 [0.944], 0.995 [0.996],
                                 0.958                           0.998

    H1  SEQUENCES OF SEQUENCES                                         PASS
    H2  THE LINKS CARRY IT                                             PASS
    H3  PLANS SHARING A RUN STAY APART                                 PASS
    H4  THE LINKS GIVE WAY FIRST, AND A LARGER PAIR RELIEVES THEM      PASS
        (64, 96) at (4000, 2000): starts 0.782 against chain 0.958; at
        (8000, 4000): starts 0.995.

**Reading.** As predicted: on the corrected divisor every bar passes again
and no reading moves by more than 0.010. The C -> S fiber is the sequence
area's only input when a plan position evokes a chunk start, so the factor of
two could not change a winner, and the per-neuron part of the error was small
against learned links. SEQUENCES-OF-SEQUENCES-ACROSS-AREAS now cites this run.
The oracle stays as a test, and the defect's other site -- the A3 transducer's
2,000 and 50,000 cells -- is rerun under PREREG_seq_a3_transducer.md
Amendment 4.

## Erratum to Amendment 34 (2026-10-08): the noise replaced the strongest winners

Amendment 34 describes its noise as "a fraction nu of every replay step's
winners replaced at random" and its cue as "half of the first state". Both
take the FIRST slots of a winner tensor (`x[:, :m]`), and the k-WTA returns
its winners ordered by drive, strongest first (now pinned by
`test_recall_returns_its_winners_strongest_first`). Measured on 20 probe
brains at (4000, 60, 0.5), tau = 33: the first sixth of a recall's slots are
1.00 members of the true next element, the last sixth 0.35. So the noise
replaced the most driven winners -- nearly all true members -- at every
step, and the cue was the strongest half of the first element by its write
drive. Amendment 34's numbers stand as measurements of THAT instrument; its
words and SEQUENCE-MEMORY-ROBUSTNESS's claim are corrected to say so.

**Found by** the theory round's probes (exploratory, disclosed): a
one-variable Markov reduction of replay, built from the one-step overlap map
with random fillers and assuming random slots, predicted a 10% noise horizon
of 228 steps at (4000, 60) against 37 measured on the same probe brains
(A34: 39). With the slots replaced as A34 replaced them the probe brains read
37.4 and 195 (A34: 39 and 232); with uniformly random slots they read 97 and
never fail in 399 steps. Probe brains: seeds 920 to 939.

## Amendment 36 (2026-10-08, before running): uniformly random noise, and a random half cue

Registered before any run of this amendment. Amendment 34 measured replay
under the deletion of the most driven winners and from the strongest half
cue. This amendment measures what its words described: noise on uniformly
random slots, and a uniformly random half cue -- beside Amendment 34's own
instrument on the same brains.

**Seen before registering** (exploratory, disclosed above and here): on 20
probe brains (seeds 920 to 939), nu = 0.1, top slots 37.4 and 195, uniform
slots 97 and 399 (no failure) at (4000, 60) and (8000, 60). The random half
cue was NOT probed. A smoke run of the module (VOID: 3 brains, L = 40, seeds
900 to 902) ran the pipeline.

### Protocol

`research/experiments/memory_noise.py` (`python -m research.runner noise`),
seeds 502 to 521 (new brains), one run from a worktree pinned at the commit
registering this amendment. Cells (4000, 60, 0.5) and (8000, 60, 0.5); beta =
theta, refraction 0.5 beta recovering over 33 rounds (Amendment 34's). One
sequence of L = 400 per brain. NOISE: nu = 0.05, 0.07, 0.1, 0.2, with the
replaced slots TOP (Amendment 34's) or UNIFORM; each brain's first-miss step,
mean of four seeded draws. CUE: the strongest half or a uniformly random half
of the first element, eta = 0 and 0.25 of it replaced at random; the replay
fraction.

    python -m research.runner noise \
        --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag noise-20261008 --seeds 502 ... 521

### Bars

    U1  UNIFORM NOISE IS MILDER. At (4000, 60), nu = 0.1: the median first
        miss under uniform slots is at least twice that under top slots.
    U2  THE LARGER AREA HOLDS. At (8000, 60), nu = 0.1, uniform slots: at
        least 18 of 20 brains replay all 399 steps.
    U3  AMENDMENT 34 REPRODUCES. Top slots, nu = 0.1: each cell's median
        within 35% of Amendment 34's (39 and 232).
    U4  A RANDOM HALF CUE WORKS. eta = 0, random half: mean replay fraction
        >= 0.9 at both cells. (Not probed: a prediction.)

Reported, not judged: every nu under both slot rules; eta = 0.25 for both
cues.

### Interpretation, stated now

* U1 to U4 pass: Amendment 34 understated the memory's tolerance of random
  activity noise -- at 10% the smaller area's horizon is at least twice what
  it recorded and the larger area does not derail in 400 steps -- and the cue
  need not be the strongest half.
* U4 fails: replay depends on WHICH half of the first element cues it, and
  every sequence result since Amendment 26 is conditional on the strongest
  half.
* A failed bar is recorded as failed and not moved.

The run is UNJUDGED until U1 to U4 are evaluated and recorded below.
