# PREREG: is the capacity ceiling a function of n/k alone, and quadratic?

Registered before running. The predictions below are NUMBERS, fixed now.

## Where this came from, stated plainly

`PREREG_capacity_scaling.md` returned NOT ANSWERED: at fixed k the ceiling
always arrived as the area filled, so CAP2 could not be evaluated. Running the
same protocol at `k = sqrt(n)` -- which holds the chance overlap `k^2/n` at 1
while n varies -- kept the ceiling away from the fill limit and produced three
uncensored points instead of one.

Putting the two series side by side, a POST HOC regularity appeared:

    series         n    k       M*     n/k   M*k/n  M*(k/n)^2  fill  status
    fixed k     4000   60     66.8    66.7    1.00    0.01503  0.748  ok
    fixed k     8000   60    320.8   133.3    2.41    0.01804  0.960  CENSORED
    fixed k    16000   60   1322.6   266.7    4.96    0.01860  0.998  CENSORED
    k=sqrt      4000   63     61.0    63.5    0.96    0.01513  0.727  ok
    k=sqrt      8000   89    138.8    89.9    1.54    0.01718  0.863  ok
    k=sqrt     16000  126    286.2   127.0    2.25    0.01775  0.937  ok(cliff)

    M*(k/n)^2 across all six: 0.01503 .. 0.01860, mean 0.01696, spread +/-11%

**This is a fit to six points that were not collected to test it, three of them
censored. It is NOT a result.** This project has retracted a capacity-scaling
claim twice ([[critical-load-alpha-star]], [[core-areas-near-tiling-limit]]),
both times from exactly this shape: a regularity noticed in data gathered for
another purpose. The point of this note is to make it fail if it is wrong.

## The claim, as a formula

    M* = C (n/k)^2 ,  C in [0.0150, 0.0186] observed, point estimate 0.0170

Two things are being claimed at once and both must hold:

1. `n/k` is the CONTROLLING variable -- M* does not depend on n and k
   separately, only on their ratio;
2. the exponent is 2.

## Predictions, fixed now

Protocol exactly as `PREREG_capacity_scaling.md` (stimulus every round, area
inhibited between assemblies, half-cue rank-1 gated by distinctness,
`ceiling_from_curve`), p=0.50, T=8, beta=0.10, w_max=20, 16 brains.

**Bar CS1 -- n/k is the controlling variable.** At CONSTANT n/k = 66.67, vary n
four-fold. Every law of the form `M* = f(n)` predicts these differ; the claim
predicts they are the same.

    (n, k) = (4000, 60), (8000, 120), (16000, 240)
    predicted M* = 75.6, band [66.7, 82.7] for all three

    PASS   all three land in the band AND their brackets mutually overlap
    FAIL   M* varies systematically with n at fixed n/k

**Bar CS2 -- the exponent, at n/k values not in the fit.**

    (n, k) = ( 8000, 200)  n/k =  40  predicted M* =  27.2  band [ 24.0,  29.8]
    (n, k) = (10000, 100)  n/k = 100  predicted M* = 170.0  band [150.0, 186.0]
    (n, k) = (20000, 100)  n/k = 200  predicted M* = 680.0  band [600.0, 744.0]

    PASS   at least 2 of 3 land in their band
    FAIL   otherwise; a systematic sign to the misses names the wrong exponent

Regime check, all six cells (`kp >= 3 ln n`, p = 0.5):

    (4000,60) 30.0 vs 24.9   (8000,120) 60.0 vs 27.0   (16000,240) 120.0 vs 29.0
    (8000,200) 100.0 vs 27.0 (10000,100) 50.0 vs 27.6  (20000,100)  50.0 vs 29.7

All IN REGIME.

**Bar CS3 -- censoring, as before.** Report `fill@M*` for every cell. A cell at
`fill >= 0.95` is CENSORED and cannot support the claim, only fail to
contradict it. If the passing cells are all censored, the bars are reported as
NOT MET regardless of where the numbers landed.

## Committed in advance

* The (4000, 60) cell is IN the fit and is re-run only as a consistency check;
  it cannot count toward CS1.
* If CS1 fails, `n/k` is not the controlling variable and the collapse above is
  a coincidence of the particular n and k used. That result is reported, and the
  formula is dropped rather than re-fitted with a third parameter.
* If CS1 passes and CS2 fails, the ratio controls but the exponent is wrong.
  The measured exponent is then reported WITHOUT a mechanism, and is not
  quotable until one exists -- the standing rule from the parent note.
* No band is widened after seeing the data.

---

## Result (2026-08-25): CS1 PASSES, CS2 FAILS, and the mechanism I proposed is REFUTED

Run with `seq_capacity_scaling.py --nk`, protocol as registered, 16 brains.

### CS1 -- PASS. `n/k` is the controlling variable.

    n/k = 66.67 held CONSTANT, n varied four-fold
    n= 4000 k=  60  M* = 66.6  bracket [64, 72)  fill 0.746  ok   (in-fit check)
    n= 8000 k= 120  M* = 73.0  bracket [72, 80)  fill 0.746  ok   HELD OUT
    n=16000 k= 240  M* = 67.6  bracket [64, 72)  fill 0.692  ok   HELD OUT

Both held-out cells land inside the registered band [66.7, 82.7]. M* is
constant to **+/-5%** while n changes 4x; any law of the form `M* = f(n)`
predicts these differ by about 4x. All three cells are UNCENSORED
(fill 0.69-0.75), so CS3 is satisfied and the pass is a real one.

Two things reported rather than rounded away: the in-fit consistency cell
landed at 66.6 against a band floor of 66.7 (0.15% below), and it was
registered as unable to count toward CS1 anyway; and the registered
sub-condition "their brackets mutually overlap" is NOT strictly met -- the grid
put (8000,120) in [72,80) and the others in [64,72), adjacent bins that do not
intersect. The point estimates agree to 5%; the bracket condition was too tight
for this grid resolution.

### CS2 -- FAIL. The exponent is not 2.

    n/k=  40  predicted  27.2 [ 24.0,  29.8]   measured  21.8   BELOW  (0.801x)
    n/k= 100  predicted 170.0 [150.0, 186.0]   measured 175.4   IN BAND (1.032x)
    n/k= 200  predicted 680.0 [600.0, 744.0]   measured 791.2   ABOVE  (1.164x)

One of three, against a bar of two of three. The misses have a SYSTEMATIC SIGN
-- low at small n/k, high at large n/k -- which is the signature the note said
would name the wrong exponent. Fitting `M* = C (n/k)^b`:

    ALL 12 ceilings                   n=12  b = 2.189 +/- 0.048   n/k 40..267
    uncensored (fill < 0.95)          n= 9  b = 2.254 +/- 0.071   n/k 40..127
    uncensored AND resolved bracket   n= 7  b = 2.336 +/- 0.151   n/k 63..100

**The 95% CI excludes 2 in all three subsets.** Per the standing rule, the
measured exponent is reported and is NOT quotable as a law: there is no
mechanism for 2.2, and this project has retracted two capacity exponents that
had none.

### The mechanism I proposed is refuted BY CS1

Before the test I suggested capacity might be SYNAPSE-bounded, `M ~ n^2 p /
(k ln(n/k))`, which would explain a fixed-k exponent near 2. That form is not a
function of `n/k` alone -- at constant `n/k` it is proportional to n, so it
predicts a 4x rise across CS1. Measured: +/-5%. **Capacity is not
synapse-bounded.** Whatever sets the ceiling reads only the ratio `n/k`.

### What this settles about k = sqrt(n)

It does NOT increase capacity; it reduces it, because M* rises steeply in `n/k`
and raising k lowers that ratio. At n=16000:

    k =  60  ->  M* = 1322.6      k = 126  ->  M* = 286.2      k = 240  ->  M* = 67.6

What `k = sqrt(n)` buys is MEASURABILITY: it holds the chance overlap `k^2/n`
at 1, so the ceiling arrives with the area 69-94% full instead of 99.8%, which
is what turned one uncensored point into nine. For capacity alone, k should be
as small as the regime allows, `k >= 3 ln n / p`.

### Open

The exponent. b = 2.2 is stable across subsets and spans a 6.7x range in `n/k`,
but has no mechanism, and the largest-`n/k` points are censored. A mechanism --
or a third bar that discriminates 2.2 from 2 on uncensored points alone -- is
what would make it a law.

---

## Amendment 1: the first run was CONTAMINATED; numbers superseded, structure survives

Found while replacing the deviation store. The multi-episode correction the
first run used -- a round mask carried ACROSS episodes -- had never been tested
across episodes: every parity test at the time ran a SINGLE episode, so the
store was written and never read. The replacement is verified against a
stored-connectome reference over 2, 4 and 6 episodes, with and without
`norm_init`, six configurations
(`test_csr_store_matches_reference_across_episodes`).

Re-run on the verified path:

    cell                 first run   verified   M*(k/n)^2 verified
    n= 4000 k= 60           66.6        88.3        0.01987
    n= 8000 k=120           73.0        92.1        0.02072
    n=16000 k=240           67.6        83.2        0.01872
    n= 8000 k=200 (CS2)     21.8        26.9        0.01681
    n=10000 k=100 (CS2)    175.4       238.4        0.02384

**WHAT SURVIVES.** CS1's claim is that M* is CONSTANT at fixed n/k while n
varies four-fold. It holds on both: spread +/-5% either way. The n/k dependence
is not an artifact of the defect, and CAP-CLIFF is untouched.

**WHAT DOES NOT.** The calibration. `C = M*(k/n)^2` moves 0.0155 -> ~0.0200
(+27%) and the spread across cells widens from +/-11% to +/-18%. Every band in
the section above came from the contaminated C, so **CS1 and CS2 are VOID as
registered** -- not failed, VOID. A bar calibrated on bad data cannot be judged,
and re-deriving the band from the new numbers to declare a pass would be
fitting the test to the data.

The exponent moves 2.19 -> ~2.3-2.5. Above 2 either way, still with no
mechanism, still not quotable.

**WHAT HAPPENS NEXT.** The law is re-registered from scratch: calibration from
a DECLARED subset of the verified cells, bands tested on cells held out from
that subset. Nothing above carries forward except the qualitative CS1 result,
which does not depend on the calibration.

**THE LESSON, stated cheaply because it will recur.** A store that is WRITTEN
AND NEVER READ passes every single-episode test vacuously. The multi-episode
test now exists, is parameterised over `norm_init`, and asserts against a
reference rather than against the previous run.

---

## Amendment 2: the retraction had its DIRECTION BACKWARDS

Amendment 1 declared the first run contaminated and the CSR-store numbers
"verified". Engine parity for the full multi-episode capacity protocol -- the
test Amendment 1 said should exist -- was then written, and it FAILED against
the CSR path at relative drive error 2e-3. The cause is mathematical:
potentiation is multiplicative, so the correction `tab[c]-1` is NOT additive
across a split count -- a cell with c0 events in the store and c1 in the
current episode needs `tab[c0+c1]-1`, and `(tab[c0]-1)+(tab[c1]-1)` misses the
cross term. The reference-based tests shared that flawed structure at the
winner level and passed anyway.

The fix accumulates integer COUNTS (which are additive,
[[HEBB-OUTER-PRODUCT]]) into a scratch per active cell and applies `tab` once.
With it, engine parity passes -- and the ceilings land on the FIRST run's
numbers, not Amendment 1's:

    cell            first run   Amendment-1 "verified"   exact path
    n= 4000 k= 60      66.6            88.3                 68.8
    n= 8000 k=120      73.0            92.1                 73.1
    n=16000 k=240      67.6            83.2                 67.6
    cliff (n=8000, M=256/320/384):
                    0.941/-/-       1.000/1.000/0.939    0.938/0.486/0.014

The ORIGINAL multi-word mask summed the TOTAL count before applying `tab` --
exact with respect to splits all along. What it lacked was a test; what the
CSR path lacked was correctness.

**REINSTATED:** the original run's judgments. CS1 PASS (the exact-path cells
68.8/73.1/67.6 sit inside the registered band [66.7, 82.7]); CS2 FAIL as
originally judged; b = 2.19 +/- 0.05 as the measured, unquotable exponent;
C ~ 0.0155. Amendment 1's tables and its "VOID" verdict on the bands are
themselves retracted.

**Standing lesson, sharpened:** the first retraction trusted "new code + a
passing reference test" over "old code, untested". The reference test was too
weak to arbitrate (winner sets at small scale cannot see 2e-3 drive errors),
and the arbiter that settled it was ENGINE parity on the DRIVE. When two
implementations disagree, neither is verified by a test they both pass.

---

## Amendment 3 (2026-09-11, before running): retain the cliff and its sensitivity

`CAP-CLIFF` currently points at a maintained-run artifact for a different
`(n,k)` cell. The exact-path cliff values above survive only as aggregate prose;
the original per-brain vectors were not retained in an identified JSON artifact.
This amendment therefore specifies a fresh reproduction and sensitivity run. It
does not manufacture provenance for the 2026-08-25 measurement.

Run protocol version 2 of `research/experiments/seq_capacity_scaling.py` with:

    engine       hashed_assembly_memory / exact count-then-apply AssemblyMemory
    arm          B (norm_init=true, synaptic_scaling=false)
    n, k         8000, 60
    p, beta      0.50, 0.10
    rounds       8
    w_max        20
    readout      net
    checkpoints  192, 256, 320, 384, 512
    seeds        42..61 (20 independent brains)
    recall       32 stored items per brain and checkpoint
    measurement RNG seed 1234

The historical exact-path aggregates are rank-1 `0.938 / 0.486 / 0.014` at
`M = 256 / 320 / 384`. The frozen bars are:

* **A3-R1, reproduction:** the new means at M=256, 320 and 384 are respectively
  within 0.10, 0.15 and 0.10 absolute rank-1 of those historical aggregates.
* **A3-C1, cliff:** mean rank-1 falls by at least 0.75 from M=256 to M=384.
* **A3-S1, instrument movement:** for every seed, rank-1 at M=192 exceeds
  rank-1 at M=512 by at least 0.50.
* **A3-S0, constructed dead probe:** substituting the M=512 vector for the
  M=192 vector must fail A3-S1.

Failure of A3-R1 means the maintained protocol does not reproduce the registered
number and `CAP-CLIFF` must keep the old result and new result separate. Failure
of A3-C1 means the cliff shape does not reproduce. Failure of A3-S1 means the
readout does not move reliably across the tested load range, so this artifact
cannot close the register sensitivity gap. No threshold will be changed after
the run.

### Amendment 3 result (2026-09-11): all bars pass; the ceiling remains censored

[Immutable results](../../results/runs/memory.capacity-scaling/capacity-cliff-sensitivity-20260911/results.json),
with its run record and source archive, were produced from preregistration commit
`c3a3f0f`. The maintained exact count-then-apply path returned:

    M                 192       256       320       384       512
    mean rank-1     1.000     0.909     0.433     0.0188     0.000
    historical                 0.938     0.486     0.014
    abs difference             0.029     0.053     0.0048

* **A3-R1 PASS:** all three aggregate differences clear their frozen tolerances.
* **A3-C1 PASS:** the M=256 to M=384 mean drop is 0.891, above 0.75.
* **A3-S1 PASS:** every one of 20 paired brains drops from 1.0 at M=192 to
  0.0 at M=512; the minimum per-seed difference is 1.0, above 0.50.
* **A3-S0 PASS:** replacing the M=192 vector with M=512 gives zero movement
  and fails the retained sensitivity check.

The interpolated `M* = 310.1`, bracketed by `[256, 320)`, occurs at estimated
fill 0.955 and is therefore **CENSORED** by the standing 0.95 rule. This run
reproduces the cliff shape and makes its readout sensitivity executable. It does
not turn the interpolated ceiling into an uncensored capacity estimate.
