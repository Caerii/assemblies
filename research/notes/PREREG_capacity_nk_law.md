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
