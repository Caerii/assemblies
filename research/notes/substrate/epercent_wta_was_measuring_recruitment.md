# E%-WTA's "emergent assembly size" was the recruited pool

#94. Implementing winner policies on `numpy_exact` immediately produced a
result about every E%-WTA number this repository has recorded.

## The mechanism, and why the engine it runs on decides what it measures

Hoff et al. (2026) replace k-WTA with a window on the drive distribution,

    F_t = { j : h_j(t) in [(1 - eps) * h_max(t), h_max(t)] }

so assembly SIZE is emergent rather than fixed at k. That is a claim about the
**distribution of the drive** -- which makes it precisely the wrong mechanism to
run on an engine that INVENTS the drive for neurons that have not fired.

`numpy_exact` listed `winner_policy` in `_UNSUPPORTED_AREA` and rejected it
outright, so every E%-WTA result here was sampler-side by default:
`assembly_calculus/epwta.py`, `programs/markov_coin.py`,
`parity/executors.py`, `test_literature_parity.py`, `test_literature_golden.py`.

## Measured, same protocol, both engines

    n=2000, k=50, p=0.05, beta=0.1, EPercentPolicy(fraction_of_max=0.5)

    round        1     2     3     4     5     6
    sparse |F|  50   100   150   200   250   300
    sparse  w   50   100   150   200   250   300      <- identical, every round
    exact  |F| 184   184   184   184   184   184

On `numpy_sparse` the firing set is **identically the set of neurons that have
ever fired**. Every recruited neuron fires again, every round, and the "emergent
size" is a readout of recruitment rather than of any competition. It grows by
exactly k per round because that is how fast the pool grows.

On exact drive the window selects 184 of 2000 and settles there -- a genuine
selection, stable, neither k nor n.

## Why the sampler does this

The candidate sampler draws fabricated drive for never-fired neurons from the
truncated upper tail of a distribution fitted to the population, and the
materialized neurons carry real accumulated weight. The materialized set
therefore sits within `(1-eps)*h_max` of the peak while the sampled candidates
fall below the window. The result is a partition by MATERIALIZATION STATUS, not
by drive -- so the mechanism whose entire point is that size is data-dependent
returns a number determined by bookkeeping.

## What this does and does not say

It does NOT say the published E%-WTA results are wrong in direction. Assembly
sizes there were reported as emergent and they are certainly not fixed at k.
It says the quantity being reported is not the one the mechanism defines, and
any conclusion resting on the SIZE -- recovery rate as a function of assembly
size, the comparison against k-WTA, the parity goldens -- is measuring the
sampler's candidate pool.

Re-deriving those is now possible and was not before.

## Pinned

`TestWinnerPolicies` in `test_exact_engine_ladder.py`:

* exact accepts a policy at all (it raised before);
* exact's size is strictly between k and n and settles;
* **the sampler's `|F| == w` identity is pinned as a KNOWN DEFECT**, so it
  cannot quietly change. If that test starts failing, the candidate drive has
  changed and every E%-WTA result needs re-deriving again;
* `TopKPolicy(k=k)` takes the fast path and agrees bit-for-bit with no policy;
* `input_noise_std` stays refused, and the message says why -- it needs an RNG
  stream, and this engine deliberately has none.

## One more instance of the same old thing

The first version of the discriminating test asserted `|F| != area.w`. That is
meaningless on this engine: `Brain.Area.w` tracks the last winner count while
`engine.w` is `n` -- 184 against 2000 for the same area at the same moment.
`w` is two different quantities depending on which object is asked
([[same-name-two-meanings]], #83), caught here for a second time, inside the
test written to catch a different degeneracy.
