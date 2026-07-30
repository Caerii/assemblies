# The potentiated-set vs never-fired-pool competition

Derived and measured 2026-07-30. This is the substrate law behind #53
(reciprocal restoration), #56 (the sequence bridge), the `norm_init` stability
threshold, and `critical-load-alpha-star`. It exists so those are designed
against a measured law rather than tuned by search.

All numbers: `numpy_sparse`, `norm_init` default, n=10⁴, k=100, p=0.05, a=k,
3 seeds. Scripts in the session scratchpad; the measurements are reproduced
below in full so the conclusions can be checked without them.

## The setup

An area holds an assembly whose synapses carry gain `g`; a projection arrives;
k-WTA chooses k winners from

    k  potentiated incumbents   drive ~ g · Binomial(a, p)
    N  never-fired candidates   drive ~     Binomial(a, p)      N = n - w

Both are divided by the same `norm_init` scale, so they are directly comparable
(this is only true since `54e6c00`; before it, heterogeneous-`n` fibers were
priced up to 10× apart).

## Finding 1 — there are two regimes, and only one is tunable

| driver | β = 0 | β = 0.05 | β = 0.1 | β = 0.2 | β = 0.4 |
|---|---|---|---|---|---|
| stimulus only, T=5 | 0.5000 | 0.8367 | 0.8367 | 0.8367 | 0.8367 |
| stim + recurrence, T=5 | 0.1367 | 0.2433 | 0.3833 | 0.6300 | 0.8400 |

**The stimulus fiber is a step function in β, not a curve.** `_norm_scale`
snapshots each neuron's own initial stimulus weight as its degree base, so every
materialized neuron starts at drive *exactly* 1.0 — there is no incumbent
variance for gain to work against, and any β > 0 immediately creates a strict
ordering. Retention is identical for β from 0.05 to 0.40.

Consequence: **do not tune β for stimulus-driven (feed-forward) construction.**
It has no graded effect there. Every tunable question in this repo lives on the
recurrent fiber. This also explains why feed-forward lexicon building has no
measurable ceiling while recurrent paths collapse
(`recurrence-is-the-collapse-channel`).

## Finding 2 — the connectivity ceiling

As g → ∞ under iid incumbents, retention → `1 − (1−p)^k`: a neuron that has
**no** afferent from the assembly can never be recalled, whatever the gain.
At k=100, p=0.05 that is 0.9941, and the Monte-Carlo model saturates at exactly
0.9944.

This is a property of connectivity alone — independent of β, T, n, and `w_max`.
It bites hard when `k·p` is small: at k=20, p=0.05 the ceiling is 0.642. That is
the same quantity as `kp-decides-whether-beta-helps`; **check `k·p` before
tuning anything.**

The real engine reaches 1.0000, above this ceiling, because assembly members are
*selected* as top-k by drive, so their afferent counts are order statistics
rather than fresh draws. Modelling that (draw the population, take the top k,
potentiate, re-compete) reproduces the 1.0000 ceiling exactly.

## Finding 3 — the transient dominates, and gain cannot fix it

The two static models bracket the asymptotics but **neither describes low T**:

| β | T | g=(1+β)^T | iid model | selected model | engine |
|---|---|---|---|---|---|
| 0.05 | 5 | 1.28 | 0.0660 | 0.9956 | **0.0300** |
| 0.10 | 5 | 1.61 | 0.2326 | 0.9999 | **0.0533** |
| 0.20 | 5 | 2.49 | 0.5630 | 1.0000 | **0.2067** |
| 0.40 | 5 | 5.38 | 0.8796 | 1.0000 | **0.4300** |
| 0.10 | 20 | 6.73 | 0.9614 | 1.0000 | 0.9933 |
| 0.20 | 20 | 38.34 | 0.9944 | 1.0000 | 1.0000 |
| 0.40 | 10 | 28.93 | 0.9944 | 1.0000 | 1.0000 |

At T=5 the engine sits **below even the pessimistic iid bound**, at every β. At
T ≥ 10 it matches the static law well (errors 0.006–0.06 for β ≥ 0.2).

The reason is that `(1+β)^T` is the gain of a synapse that co-fires on *every*
round, and during the transient the winner set is still churning so most do not.
Measured realized gain (mean weight of present within-assembly synapses,
ambient = 1.0):

| β | T=5 | T=10 | T=20 | nominal at T=20 |
|---|---|---|---|---|
| 0.05 | 1.01 | 1.08 | 1.64 | 2.65 |
| 0.10 | 1.03 | 1.42 | 3.69 | 6.73 |
| 0.20 | 1.11 | 2.55 | 15.61 | 38.34 |
| 0.40 | 1.34 | 6.85 | **20.00** | 836.68 |

**The design rule: rounds buy convergence, β buys maintenance, and β cannot
substitute for rounds.** β=0.4 at T=5 (0.43) is far worse than β=0.1 at T=20
(0.99), despite 5.4× the nominal gain. Below T_conv ≈ 10 at these parameters,
no amount of gain helps.

## Finding 4 — `w_max` is a hard ceiling on gain

Realized gain saturates at exactly **20.00 = `DEFAULT_W_MAX`**
(`neural_assemblies/constants/default_params.py`). Gain cannot exceed it however
long or hard you train. The required gain grows only *logarithmically* in n
(closed form `g_c = 1 + √(2 ln N (1−p) / (a p))`: 2.61 at n=10³, 3.29 at n=10⁶),
so at k=100, p=0.05 `w_max` is comfortable — but it is a real ceiling to check
before assuming a target is reachable, and it interacts with Finding 2: no gain
at all rescues a neuron with zero afferents.

## Application

**#56 (the sequence bridge).** The recorded structural cause is that
`stim_rounds = rounds_per_step - 2` leaves Phase B at exactly two rounds
regardless of T. Finding 3 says why `phase_b_ratio` alone did nothing and why
`beta_boost=0.3` only reached 1.80 of 3: **two rounds is inside the transient**,
where gain has no leverage. So the order of work is forced — make Phase B scale
with T *first*, and only then does `beta_boost` have anything to act on. Do not
search over `beta_boost` before the round structure is fixed; the search is
being run in the regime where the parameter provably cannot help.

**#53 (reciprocal restoration).** Restoration is a recurrent-regime problem, so
Findings 2–3 apply. Before treating the ~0.77× gap as a defect, check the
connectivity ceiling at the *sparse k/n* parameters where the gap was measured —
`1 − (1−p)^k` may already be below the reference figure, in which case part of
the gap is not recoverable by tuning.

## What this law does NOT do

It predicts steady-state retention (T ≥ 10) to within ~0.03, and it gives exact
asymptotic ceilings. It does **not** model the convergence transient, which is
where every low-T failure in this repo lives. A predictive transient model would
need the churn dynamics — how the winner set contracts round over round — and
that is not attempted here. Treating the static law as if it applied at T=5
would over-predict by up to 0.45.
