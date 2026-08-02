# The recurrence ceiling does scale with n — and not by the law anyone stated

> ## ⚠️ THE SUPER-LINEAR READING IS WITHDRAWN PENDING A CONTROL
>
> Everything below about the coverage law and α\* being "refuted" was measured
> at **fixed absolute gain**: `beta = 0.10` and `T = 6` held constant while `n`
> varied. [[critical-load-alpha-star]] states the rule in one line —
> *"whenever `g_c` depends on an axis, comparing along that axis at fixed
> ABSOLUTE gain is confounded — this bit the n sweep and then the k sweep
> identically"* — and this is the same n sweep with the same design.
>
> The withdrawn `n^1.49` came from exactly this. The **controlled** version of
> that measurement, at fixed *relative* gain `g = 0.88·g_c(n)`, came out
> **extensive: exponent 1.01, `M_max ≈ 1.15 n/k`**. My fine grid reports
> **1.70**, which is what the confound is known to manufacture, and its
> proximity to 1.49 is the signature rather than corroboration.
>
> **What survives:** the ceiling grows with n (true at `a = 1` too), the
> sampler did not produce that growth, and the feed-forward control has no
> ceiling. **What is suspended:** super-linearity, "the coverage law is
> refuted", "α\* is refuted", and "one big area holds more than the sum of its
> parts". The many-areas SPLIT arm inherits the same suspension, since its
> capacity loss was read through the same lens.
>
> `task90_gain_confound.py` is the decisive test — measure `a` at several
> absolute gains; if it moves, it was never a property of the substrate.

Task #90, second half. `research/experiments/task90_ceiling_n_scaling.py`,
k=50, beta=0.1, p=0.05, buildT=6, one shared area, 3 seeds. **k and p held
fixed while n varies**, which is what a coverage law's prediction requires.

## The claim under test

`core/brain.py` said the ceiling "scales with n (M=32 / 64 / 256 at n=1000 /
2000 / 4000), which is what identifies it as accumulated potentiation rather
than degree bias." The companion half of that same table did not survive exact
drive (`recurrence_ceiling_on_exact_drive.md`: norm_init's 8.0x capacity gain
is 1.0x), and n at fixed k IS load, so the scaling claim was marked
UNSUPPORTED pending this measurement.

## Result — the claim survives

    rec ceiling (rank-1 identity > 0.90)
      numpy_sparse    n=1000 M=32    n=2000 M=64    n=4000 M>=256
      numpy_exact     n=1000 M=16    n=2000 M=64    n=4000 M>=256

    sampler inflation (sparse / exact):  2.0x   1.0x   1.0x

    ff ceiling, BOTH engines:  M>=256 at every n   (the control)

**This is the one #90 claim that survives contact with exact drive.** The
inflation does not grow with n, so the sampler did not manufacture the scaling.
`brain.py`'s inference was right — it just had not established it.

My pre-registered Z3 said the opposite: that the 4x final step was where the
sampler's low-load error runs away. **Falsified.**

## But it is not the law anyone stated

On the one uncensored doubling, exact goes **16 -> 64: four-fold per doubling
of n**, not two-fold.

* Not the **coverage law**. That predicts `M_max ∝ n` at fixed k. Critical
  coverage `M·k/n` at the ceiling instead RISES with n — 0.8, 1.6, >=3.2 —
  which is precisely what a coverage law forbids.
* Not **α\***. `M_max ≈ 1.15 n/k` predicts 23 / 46 / 92 against measured
  16 / 64 / >=256.

## What is NOT established: the exponent

n=4000 is **censored** — both engines still read acc 1.0000 at M=256, the top
of the sweep, with spread 0.015 against a floor of 0.0125. So "M=256" there is
a statement about `M_SWEEP`, not about the substrate, and "four-fold per
doubling" rests on **one clean step and three points**.

That bar is deliberately high here: an `n^1.49` capacity claim in this repo was
already withdrawn once as a fixed-absolute-gain artifact
([[critical-load-alpha-star]]). Two clean steps would be enough to say
"super-linear, and not the stated law". They would still not be enough to fit
an exponent. `task90_ceiling_extend_n4000.py` goes after the second step.

## The mechanism to test next

Coverage is the wrong frame. `norm-init-stability-threshold` says an assembly
survives its own recurrence only while `(1+beta)^T` beats the extreme value of
the candidate pool — and the pool is n draws. Extreme-value statistics over n
candidates grows like `sqrt(2 ln n)` in the Gaussian case, so the *threshold*
moves slowly while the number of assemblies that can coexist below it can move
much faster. That predicts a ceiling growing faster than n, with no reference
to coverage at all, and it is testable directly: sweep `T` and `beta` at fixed
n and check whether the ceiling tracks `(1+beta)^T` against the pool maximum
rather than tracking `n/k`.

## Reading this alongside the first half

| | sampler | exact | verdict |
| --- | --- | --- | --- |
| norm_init's gain under recurrence | 8.0x | **1.0x** | claim was the instrument |
| ceiling scales with n | yes | **yes** | claim survives |
| ff has no ceiling in range | yes | **yes** | control, holds |

So of the two inferences `brain.py` drew from one table, one was the instrument
and one was real. That is the useful shape of the answer: the sampler is not
uniformly wrong, and "measured on the sampler" is a reason to check rather than
a reason to discard.
