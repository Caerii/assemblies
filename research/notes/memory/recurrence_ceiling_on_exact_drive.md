# norm_init buys 8x capacity under recurrence — or 1x, depending on the engine

Task #90. `research/experiments/task90_recurrence_ceiling_on_exact.py`,
n=1000, k=50, beta=0.1, p=0.05, one shared area, 6 seeds, M items each built
for 6 rounds. Ceiling = largest M in `(2,4,8,16,32,64)` with rank-1 identity
above 0.90.

## The claim under test

`core/brain.py` justifies keeping `recurrent_projection` off with a measured
table and an inference drawn from it: norm_init "moves the ceiling from M=4 to
M=32", and the ceiling "scales with n (M=32 / 64 / 256 at n=1000 / 2000 /
4000), which is what identifies it as accumulated potentiation rather than
degree bias."

Every number there was measured on `numpy_sparse`, whose candidate sampler
invents a drive for neurons that have not fired, with an error that moves with
LOAD (`graded_similarity_and_sampler_load.md`).

## Result

|  | sparse ceiling | exact ceiling |
| --- | ---: | ---: |
| recurrent, norm_init ON | M=32 | **M=16** |
| recurrent, norm_init OFF | M=4 | **M=16** |
| feed-forward, norm_init ON | M=64 | M=64 |

**norm_init's capacity gain under recurrence: 8.0x on the sampler, 1.0x on
exact drive.**

The gain is not small on exact — it is absent at this resolution. norm_init
still does something (at M=32 accuracy is 0.73 with it and 0.44 without), but
it does not move the ceiling, and the 8x is the sampler's.

## The sampler's error CHANGES SIGN between the arms

`spread`, exact minus sparse (positive = the sampler called it more distinct):

| arm | M=2 | M=4 | M=8 | M=16 | M=32 | M=64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rec, norm | +0.063 | +0.038 | +0.020 | +0.014 | +0.040 | +0.142 |
| rec, RAW | **-0.320** | **-0.322** | **-0.443** | **-0.575** | **-0.682** | **-0.487** |
| ff, norm | +0.020 | +0.023 | +0.022 | +0.011 | +0.010 | +0.008 |

With norm_init on, the sampler slightly overstates distinctness. With it off,
the sampler overstates COLLAPSE, by up to 0.68 overlap. Feed-forward — where
little recruitment happens — is clean on both engines, which is the internal
control that the disagreement is about recruitment and not about the protocol.

## What this costs a standing conclusion

`graded_similarity_and_sampler_load.md` concluded: absolute overlaps are
suspect, but *"paired comparisons survive — both arms of any A/B run on the
same engine at the same load, so the difference is real."*

That reassurance is too strong, and this is a counterexample. The pairing here
is norm_init on vs off. It does not survive: 8.0x becomes 1.0x. The reason is
that the sampler's error is a function of load, and **norm_init changes which
neurons win, hence how fast the area recruits, hence load.** The two arms
therefore do not share the sampler's error — the difference between them is
partly the difference between two different errors.

The corrected rule: an A/B measured on the sampler is safe only when the
manipulation does not move recruitment. `beta`, plasticity on/off, and frozen
vs plastic are probably safe. Anything that changes WHO wins is not.

## What survives

* Recurrence is still the collapse channel, and still much worse than
  feed-forward. Both engines agree on the sign, at every M.
* Feed-forward still has no ceiling in this range on either engine (M=64,
  spread 0.058 against a floor of 0.050).
* `recurrent_projection` should still not be flipped on globally. If anything
  the exact reading strengthens that for the ON case (ceiling M=16, not M=32)
  and weakens the stated REASON (it is not that norm_init rescues recurrence).

## What is now unsupported

The n-scaling claim. It was inferred from ceilings measured on an instrument
whose error moves with load, and n at fixed k IS load. This is not a claim that
the ceiling does not scale with n — it is that the evidence for it does not
survive. Settling it needs the sweep re-run on exact drive at n = 1000 / 2000 /
4000, which is now affordable.

## Two defects found in the process, both mine

1. **`_stim_norm` was degenerate.** A stimulus fires in full and is stored
   pre-summed, so its base count IS the target neuron's in-degree from that
   stimulus; dividing it by itself gave exactly 1.0 everywhere, k-WTA fell
   through to the index tie-break, and every stimulus elected the same k
   neurons. `numpy_sparse._norm_scale` documents this exact trap and passes
   `n_pre = target.n, rows_known = stim.size` instead. Fixed to match.

2. **`Brain` forwards `norm_init` only when True**, so omission is how it says
   False — and this engine defaulted it to True, silently ignoring
   `Brain(norm_init=False)`. Every "un-normalised" run on it was the production
   substrate. Default flipped to False; a registry-walking contract test now
   pins it for every engine.

The first sweep, run with both defects live, "confirmed" the pre-registered
hypotheses at max gap 0.87 and would have been a much more dramatic result.
What refused it was not a hypothesis test: it was the feed-forward CONTROL
reading 0.89 spread when it cannot collapse, and the norm/RAW arms agreeing to
four decimals when a live parameter cannot do that. The control was worth more
than the contrast.
