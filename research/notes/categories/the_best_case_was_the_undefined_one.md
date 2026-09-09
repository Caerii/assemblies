# The best case was the undefined one

`diagnostics.area_health` reports a `margin` -- best match over second best --
as the guard against a readout that is nominally accurate but one step from
failing. Measured on 2026-08-05: **the margin was undefined exactly when
separation was perfect.**

## The construction

Three disjoint stored assemblies, each cue re-presenting exactly what was
stored. Nothing about this case is degenerate; it is the best result the metric
can describe.

```
=== PERFECT separation (disjoint, exact re-cue)
  accuracy   1.0
  margin     nan
  identity   1.0
  trustworthy True   collapsed False
    [ok ] live probe: margin nanx
```

`accuracy 1.0000`, `margin nan`, and the verdict line reads **OK**.

## Why

```python
if len(sims) > 1 and sims[1][0] > 0:
    margins.append(sims[0][0] / sims[1][0])
```

A margin was recorded only when the runner-up had NONZERO overlap with the live
read. Zero overlap with every non-match is the best possible separation, so the
best probes contributed nothing and `margins` came out empty -- and empty meant
`float("nan")`.

Three distinct situations were collapsed into that one NaN:

| situation | truth | was reported |
|---|---|---|
| runner-up overlaps zero, best > 0 | margin is **unbounded** | dropped |
| only one item stored | genuinely **undefined** | NaN |
| live read matches nothing at all (0/0) | genuinely **undefined** | NaN |

## What it cost

NaN does not raise; it answers False. So:

* `abs(nan - 1.0) < 1e-9` is False -- the **dead-probe check could not fire** on
  an unmeasurable margin, and certified the probe live.
* `.trustworthy` covers only the `live probe` and `index space` verdicts, so it
  returned True.
The bias operates at two levels, and the first is unconditional:

* **Per probe, always.** `margins` never contained a perfectly-separated probe,
  so `statistics.mean(margins)` — the margin `area_health` reports for a single
  seed — was already an average over the imperfect probes only. Every reported
  margin in this codebase is downward-biased by construction, in every seed.
* **Per seed, when a whole seed was perfect.**
  `research/experiments/overnight_characterization.py:143` then filtered NaN by
  hand (`h.margin == h.margin`) while averaging `accuracy` unguarded on the line
  above, dropping any seed in which EVERY probe separated cleanly.

How large the second effect was is **not measured** — it depends on how often an
entire seed came out perfect, which nothing recorded. The first effect needs no
measurement: it is structural, and it is why the number was **biased downward by
a filter that looked like hygiene**.

## What has to be retracted: nothing

Checked before claiming a cost. No note in `research/notes/` or `docs/` cites an
`area_health` margin numerically — the `margin 11.9x` figures in
`a_grammar_that_cannot_reject.md` and `erp_metric_is_clipped.md` are
`p600_margin`, a detector THRESHOLD, an unrelated quantity. The biased number
appears only in `overnight_characterization`'s capacity table and in the
self-check's console output, neither of which is cited as a result.

So the defect is real and structural, and its published blast radius is zero.
Both halves of that are worth stating: "biased by construction" without "and
here is what it changed" is the kind of alarm that costs a day.

## The fix

*Unbounded is not undefined.* `best/0` with `best > 0` is infinity, and saying
so makes the mean infinite -- loud, and impossible to publish by accident. Only
`0/0` is undefined. Every quantity on `AreaHealth` is now a
`core.measurement.Measured`, so an undefined one cannot be compared, averaged,
or formatted without raising; the verdicts test `.defined` instead of the NaN
idiom; and `format_report` prints `n/a` rather than a number it does not have.

## A second defect the fix exposed

Writing the guard test in both directions (memory rule 6) turned up a case the
dead-probe check could never see, **before this change or after it**:

> reads are constant, but the STORED assemblies are distinct.

Every cue returns the same assembly -- the definition of a dead probe -- yet the
runner-up overlap is zero, so the margin is unbounded: the best-looking value in
the range. Keyed on the margin, the guard passes it. The check now asks the
reads directly (`len(read_signatures) == 1`), which is the quantity it was
always trying to infer.

## Siblings checked

Two other margins exist -- `parse_errors.LexicalReadout.margin` and
`programs/colt_mnist_lri_readout` -- and both are **differences**, not ratios,
so a zero runner-up gives `best - 0`, not a division. Neither shares the defect.
The ratio form was unique to `area_health`.

## Guards

`neural_assemblies/tests/test_area_health_definedness.py`, 13 tests, each
asserting in both directions: perfect separation yields a defined unbounded
margin AND imperfect separation still yields the same finite ratio it always
did; the empty area gets no distinctness verdict AND a real area still gets one;
constant reads are flagged AND a healthy probe is not.
