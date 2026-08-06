# Afferent energy: NOT adopted, and the old rejection was void anyway

**Verdict: keep `_self_recurrent_energy`.** Afferent drive does not discriminate
better; it only uses more of the range, and not enough more to matter.

## Why the question had to be re-asked

`afferent_energy` was rejected on measurement: p600 AUC exactly **0.000, with
ZERO variance across four seeds**. That reading is void. It was taken under the
old dispatch, where the violation arm probed **VP** and the control probed
**ROLE_PATIENT** — so the number describes two different brain areas, not the
metric. Zero variance across seeds is the signature of a *constant*, which is
what a probe on an area with no self-fiber returns.

## Why it is now a different question, not a re-run

`afferent_energy` existed because it was the only quantity **defined for both
arms**: `VP -> VP` is shape (0,0) with zero synapses, so the shipped metric
returned a constant on the violation arm. With `expected_slot` adopted
(`f79c4f5`) both arms probe ROLE_PATIENT, which has a self-fiber — **the shipped
metric is now defined on both arms too, and the candidate's structural
justification is dissolved.** What remains is a plain comparison of two
well-defined metrics, which admits the boring answer.

## The measurement

Cold, 10 seeds, counterbalanced, `disk_hits=0 trained_fresh=10`:

| metric | self-recurrent | afferent | delta |
|---|---|---|---|
| `p600_auc_of_raw` | 0.7167 ±0.0805 | 0.7500 ±0.0732 | **+0.0333 ±0.0627** |
| `p600_span_of_raw` | 0.0064 ±0.0008 | 0.0123 ±0.0010 | **+0.0059 ±0.0005** |

**AUC: no detectable difference.** The delta interval is [−0.029, +0.096] and
spans zero. At 10 seeds and 1/9 AUC granularity, afferent drive does not order
the conditions better.

**Span: decisively wider, ~1.9x.** That interval excludes zero comfortably.

Those two rows say different things and both are worth keeping: the candidate
spreads the conditions further apart without ordering them better.

## The span is real and still not enough

`erp_metric_is_clipped.md` records that the graded detector needs a margin of
0.0760 against a max observed excess of 0.0064 — 11.9x short. The
self-recurrent span here reads **0.0064**, reproducing that figure exactly,
which is a useful check that this is the same quantity.

| metric | span | the 0.0760 margin is |
|---|---|---|
| self-recurrent | 0.0064 | 11.9x above it |
| afferent | 0.0123 | 6.2x above it |

So afferent drive **halves the shortfall and the graded detector is still
dead.** That is the strongest thing that can be said for it, and it is not
enough to justify carrying a second energy definition.

## The methodological error, which is the part to remember

The first run printed `VERDICT: PASS`. It was answering the criteria I encoded —
`above`, `on_every_seed`, `must_vary`, `allow_decrease=False` — none of which
test a difference. The docstring said "adoption requires the candidate to BEAT
the shipped metric"; `Criteria.delta_excludes_zero` existed, defaulted to False,
and I did not set it.

> A pre-registration is only as good as the predicate it encodes. Prose in a
> docstring is not a bar.

The harness was built precisely so an adoption bar is stated before the numbers
arrive, and it worked — it just enforced a weaker claim than the one I wrote one
paragraph above it. The criteria are corrected in the script; the recorded delta
fails the corrected bar, which is why the verdict here is NOT ADOPTED.
