# The #108 fix reached two of its three consumers

`expected_slot` was adopted to remove an area-identity confound: the probe's
target area was dispatched on the OBSERVED word category, so a category
violation — a word whose observed category differs from the expected one — read
a different brain area from its control by construction.

`measure_live_integration` reads "the category" **three** times:

| what | what it decides | fixed by #108? |
|---|---|---|
| `role_area` | the probe's TARGET area | yes |
| `phrase_category` | which phrase areas are read | yes |
| `core` | the probe's SOURCE area | **no** |

`core = CATEGORY_TO_CORE.get(category)` is computed before the expected-slot
block and never revisited. So with `expected_slot` on, both arms probe
ROLE_PATIENT — and the violation arm reads **VERB_CORE → ROLE_PATIENT** while
its grammatical control reads **NOUN_CORE → ROLE_PATIENT**.

The comment directly above the block says it out loud:

> When the slot is predicted, the category it predicts is nominal.

…and then applies that to `phrase_category` only. This is rule 1 of
[[one-canonical-way]] — a fix applied to one sibling — committed *by the fix
that exists to close this exact defect class*, which is a sharper version of the
same lesson: knowing the rule does not apply it.

## How it surfaced

Not by inspection. `TRAINED_AREA_MATCHED_CALIBRATION_FRAMES` — items whose words
are all trained and stably classified — scored **p600 AUC exactly 1.0000 with
zero variance across 10 cold seeds**. The pre-registered `must_vary` criterion
failed it, and asking *why a perfect score* pointed here
(`trained_frames_score_1000_and_that_is_the_problem.md`).

Repairing the items did not create the confound. It **removed the noise that was
hiding it**: five of nine default items had no main verb, so the arms differed in
several uncontrolled ways at once.

## The degenerate control

`research/experiments/erp_source_core_identity_control.py`. Every item is
GRAMMATICAL, every critical word is a trained noun in object position, and the
only thing that changes is the category **label** handed to the metric — via
`trial_category_in_sentence`, which is production code, not a reimplementation.

| seed | AUC(VERB-labelled ranked above NOUN-labelled) | span |
|---|---|---|
| 11 | 0.7778 | 0.0058 |
| 12 | 0.8889 | 0.0071 |
| 42 | 0.8889 | 0.0065 |

For comparison, the shipped contrast reads AUC 0.7167 with span 0.0064.

**A control containing no violation of any kind, on identical sentences, scores
about as high as the reported effect, with the same span.** Per item:

```
critical=food   as NOUN=0.9931  as VERB=0.9917   (reverses)
critical=book   as NOUN=0.9892  as VERB=0.9936
critical=bed    as NOUN=0.9920  as VERB=0.9950
```

Not perfectly deterministic — `food` reverses on every seed — so this is a large
confound rather than a complete explanation.

## What it does and does not establish

**Does:** labelling a word VERB rather than NOUN, with sentence, position, word
and target area all held fixed, moves the P600 in the same direction and at a
comparable rank magnitude as a real category violation.

**Does not:** that the whole 0.7167 is artefact. In the real contrast the
critical WORD differs too (`cat` vs `runs`), so genuine signal may sit on top.
Separating them is what the follow-up study is for.

## The fix, measured and ADOPTED

`protocol.expected_slot_source_core` takes `core` from the expected category as
well. Measured cold before adopting, 10 seeds, `disk_hits=0 trained_fresh=10`
(`research/experiments/erp_source_core_study.py`):

| metric | observed source core | expected source core | delta |
|---|---|---|---|
| `p600_auc_of_raw` | 0.7167 ±0.0805 | **0.6056 ±0.0545** | −0.1111 ±0.0898 |
| `p600_span_of_raw` | 0.0064 ±0.0008 | 0.0064 ±0.0008 | +0.0000 ±0.0002 |

`VERDICT: PASS` — above chance on **every** seed, not constant across seeds,
delta CI excluding zero. It shrinks and does not invert, which is what removing
a confound looks like and is the same shape `expected_slot` itself produced.

A large drop was the pre-registered expectation, and **a drop to chance was named
in the study docstring in advance** as a real possible outcome, so that
publishing it would be the default rather than a decision made after seeing the
number. It did not happen: the effect survives.

## The headline has now shrunk twice

| | p600 AUC | what is matched |
|---|---|---|
| originally reported | 0.9056 | nothing |
| `f79c4f5` | 0.7167 | the probe's TARGET area |
| this | **0.6056** | the SOURCE core as well |

**About two thirds of the original figure was area identity.** The remaining
0.6056 is above chance on every seed, and it is the number to quote — with the
standing caveats that AUC granularity here is 1/9, and that the DEFAULT frames
these were measured on still contain untrained words
(`the_calibration_frames_are_untrained.md`), which is a separate open problem
about the ITEMS rather than the metric.

Related: [[one-canonical-way]], [[fake-perfect-probe-signatures]],
`p600_is_confounded_with_area_identity.md`, `p600_the_honest_number_is_0717.md`.
