# Passive voice works end to end — after two spellings of one lookup disagreed

## The result

```
'the cat is chased by the dog'   is_passive=True    cat=PATIENT  dog=AGENT
'the ball is held by the boy'    is_passive=True    ball=PATIENT boy=AGENT
'the dog chases the cat'         is_passive=False   dog=AGENT    cat=PATIENT
'the boy holds the ball'         is_passive=False   boy=AGENT    ball=PATIENT
```

`is_passive` fires 2/2 on passives, 0/2 on actives. The active control is what
matters as much as the passive: the failure mode of a passive-bearing corpus is
teaching the DETERMINER to reverse, after which every sentence parses
backwards.

And the gating is learned, not spelled:

| subcategory | reverses | confidence | n_examples | n_contrast | p_present | p_absent |
|---|---|---|---|---|---|---|
| `MARKER` | **True** | 0.960 | 9 | 50 | 1.00 | 0.04 |
| `DET` | False | 0.000 | 59 | 0 | 0.19 | nan |

`n_contrast = 50` is the load-bearing number: the marker is present in 9
sentences and ABSENT in 50, so its effect is a contrast rather than a marginal
rate. `DET` occurs in every sentence, has no contrast set, and correctly scores
zero — which is the guard that keeps the determiner from gating voice.

## Why the passive is the point

In an all-active corpus, position predicts role perfectly, so a role
representation buys nothing over a position counter. The passive states the
SAME event in the opposite order. It is the one construction where a positional
reading is not merely uninformative but INVERTED — which makes it both the
reason roles must come from the scene and the evidence that they now do.

The generated pair, from one event record:

```
event(agent=dog, patient=cat, action=chase)
    the dog chases the cat            agent  ... patient
    the cat is chased by the dog      patient ... agent
```

Roles verified BEFORE anything trained on them (the guard that would have
caught "the store build the dog"): of the passives generated, subject=patient
and by-phrase=agent throughout, with the only exceptions being ties where two
participants carry identical feature bundles and `role_of_features` correctly
declines to guess.

## The two defects, both one-question-two-spellings

This is the third and fourth instance of the same shape in this arc.

**1. The learner could not see the marker.** `_learn_gating_patterns` collected
subcategories only from words where `not ctx.is_grounded`. But "by" carries
spatial grounding, so it IS grounded and was filtered out before
`_func_subcat_of` was ever called — and that method's docstring says precisely:

> Falls back to the grounding signature so a role marker such as "by" — which
> carries spatial grounding and therefore is not an ungrounded function word —
> is still recognised as a MARKER rather than dropping to None.

The fallback was written, and then made unreachable from the only caller that
needed it. **Measured: a passive-bearing corpus taught only `DET`.**

The fix admits prepositions by CATEGORY rather than dropping the guard, because
`_func_subcat_of` calls anything with spatial grounding a MARKER and that would
sweep in ordinary nouns like "beach" — letting a content word gate the voice.

**2. The reader used a different lookup than the writer.**
`_determine_role_order` resolved subcategories with raw
`get_func_subcategory`, which returns only what frame analysis learned.
`_learn_gating_patterns` writes entries keyed on `_func_subcat_of`, which
includes the grounding fallback. So MARKER was learned at confidence 0.960 and
**still could not be found**: `is_passive` fired 0/2. Pointing the reader at the
same function as the writer took it to 2/2.

Neither defect produces an error. Both produce a mechanism that runs, returns,
and decides nothing.

## What this cost, and the honest limits

* **The passive rate is 15.3%**, against roughly 2-10% in English. `PASSIVE_EVERY = 4`
  is a stated choice, not a measurement: the gating learner is contrastive and
  the generated corpus is only ~50 sentences per stage, so a 5% rate would give
  it two examples. Raising the rate toward 1 would make the corpus
  passive-dominant, which is the `DET`-reverses failure.
* **Token frequencies changed.** The corpus now states some events twice, so
  `adaptive_rounds` and anything measured on frequency describes a different
  corpus than before this commit.
* The parse probes are **four sentences on one seed**. They demonstrate the
  mechanism fires and inverts; they are not a rate. A role-accuracy sweep over
  seeds and over the full generated corpus is not done.
* **Semantic plausibility is still unchecked**: "the mouse is enjoyed by the
  cake" is well-formed, correctly role-labelled, and nonsense. The generator
  enforces agreement, transitivity and animacy-of-agent; it does not know that
  cakes do not enjoy.
* `_ROLE_MAP` still covers 3 of 7 thematic areas (#116). It is now defined once
  in `core.areas.ROLE_LABEL_TO_AREA` instead of twice, so the gap is recorded
  in one place rather than rediscovered per call site — but it is not closed.
