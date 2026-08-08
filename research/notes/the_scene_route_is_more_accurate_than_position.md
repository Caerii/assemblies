# The role thread is open — and the scene beats the positional inducer

## What was done

`roles=[None] * len(sent)` is gone. The curriculum generator now emits
`SentencePlan(tokens, event)`, where `event` is the `SceneEvent` the sentence
describes, and `ground_plans` derives per-word roles from perception at the one
point where grounding is resolved.

The gate the previous note demanded, run first:

| stage | plans | carrying a `SceneEvent` | sentences with >=1 role |
|---|---|---|---|
| `SENTENCES` | 50 | 50 | 50 |
| `COMPLEX_GRAMMAR` | 50 | 50 | 50 |

Non-`None` roles reach the pipeline. Everything after this is not void.

## The free correctness check, and what it caught

Scene-derived roles and the positional inducer compute the same quantity by
unrelated means — perceptual features plus causal order on one side, position
relative to the verb on the other. So they can be compared, and disagreement
localises a defect rather than merely reporting one. Three fell out.

**1. Resemblance was being treated as reference.** `role_of_features` scored
participants by best feature overlap and took the winner. That is correct when
the word denotes one of the participants — it is what separates `boy` from
`girl`, both `PERSON`. It is wrong when the word denotes neither, because then
there is no competitor for it to lose to and a single shared superordinate
feature wins by default:

```
the small meat creates the money on the chicken   ->  chicken = AGENT
the proud friend wakes away the bear              ->  bear    = AGENT
```

10 of 130 role assignments, every one a noun inside a prepositional phrase. The
fix is a containment requirement — the word's features must be a subset of the
participant's or a superset of them — and `girl` against `[BOY, PERSON]` is now
correctly rejected rather than merely out-ranked.

**2. The same bug had a second spelling.** `is_action` used a bare
intersection, so on "the closed lamp seems towards the fish" both `closed` and
`seems` scored as the action. Two matchers asking the same question, one
looser, is the `one-canonical-way` shape exactly; they are now one function,
`_denotes`.

**3. A duplicated fact had drifted.** `grounded_corpus.REFERENT_FEATURES`
listed `table` as `[TABLE, OBJECT]` while the corpus grounds it
`[TABLE, FURNITURE]` — and the file's own comment said this must not happen,
because "a mismatch would make `role_of_features` silently return None".
`check()` still read 198/198, because partial overlap on `TABLE` alone was
enough to win. Tightening the matcher exposed it as 180/198. The table is
deleted: participants are read from the grounding through the same accessor the
derivation uses, so the two cannot disagree at all. Back to 198/198, now by
construction rather than by coincidence.

## The result that matters

After those fixes, on clauses the positional inducer can analyse:

| stage | agreement |
|---|---|
| `SENTENCES` | **120/120 (1.0000)** |
| `COMPLEX_GRAMMAR` | 113/116 (0.9741) |

And **the surviving disagreements are the positional inducer's errors, not the
scene's**:

```
the open beach meets the street off the couch   `open` classifies as VERB, so
the closed lamp seems towards the fish          the inducer splits the clause
                                                at the ADJECTIVE and calls the
                                                subject a post-verb PATIENT
```

The scene says `beach` and `lamp` are the agents. They are. This is the first
measured evidence that the scene route is not merely non-circular but *more
accurate* than the route the curriculum has been using — on ACTIVE sentences,
where positional induction is supposed to be at its best.

The third, `the excellent bear arrives inside the monkey`, is a genuine limit
rather than a bug: `bear` and `monkey` carry identical feature bundles at this
grounding resolution, so a learner perceiving only these features cannot tell
which one acted. `roles_from_scene` returning the wrong one there is honest
about the resolution; it is not honest about the tie, and that is a real
weakness of scoring by best overlap rather than by identity.

## Two corpus defects found on the way

* The locative ground could be the SAME noun as a participant — "the library
  sleeps among the library". One referent in a thematic role and a PP at once
  is unanalysable, not merely odd. Excluded.
* Several adjectives in the lexicon are verb-ambiguous (`open`, `closed`,
  `first`), and the generator inserts adjectives at complexity >= 5. This is
  not fixed. It makes most `COMPLEX_GRAMMAR` clauses unanalysable by the
  positional inducer, which is why the agreement test accumulates across stages
  rather than being parametrized — `COMPLEX_GRAMMAR` alone can yield ZERO
  comparable clauses, and a test that silently finds nothing to check is not a
  test.

## The thread woke three training paths that had NEVER run

This is the largest finding here, and I did not predict it. Three routines
select their inputs by role:

| routine | selector | ran on the curriculum? |
|---|---|---|
| `train_phrases` | `role == "agent"` / `"action"` | **no** |
| `build_role_pathway_protocol` | `role in (agent, patient)` | **no** |
| `build_vp_pathway_protocol` | `role == "agent"` etc. | **no** |

With `roles=[None]` every selector matched nothing, so all three compiled zero
work and returned silently. The `phrases` phase is listed in six of the nine
stages and was doing nothing in all of them. Waking them broke three tests, and
each break was informative.

### The break that was a real bug: consolidation destroys the lexicon

`consolidate()` is documented as replaying "without resetting area
connections" — true of connections, false of the INDEX SPACE. Every step ran
`prepare_area_for_replay` on its SOURCE area, which rewinds `w` and clears
`compact_to_neuron_id`. For a role replay the source IS a core area, holding
the stabilized lexicon. Measured at `DIALOGUE`:

```
NOUN_CORE  w 2493 -> 976   48 of 74 stored nouns unmappable
VERB_CORE  w 2248 -> 890   34 of 44 stored verbs unmappable
```

so a later parse raised `Assembly neuron 2887 not in area 'NOUN_CORE' mapping
(len=976)`. `consolidation.py` already documents this exact hazard
("Re-snapshot after consolidation; do not carry pre-consolidation lexicons
across") and ships `drop_stale_assemblies` for it — called by exactly one of
the two consolidation call sites. But dropping is not a fix here, it is a
description of the damage: it would delete two thirds of the lexicon.

The fix is `prepare_areas=False`, which makes "without reset" true. Preparation
exists for replay onto a connectome that was just CLEARED; these replay onto a
live one. Afterwards: **0 of 74 stale**, and `NOUN_CORE` *grows* to 2657 —
consolidation still does work, it just no longer destroys what it replays.

All four `consolidate_*_pathways` wrappers said "without reset" and all four
prepared; all four are fixed, because a fix applied to one sibling is the
signature this repo keeps paying for.

### The break that was a bad assertion

`test_calibration_separates_category_violation_from_grammatical` asserted
`catv["p600_excess_median"] > gram[...]`. `p600_excess` is
`max(0, deficit_raw - baseline_median)` — clipped at its own null — and
`calibration.py`'s own docstring says "roughly half the mass sits exactly at 0
and it is not an effect size", while the AUC on the raw deficit "IS THE ONE TO
READ". Measured, same probe, before and after:

| statistic | before | after |
|---|---|---|
| `p600_auc` | 1.000 | **1.000** |
| `n400_auc` | 0.889 | **1.000** |
| `p600_excess_median` (catv) | 0.0016 | 0.0 |

The separation is perfect in both, and *improved* on N400. Only the clipped
statistic moved, because both arms now land on the clip point. The strict `>`
was testing where the clip fell. It is now `>=` (the effect must not invert)
plus the AUC, which is the assertion that carries the claim.

### The break that was an improvement

`test_the_violation_arm_is_alive_too` was a `strict` xfail: VP read *exactly*
0.000000 with 30 winners, and the recorded diagnosis blamed a self-fiber whose
block spans `w` rather than `n`, concluding "fixing it means an engine change".
That diagnosis named the wrong mechanism. VP was empty because `train_phrases`
never ran:

```
VP  self-fiber extent 2061   materialized 2458   winners 30
    energy 0.001172          vp_assemblies stored 72
```

Alive, not equal — `ROLE_PATIENT` reads 0.010442, so VP is still ~9x weaker.
The arm is no longer structurally empty; the two arms are not now comparable in
strength, and #104's saturation question is untouched.

## One regression fixed that this arc did not cause

`evaluation/parity.py` registers stage lemmas but never called
`_register_surface_forms`. Since the grammaticality fix the generator emits
finite forms (`builds`), and `compile_corpus` skips tokens missing from
`stim_map` — so every verb in that stage was being silently dropped. Introduced
by my own previous commit, in exactly the silent-drop class the helper was
written to close. One registration function, now called by both paths.

## Limits

* Passives are still NOT generated. This note covers only the thread and the
  matcher; the voice alternation is the next step and nothing here demonstrates
  it works.
* Everything the three woken paths now do is UNMEASURED beyond "it runs and
  nothing crashes". `train_phrases` performing 72 merges is not evidence that
  phrase structure is good — #31 recorded that VP merges collapse onto one
  assembly, and that question is now live for the first time on this path
  rather than settled.
* The `prepare_areas=False` change alters consolidation for every caller of
  the four wrappers, including paths this arc never exercised.
* `_learn_gating_patterns` is called from `train_roles`, which does not run on
  the curriculum path — the roles now arriving are consumed by
  `train_unsupervised`. Threading roles was necessary for voice gating but is
  not sufficient, and that gap is measured, not assumed.
* Agreement is measured against the positional inducer, which the same data
  shows to be wrong in some cases. It is a cross-check, not a ground truth.
