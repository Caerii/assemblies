# Passives cannot be added to the corpus alone — the blocker is `roles=[None]`

## What I set out to do

Add passive voice, per the measured gap: the corpus is 100% active SVO, so
position predicts role perfectly and a role representation buys nothing over a
position counter.

## Why generating passives ALONE would make things worse

Two independent role paths read the corpus, and they disagree about passives.

**1. The unsupervised inducer is purely positional.**
`corpus_index._assign_noun_roles` maps nouns onto typology slots split at the
verb: pre-verb → `S` → `ROLE_AGENT`, post-verb → `O` → `ROLE_PATIENT`. For

```
the cat was chased by the dog
```

that yields `cat = AGENT` and `dog = PATIENT` — **both inverted**. Adding
passives without touching this path would inject systematically wrong labels
into role training, which is worse than the gap it closes.

**2. The parser's voice gating is LEARNED, and deliberately so.**
`_determine_role_order` flips the role ranking only if `learned_gating` says a
function-word subcategory reverses roles. Its docstring is explicit:

> There is deliberately no spelling-based passive fallback: if the corpus
> contained no voice alternation, the parser has not learned one and should not
> pretend otherwise.

That is the right design, and it means the flip has to be *taught*.

## What teaches it, and why the corpus can't

`train_gating` learns contrastively: for each function-word subcategory it
compares `P(patient-first | subcat present)` against `P(subcat absent)`, so a
word occurring in both voices (like "the") correctly scores ~0 and only a true
marker crosses threshold. Good mechanism.

But it reads `sent.roles`, and **`TrainingScheduleExecutor.build_stage_schedule`
hardcodes**:

```python
roles=[None] * len(sent)
```

So `train_gating` sees no role on any curriculum sentence and can learn nothing.
The passive branch is dormant not because the corpus lacks passives but because
the pipeline discards the only signal that could label them.

## The deeper point: roles cannot come from the text

This is not a plumbing detail. **You cannot learn that "by" reverses roles from
raw text alone** — the reversal is precisely the case where surface order stops
predicting the answer. The label has to come from outside the string.

The repo already established this: role induction from raw text is circular, and
the non-circular signal is the perceived event (`grounded-role-learning`, where
word order was induced 6/6 with text annotations deleted but the scene kept).

So passives must be generated as **active/passive pairs describing the SAME
event**, with roles taken from the event:

```
event(agent=dog, patient=cat, action=chase)
  ->  "the dog chases the cat"            roles: [None, agent, None, None, patient]
  ->  "the cat is chased by the dog"      roles: [None, patient, None, None, None, agent]
```

Same underlying roles, opposite surface order. That is exactly the contrast
`train_gating` needs, and it is honest: the supervision is the scene, not the
sentence.

## Consequence for the plan

The generator must return **(tokens, roles)**, not tokens, and
`build_stage_schedule` must carry the roles through instead of discarding them.
That is a signature change across `_generate_sentences_generic` →
`_generate_sentences` → `build_stage_schedule` → `GroundedSentence`.

**I did not start it.** Shipping passive generation without the role thread
would inject inverted labels; shipping half a signature migration would leave
the repo worse than either end state. The sequencing is: thread roles first,
then generate passives, then verify.

## Verification the next pass must do, in order

1. `roles` actually arrive: count non-`None` roles reaching `train_gating`.
   Zero means the thread is still broken and everything after is void.
2. Passive wellformedness BEFORE any is trained on: aux + `ppart` + `by` +
   animate agent, checked in `test_corpus_grammaticality.py` — the same guard
   that would have caught "the store build the dog".
3. `train_gating` learns `MARKER` reverses roles: `confidence > 0` and
   `reverses_roles` True, with `n_contrast > 0` so it is contrastive rather
   than "present in every sentence".
4. `is_passive` actually fires during a parse — **count it, do not assume**.
5. Role assignment on a passive puts the by-phrase noun in AGENT.
6. Active-voice role accuracy does NOT regress — the failure mode is teaching
   the determiner to reverse everything.

## Limits

The active/passive pairing means the corpus states each event twice, which
changes token frequencies and therefore `adaptive_rounds`. Any capacity or
frequency result measured before this describes a different corpus.
