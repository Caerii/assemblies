# The corpus now varies on every axis the census found constant

## Why

The grammar-gap census (the measurement that motivated this) found the
generated corpus was one tense, one number, one determiner, adverb-free,
pronoun-free: **a cue that never varies carries no information**, so agreement,
tense and determiner statistics were unlearnable however long anything trained.
The user's direction: make the synthetic corpus more realistic before reaching
for CHILDES.

## What was added — each licensed by the lexicon, none hand-listed

| dimension | rate | lexicon source |
|---|---|---|
| past tense | ~55% of frames carry past/ppart | `forms["past"]`, all 119+ verbs |
| plural subjects | ~25% | `forms["plural"]`, 108/116 nouns; plural-present agreement is the bare lemma, past is number-invariant |
| determiner variety | 10 types (the ~70%, possessives, a/an) | `features.definite/possessive`, a/an by the following token's initial |
| pronoun subjects | ~13% | grounded 3rd person (he/she/they), animate-safe by gender/number features |
| manner adverbs | ~17%, pre-verbal | `features.manner` |
| passive tense | was/is consistently | `be.forms["past"/"3sg"]`, aux agrees with the passive subject |

Measured after (seed 42): SENTENCES n=58 — past 31, plural 14, pronoun 7,
adverb 9, passive 8; COMPLEX_GRAMMAR similar. Grammaticality audit: **0
violations** in all three stages, under the UPDATED contract (licensed
agreement is now a set: past always, 3sg for singular, bare lemma for plural;
animacy admits personal pronouns by gender/number).

Sample of what the corpus now sounds like:

```
your girls grew the boy          his books walk
the young worlds fast play the cloud behind the church
the cat was chased by the dog    she holds the ball
```

## Three defects caught by the existing guards, none by inspection

1. **The lemma-identity guard had a surface hole.** With plural subjects,
   *"her sharp couches hold your brain between the couch"* put the SAME
   referent in a thematic role and the locative PP — the exact "library sleeps
   among the library" unanalysable shape — because the exclusion checked
   surface tokens and `couches` ≠ `couch`. Caught by the scene-vs-position
   agreement test (1/136 disagreement); fixed by excluding on lemmas.
2. **`whose` is possessive AND interrogative**, so the feature-driven
   determiner pool emitted *"whose girls grew the boy"* — a question wearing a
   declarative frame. Caught by the variation census (not by the audit — mood
   is not one of its three checks); excluded by the `interrogative` feature.
3. **Pronoun frames can be underivable**: a pronoun's features are thin, so
   against certain nouns the scene reference test ties and roles come back
   None. Guarded at generation: a pronoun frame is emitted only if
   `roles_from_scene` would recover every role — the wellformedness bar the
   passive arc established, applied before anything trains.

## One test moved, and why that is correct

`test_active_voice_does_not_regress` failed — but through
`_assign_roles_neural`, the DEMOTED margin route, which the test predates: with
the richer corpus, `ball` now occurs as a subject, its word-level role
statistic moved, and the margin flipped an active sentence. **Production was
never wrong** — `parse_roles_by_reconstruction` passes the same sentence in the
same run. The test now exercises the production route; pinning the demoted
instrument's word statistics would pin exactly the quantity the demotion
retired.

That failure is also a small confirmation of the whole #33 arc: the margin
route degrades as the corpus gets more realistic (word-role statistics get less
deterministic), while the reconstruction route does not care.

## Limits

* Rates are stated choices, not fitted to corpora; frequencies are still
  uniform per pool, not Zipfian — Zipf is its own measured arc
  (`zipf-gain-must-come-from-corpus`: it CLOSES composition without the g
  compensation, so it must not be sneaked in as "realism").
* Object number is not varied (passive aux number therefore never needs
  are/were). Ditransitives/ROLE_GOAL (#116), questions/imperatives
  (multi-mood), and relative clauses in the GENERATED corpus remain absent.
* Semantic plausibility is still unchecked: "his books walk" is licensed,
  grammatical, and silly.
* What the new variation BUYS downstream (tense/number detection accuracy,
  agreement as a cue) is not yet measured — this note claims only that the
  cues now vary and nothing regressed.
