# The generated training corpus was not English — and fixing it repaired the golden

## What the generator emitted

`CurriculumTrainer._generate_sentences_generic` built every sentence as

```python
sent = [det, random_noun, verb.lemma]          # no agreement
if complexity >= 4:
    sent += [det, random_noun]                 # object regardless of transitivity
```

with subject and object drawn **uniformly over all nouns**. Output:

```
the store build the dog
```

Three independent defects, none of which needed new data to fix — the lexicon
already carried the answer to all three and none of it was being read:

| defect | what it produced | the lexicon field ignored |
|---|---|---|
| no subject–verb agreement | "the store **build**" | `forms["3sg"]` — present on **all 136 verbs** |
| object forced on any verb | "the dog **runs the house**" | `features.transitive / intransitive / ambitransitive` |
| no selectional restriction | "**the store** builds" | `features.animate / abstract`, `arguments[0]` |

## Why it mattered more than it looks

**Role induction learns AGENT/PATIENT from noun position relative to the verb**,
so this corpus *is* the supervision. Garbage frames produce garbage bindings.

`dog` occurred **exactly once** in the whole role-training corpus — as the object
of "the store build the dog" — so `ROLE_PATIENT` was its only binding. That is
why `nemo2025_curriculum` asserted `dog = AGENT` and could never get it, and why
improving the substrate (β, `phon_weight`) moved the metric by exactly zero: it
was retrieving a correct binding of a wrong fact.

## The fix

Read the lexicon that was already there:

- finite form via `forms["3sg"]`, since every generated subject is `the <noun>`
- object **iff** the verb takes one; `ambitransitive` alternates
- agent/experiencer subjects must be `animate`; abstract nouns excluded from
  concrete slots
- copulas excluded — this frame builds no predicate, so `be` could only ever
  yield "the dog is"
- at complexity 3 (no object slot) prefer verbs that can stand alone
- locative PPs restricted to **static spatial** prepositions via
  `features.spatial and not motion/goal/source` — data-driven, not a hand list;
  this is what stopped "plays the paper **away the food**"

Inflected forms are now registered with **the lemma's grounding**, because
`compile_corpus` skips any token missing from `stim_map` — without that, every
finite verb would have been silently dropped from role training.

## Measured, not eyeballed

`research/experiments/curriculum_corpus_grammaticality.py` checks the three
properties mechanically against the lexicon:

| stage | sentences | violations |
|---|---|---|
| TWO_WORD | 50 | **0** |
| SENTENCES | 50 | **0** |
| COMPLEX_GRAMMAR | 50 | **0** |

```
the person speaks          the dad runs
the girl grows             the boy listens
```

**One of my own audit findings was a false positive** and worth recording: the
checker first flagged 15 "objects on intransitive verbs" like *"the boy sleeps
in the house"* — it was counting the **PP object** as a direct object. That is an
audit bug, and "fixing" the generator against it would have made the corpus
worse. Nouns governed by a preposition are now excluded.

## The payoff

```
role_updates for 'dog':  {ROLE_PATIENT: 1}   ->   {ROLE_AGENT: 1}
    "the store build the dog"                     "the dog says the cat"

TestNemo2025CurriculumGolden:  FAILED (0.6667)  ->  PASSED (1.0)
```

The golden passes **at its original 1.0 threshold**. It was never re-recorded
downward — the claim was correct all along and the training data was wrong.

## Limits and what is still not checked

- **Semantic plausibility is out of scope and stated as such.** "the dog says
  the cat" and "the kind world plays the paper off the food" pass every check
  here. They are grammatical; they are not sensible. Selectional restriction is
  enforced only for animacy of agents.
- Only three properties are checked. Determiner–noun number agreement, mass vs
  count nouns ("the water"), and tense consistency are not.
- This changes the training corpus for **every** curriculum run, so any golden
  recorded before it describes a different corpus.
